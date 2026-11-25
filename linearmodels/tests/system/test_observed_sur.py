"""
Tests for SUR estimation on partially observed dependent data
(``missing="observed"``).
"""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.shared.hypotheses import InvalidTestStatistic
from linearmodels.system import SUR
from linearmodels.system.ml import (
    ObservedSURConvergenceError,
    ObservedSURError,
    _ObservedLikelihood,
    _build_patterns,
    _ols_start,
    _initial_sigma,
    check_known_sigma,
)

RTOL = 1e-5
ATOL = 1e-7


def generate(n=600, k=3, miss=0.3, seed=0, px=None):
    rng = np.random.default_rng(seed)
    if px is None:
        px = (2, 3, 2)[:k] if k <= 3 else tuple([2] * k)
    a = rng.standard_normal((k, k))
    sigma = a @ a.T / k + np.eye(k)
    err = rng.standard_normal((n, k)) @ np.linalg.cholesky(sigma).T
    xs, ys, betas = [], [], []
    for i in range(k):
        p = px[i] if isinstance(px, (tuple, list)) else px
        x = np.column_stack([np.ones(n), rng.standard_normal((n, p))])
        b = rng.standard_normal((p + 1, 1))
        xs.append(x)
        ys.append(x @ b + err[:, [i]])
        betas.append(b)
    y_mis = [y.copy() for y in ys]
    if miss:
        m = rng.random((n, k)) < miss
        for i in range(k):
            y_mis[i][m[:, i]] = np.nan
    for rr in (0, n - 1):
        for i in range(k):
            y_mis[i][rr] = np.nan
    obs = np.isfinite(np.hstack(y_mis))
    return xs, y_mis, obs, sigma, betas


def equations(xs, ys):
    return {f"eq{i}": {"dependent": ys[i], "exog": xs[i]} for i in range(len(xs))}


@pytest.fixture(scope="module")
def data3():
    return generate()


def test_smoke_unknown_sigma(data3):
    xs, ys, obs, _, _ = data3
    res = SUR(equations(xs, ys), missing="observed").fit(method="ml")
    assert res.converged
    assert np.isfinite(res.loglik)
    assert res.nobs == int(obs.sum())
    assert list(res.nobs_per_equation) == list(obs.sum(0))
    assert res.known_sigma is False
    assert res.method == "ML (observed data)"
    # residuals retain every original row with NaN in unobserved cells
    rdf = res.resids
    assert rdf.shape == (obs.shape[0], obs.shape[1])
    assert np.array_equal(rdf.isna().values, ~obs)
    # fitted values are defined everywhere
    assert not res.fitted_values.isna().any().any()


def test_smoke_known_sigma(data3):
    xs, ys, _obs, sigma, _ = data3
    res = SUR(equations(xs, ys), sigma=sigma, missing="observed").fit(
        method="ml", cov_type="unadjusted"
    )
    assert res.known_sigma
    assert_allclose(np.asarray(res.sigma), sigma, rtol=RTOL, atol=ATOL)
    assert res.information.shape[0] == len(res.params)
    assert res.iterations == 0


def test_known_sigma_must_be_pd():
    xs, ys, _, _, _ = generate(k=2)
    with pytest.raises(ObservedSURError, match="positive definite"):
        check_known_sigma(np.zeros((2, 2)), 2)
    bad = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ObservedSURError, match="positive definite"):
        check_known_sigma(bad, 2)
    with pytest.raises(ObservedSURError, match="symmetric"):
        check_known_sigma(bad + np.triu(np.ones((2, 2)), 1), 2)
    with pytest.raises(ObservedSURError, match="finite"):
        check_known_sigma(np.full((2, 2), np.inf), 2)
    with pytest.raises(ObservedSURError, match="dimensions"):
        check_known_sigma(np.eye(3), 2)
    with pytest.raises(ObservedSURError):
        SUR(equations(xs, ys), sigma=np.zeros((2, 2)), missing="observed")


def test_only_ml_method(data3):
    xs, ys, _, _, _ = data3
    mod = SUR(equations(xs, ys), missing="observed")
    with pytest.raises(ObservedSURError, match="only support"):
        mod.fit(method="gls")
    with pytest.raises(ObservedSURError, match="only support"):
        mod.fit()
    with pytest.raises(ValueError, match="supports cov_type"):
        mod.fit(method="ml", cov_type="kernel")
    with pytest.raises(ValueError, match="supports cov_type"):
        mod.fit(method="ml", cov_type="clustered")
    with pytest.raises(ValueError, match="Debiased"):
        mod.fit(method="ml", debiased=True)


def test_drop_mode_unchanged():
    # The default path must retain the original OLS/GLS behavior.
    xs, ys, obs, _, _ = generate(k=2, miss=0.2)
    mod_default = SUR(equations(xs, ys))
    mod_drop = SUR(equations(xs, ys), missing="drop")
    assert mod_default.missing == "drop"
    assert mod_drop.missing == "drop"
    r1 = mod_default.fit()
    r2 = mod_drop.fit()
    assert_allclose(r1.params, r2.params)
    with pytest.raises(AttributeError):
        mod_default.observed_mask  # noqa: B018
    # Drop mode only keeps rows complete in every equation; the system nobs
    # counts the retained cells (rows times equations).
    complete = obs.all(axis=1).sum()
    assert r1.resids.shape[0] == complete
    assert r1.nobs == complete * obs.shape[1]
    # Observed mode retains more information and a distinct nobs count.
    r_obs = SUR(equations(xs, ys), missing="observed").fit(method="ml")
    assert r_obs.nobs == int(obs.sum()) > r1.nobs
    assert r_obs.resids.shape[0] == obs.shape[0]


def test_invalid_missing_value():
    xs, ys, _, _, _ = generate(k=2)
    with pytest.raises(ValueError, match="missing must be"):
        SUR(equations(xs, ys), missing="banana")


def test_reject_weights():
    xs, ys, _, _, _ = generate(k=2)
    eqns = equations(xs, ys)
    for key in eqns:
        eqns[key]["weights"] = np.ones((len(ys[0]), 1))
    with pytest.raises(ObservedSURError, match=r"[Ww]eights"):
        SUR(eqns, missing="observed")
    # tuple form with weights
    eqns2 = {f"eq{i}": (ys[i], xs[i], np.ones((len(ys[0]), 1))) for i in range(2)}
    with pytest.raises(ObservedSURError, match=r"[Ww]eights"):
        SUR(eqns2, missing="observed")


def test_reject_constraints(data3):
    xs, ys, _, _, _ = data3
    mod = SUR(equations(xs, ys), missing="observed")
    r = pd.DataFrame(np.zeros((1, len(mod.param_names))), columns=mod.param_names)
    r.iloc[0, 0] = 1.0
    with pytest.raises(ObservedSURError, match=r"[Cc]onstraint"):
        mod.add_constraints(r)


def test_dependent_inf_rejected():
    xs, ys, _, _, _ = generate(k=2)
    ys[1][3, 0] = np.inf
    with pytest.raises(ObservedSURError, match=r"[Ii]nfinite"):
        SUR(equations(xs, ys), missing="observed")
    ys[1][3, 0] = -np.inf
    with pytest.raises(ObservedSURError, match=r"[Ii]nfinite"):
        SUR(equations(xs, ys), missing="observed")


def test_exog_missing_rejected_even_when_dep_missing():
    xs, ys, _, _, _ = generate(k=2)
    xs[0][5, 1] = np.nan
    ys[0][5, 0] = np.nan  # dependent also missing there
    with pytest.raises(ObservedSURError, match="missing or infinite"):
        SUR(equations(xs, ys), missing="observed")
    xs[0][5, 1] = np.inf
    with pytest.raises(ObservedSURError, match="missing or infinite"):
        SUR(equations(xs, ys), missing="observed")


def test_pandas_index_requirements():
    xs, ys, _, _, _ = generate(k=2)
    n = len(ys[0])
    idx = pd.date_range("2020", periods=n)
    # duplicate index
    dup_values = list(idx[: n - 1]) + [idx[0]]
    dup_idx = pd.Index(dup_values)
    assert not dup_idx.is_unique
    eqns = {
        f"eq{i}": {
            "dependent": pd.DataFrame(ys[i], index=dup_idx, columns=["y"]),
            "exog": pd.DataFrame(
                xs[i],
                index=dup_idx,
                columns=[f"x{j}" for j in range(xs[i].shape[1])],
            ),
        }
        for i in range(2)
    }
    with pytest.raises(ObservedSURError, match="duplicate"):
        SUR(eqns, missing="observed")
    # mismatched indexes within an equation
    eqns = {
        f"eq{i}": {
            "dependent": pd.DataFrame(ys[i], index=idx, columns=["y"]),
            "exog": pd.DataFrame(
                xs[i],
                index=range(n),
                columns=[f"x{j}" for j in range(xs[i].shape[1])],
            ),
        }
        for i in range(2)
    }
    with pytest.raises(ObservedSURError, match="index"):
        SUR(eqns, missing="observed")
    # mismatched indexes across equations
    eqns = {}
    for i in range(2):
        eqns[f"eq{i}"] = {
            "dependent": pd.DataFrame(
                ys[i], index=idx if i == 0 else idx[::-1], columns=["y"]
            ),
            "exog": pd.DataFrame(
                xs[i],
                index=idx if i == 0 else idx[::-1],
                columns=[f"x{j}" for j in range(xs[i].shape[1])],
            ),
        }
    with pytest.raises(ObservedSURError, match="index"):
        SUR(eqns, missing="observed")


def test_length_mismatch():
    xs, ys, _, _, _ = generate(k=2)
    eqns = {
        "eq0": {"dependent": ys[0], "exog": xs[0]},
        "eq1": {"dependent": ys[1][:10], "exog": xs[1][:10]},
    }
    with pytest.raises(ObservedSURError, match="same number of observations"):
        SUR(eqns, missing="observed")


def test_equation_without_observations():
    xs, ys, _, _, _ = generate(k=2)
    ys[1][:] = np.nan
    with pytest.raises(ObservedSURError, match="no observed"):
        SUR(equations(xs, ys), missing="observed")


def test_rank_deficient_observed_design():
    xs, ys, _, _, _ = generate(k=2)
    xs[0][:, 2] = xs[0][:, 1]
    with pytest.raises(ObservedSURError, match="full rank"):
        SUR(equations(xs, ys), missing="observed")


def test_no_joint_overlap_unknown_sigma():
    n = 400
    xs, ys, _, sigma, _ = generate(n=n, k=2, miss=0.0)
    # remove the injected empty rows first
    for i in range(2):
        ys[i][0] = 1.0
        ys[i][n - 1] = 1.0
    ys[0][: n // 2] = np.nan
    ys[1][n // 2 :] = np.nan
    with pytest.raises(ObservedSURError, match="never jointly observed"):
        SUR(equations(xs, ys), missing="observed").fit(method="ml")
    # a known sigma makes the design estimable
    res = SUR(equations(xs, ys), sigma=sigma, missing="observed").fit(method="ml")
    assert np.isfinite(res.loglik)


def test_no_complete_rows_but_identified():
    """No row observes every equation, yet every pair overlaps somewhere."""
    n = 600
    xs, ys, _, sigma, _ = generate(n=n, k=3, miss=0.0)
    for i in range(3):
        ys[i][0] = 1.0
        ys[i][n - 1] = 1.0
    rng = np.random.default_rng(0)
    pattern = rng.integers(0, 3, size=n)  # each row drops one equation
    for row, drop in enumerate(pattern):
        ys[int(drop)][row] = np.nan
    res = SUR(equations(xs, ys), missing="observed").fit(method="ml")
    assert res.converged
    assert res.nobs == 2 * n
    assert_allclose(np.asarray(res.sigma), sigma, rtol=0.2, atol=0.2)


def test_all_empty_rows_retained():
    xs, ys, obs, _, _ = generate(k=2)
    # force additional fully empty rows
    for rr in (10, 11, 12):
        for i in range(2):
            ys[i][rr] = np.nan
    obs = np.isfinite(np.hstack(ys))
    mod = SUR(equations(xs, ys), missing="observed")
    res = mod.fit(method="ml")
    assert set(res.empty_rows["row"].values) <= set(mod.observed_mask.index)
    # the empty rows remain in every output layout
    assert res.resids.shape[0] == obs.shape[0]
    assert res.fitted_values.shape[0] == obs.shape[0]
    assert res.resids.iloc[10:13].isna().all().all()
    assert not res.fitted_values.iloc[10:13].isna().any().any()


def test_iteration_budget_exhausted():
    xs, ys, _, _, _ = generate(n=500, k=4, seed=3)
    with pytest.raises(ObservedSURConvergenceError):
        SUR(equations(xs, ys), missing="observed").fit(method="ml", iter_limit=1)


def test_tol_must_be_positive(data3):
    xs, ys, _, _, _ = data3
    with pytest.raises(ValueError, match="tol"):
        SUR(equations(xs, ys), missing="observed").fit(method="ml", tol=0.0)
    with pytest.raises(ValueError, match="iter_limit"):
        SUR(equations(xs, ys), missing="observed").fit(method="ml", iter_limit=0)


def test_gradient_and_information_consistency(data3):
    xs, ys, _obs, _, _ = data3
    lik = _ObservedLikelihood(xs, ys, _obs, _build_patterns(_obs))
    b = _ols_start(lik)
    s = _initial_sigma(lik, b)
    # row scores sum to the total analytic score
    _, gb, gv = lik.loglik_grad(b, s)
    rs, rows = lik.row_scores(b, s)
    assert_allclose(rs.sum(0), np.r_[gb, gv], rtol=RTOL, atol=ATOL)
    assert (rows == np.sort(rows)).all()
    # analytic information matches finite differences
    h_analytic = lik.observed_information(b, s)
    h_fd = lik.numerical_hessian(b, s, eps_fd=1e-7)
    assert_allclose(h_analytic, h_fd, rtol=1e-5, atol=1e-5)
    # known-sigma information covers beta only
    h_beta = lik.observed_information(b, s, include_sigma=False)
    assert h_beta.shape == (lik.n_params, lik.n_params)


def test_covariance_estimators_agree_with_likelihood(data3):
    xs, ys, _obs, _, _ = data3
    mod = SUR(equations(xs, ys), missing="observed")
    res_u = mod.fit(method="ml", cov_type="unadjusted")
    res_r = mod.fit(method="ml", cov_type="robust")
    lik = mod._fit_engine_likelihood
    beta = np.asarray(res_u.params)[:, None]
    sigma = np.asarray(res_u.sigma)
    info = lik.observed_information(beta, sigma)
    info_inv = np.linalg.inv(info)
    p = beta.shape[0]
    assert_allclose(np.asarray(res_u.cov), info_inv[:p, :p], rtol=RTOL, atol=ATOL)
    rs, _ = lik.row_scores(beta, sigma)
    sand = info_inv @ rs.T @ rs @ info_inv
    sand = (sand + sand.T) / 2
    assert_allclose(np.asarray(res_r.cov), sand[:p, :p], rtol=RTOL, atol=ATOL)
    # both covariances are positive definite
    assert np.linalg.eigvalsh(np.asarray(res_u.cov)).min() > 0
    assert np.linalg.eigvalsh(np.asarray(res_r.cov)).min() > 0


def test_known_sigma_covariance_beta_only(data3):
    xs, ys, _, sigma, _ = data3
    res = SUR(equations(xs, ys), sigma=sigma, missing="observed").fit(
        method="ml", cov_type="unadjusted"
    )
    lik = res.model._fit_engine_likelihood
    info = lik.observed_information(
        np.asarray(res.params)[:, None], sigma, include_sigma=False
    )
    assert_allclose(np.asarray(res.cov), np.linalg.inv(info), rtol=RTOL, atol=ATOL)


def test_results_layout_and_equation_stats(data3):
    xs, ys, obs, _, _ = data3
    res = SUR(equations(xs, ys), missing="observed").fit(method="ml")
    # per-equation nobs and parameter/SE consistency
    for i, label in enumerate(res.equation_labels):
        eq = res.equations[label]
        assert eq.nobs == int(obs[:, i].sum())
        assert eq.resids.shape[0] == obs.shape[0]
        assert eq.fitted_values.shape[0] == obs.shape[0]
        assert eq.f_statistic is not None
    # system nobs is the sum across equations
    assert res.nobs == sum(eq.nobs for eq in res.equations.values())
    # parameter names and order preserved
    assert len(res.params) == len(res.param_names)
    assert len(res.std_errors) == len(res.params)
    ci = res.conf_int()
    assert ci.shape == (len(res.params), 2)
    assert (ci["upper"] >= ci["lower"]).all()


def test_predict_in_and_out_of_sample(data3):
    xs, ys, obs, _, _ = data3
    n = obs.shape[0]
    idx = pd.date_range("2020-01-01", periods=n)
    eqns = {
        f"eq{i}": {
            "dependent": pd.DataFrame(ys[i], index=idx, columns=["y"]),
            "exog": pd.DataFrame(
                xs[i],
                index=idx,
                columns=[f"x{j}" for j in range(xs[i].shape[1])],
            ),
        }
        for i in range(3)
    }
    res = SUR(eqns, missing="observed").fit(method="ml")
    # in-sample, full original layout
    pred = res.predict(dataframe=True, missing=True)
    assert pred.shape == (n, 3)
    assert pred.index.equals(idx)
    # idiosyncratic residuals keep NaNs
    ids = res.predict(idiosyncratic=True, fitted=False, dataframe=True, missing=True)
    assert np.array_equal(ids.isna().values, ~obs)
    # out-of-sample, equation subset
    new_idx = pd.date_range("2022-01-01", periods=8)
    new_eqns = {
        "eq0": {
            "exog": pd.DataFrame(
                xs[0][:8],
                index=new_idx,
                columns=[f"x{j}" for j in range(xs[0].shape[1])],
            )
        }
    }
    oos = res.predict(equations=new_eqns, dataframe=True)
    assert oos.shape == (8, 1)
    assert oos.index.equals(new_idx)


def test_unavailable_rectangular_tests(data3):
    xs, ys, _, _, _ = data3
    res = SUR(equations(xs, ys), missing="observed").fit(method="ml")
    bp = res.breusch_pagan()
    lr = res.likelihood_ratio()
    assert isinstance(bp, InvalidTestStatistic)
    assert isinstance(lr, InvalidTestStatistic)
    assert np.isnan(res.system_rsquared).all()


def test_summary_renders(data3):
    xs, ys, _, _, _ = data3
    res = SUR(equations(xs, ys), missing="observed").fit(method="ml")
    text = res.summary.as_csv()
    assert "ML (observed data)" in text
    assert "Log-likelihood" in text
    assert "Termination" in text
    # equation-level summary
    eq = res.equations[res.equation_labels[0]]
    eq.summary.as_csv()


def test_equation_reordering_equivalence():
    xs, ys, _, _, _ = generate(k=3, seed=11)
    r1 = SUR(equations(xs, ys), missing="observed").fit(method="ml")
    perm = [2, 0, 1]
    eqns2 = {
        f"eq{perm[i]}": {"dependent": ys[perm[i]], "exog": xs[perm[i]]}
        for i in range(3)
    }
    r2 = SUR(eqns2, missing="observed").fit(method="ml")
    assert_allclose(r1.loglik, r2.loglik, rtol=RTOL, atol=ATOL)
    for old in range(3):
        assert_allclose(
            r1.equations[f"eq{old}"].params,
            r2.equations[f"eq{old}"].params,
            rtol=RTOL,
            atol=ATOL,
        )


def test_row_reordering_equivalence():
    xs, ys, obs, _, _ = generate(k=2, seed=12)
    r1 = SUR(equations(xs, ys), missing="observed").fit(method="ml")
    perm = np.random.default_rng(0).permutation(obs.shape[0])
    xs2 = [x[perm] for x in xs]
    ys2 = [y[perm] for y in ys]
    r2 = SUR(equations(xs2, ys2), missing="observed").fit(method="ml")
    assert_allclose(r1.loglik, r2.loglik, rtol=RTOL, atol=ATOL)
    assert_allclose(np.asarray(r1.sigma), np.asarray(r2.sigma), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("k", [1, 2, 4])
def test_dimensions(k):
    xs, ys, _obs, _sigma, _ = generate(k=k, seed=20 + k)
    res = SUR(equations(xs, ys), missing="observed").fit(method="ml")
    assert res.sigma.shape == (k, k)
    assert res.resids.shape[1] == k
    assert len(res.equation_labels) == k


def test_complete_data_matches_alternating_ml():
    rng = np.random.default_rng(30)
    n, k = 800, 3
    a = rng.standard_normal((k, k))
    sigma0 = a @ a.T / k + np.eye(k)
    err = rng.standard_normal((n, k)) @ np.linalg.cholesky(sigma0).T
    xs, ys = [], []
    for i in range(k):
        x = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
        b = rng.standard_normal((3, 1))
        xs.append(x)
        ys.append(x @ b + err[:, [i]])
    assert np.isfinite(np.hstack(ys)).all()
    res = SUR(equations(xs, ys), missing="observed").fit(method="ml")
    p = sum(x.shape[1] for x in xs)

    # Independent reference: Newton iteration on the same observed
    # likelihood, started from a deliberately distant point.
    lik = res.model._fit_engine_likelihood
    beta = np.zeros((p, 1))
    sig = np.diag(np.linspace(0.5, 2.0, k))
    for _ in range(100):
        polished = lik.newton_polish(beta, sig, tol=1e-12)
        beta, sig = polished["beta"], polished["sigma"]
        if polished["score_norm"] < 1e-10:
            break
    assert_allclose(np.asarray(res.params).ravel(), beta.ravel(), rtol=RTOL, atol=ATOL)
    assert_allclose(np.asarray(res.sigma), sig, rtol=RTOL, atol=ATOL)
    # MLE sigma is positive definite
    assert np.linalg.eigvalsh(np.asarray(res.sigma)).min() > 0


def test_singular_boundary_detected():
    # A nearly perfectly collinear pair drives the covariance to the edge.
    xs, ys, _, _, _ = generate(n=200, k=2, miss=0.1, seed=40)
    # duplicate the dependent of eq0 into eq1 to force singular covariance
    ys[1] = ys[0].copy()
    # introduce some missingness so the design stays estimable
    rng = np.random.default_rng(1)
    m0 = rng.random(200) < 0.2
    ys[0][m0] = np.nan
    with pytest.raises((ObservedSURError, ObservedSURConvergenceError)):
        SUR(equations(xs, ys), missing="observed").fit(method="ml")


def test_formula_interface_observed():
    pytest.importorskip("formulaic")
    n = 300
    rng = np.random.default_rng(50)
    frame = pd.DataFrame(
        rng.standard_normal((n, 6)),
        columns=["y1", "y2", "x1", "x2", "z1", "z2"],
    )
    frame["y1"] = 1.0 + frame["x1"] - 0.5 * frame["x2"] + rng.standard_normal(n)
    frame["y2"] = 0.5 + frame["z1"] + rng.standard_normal(n)
    frame.loc[frame.sample(40, random_state=1).index, "y1"] = np.nan
    formula = {"eq1": "y1 ~ 1 + x1 + x2", "eq2": "y2 ~ 1 + z1 + z2"}
    mod = SUR.from_formula(formula, frame, missing="observed")
    res = mod.fit(method="ml")
    assert res.converged
    assert res.resids.shape[0] == n
    # formula-based out-of-sample prediction still works
    new = frame.iloc[:10]
    oos = res.predict(data=new, dataframe=True)
    assert oos.shape[0] == 10
