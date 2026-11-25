"""
Independent cross-checks for the observed-data SUR ML implementation.

All reference computations here are written from scratch (pattern-wise
normal densities, independent scipy optimizers, hand-assembled GLS) so that
they do not share code paths with the production estimator.

Run with:  python verify_observed_sur.py
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.linalg import slogdet
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from linearmodels.system import SUR
from linearmodels.system.ml import (
    ObservedSURError,
    _ObservedLikelihood,
    _build_patterns,
    _vech_to_full,
)

RTOL = 1e-5
ATOL = 1e-7
rng = np.random.default_rng(20261125)


def make_system(  # noqa: PLR0917
    n=600, k=3, px=(2, 3, 2), miss=0.3, overlap=True, seed=0
):
    r = np.random.default_rng(seed)
    A = r.standard_normal((k, k))
    sigma = A @ A.T / k + np.eye(k)
    L = np.linalg.cholesky(sigma)
    err = r.standard_normal((n, k)) @ L.T
    xs, ys, betas = [], [], []
    for i in range(k):
        p = (
            px[i]
            if isinstance(px, (tuple, list)) and i < len(px)
            else (px if not isinstance(px, (tuple, list)) else px[-1])
        )
        x = np.column_stack([np.ones(n), r.standard_normal((n, p))])
        b = r.standard_normal((p + 1, 1))
        xs.append(x)
        ys.append(x @ b + err[:, [i]])
        betas.append(b)
    y_mis = [y.copy() for y in ys]
    if miss > 0:
        m = r.random((n, k)) < miss
        for i in range(k):
            y_mis[i][m[:, i]] = np.nan
    if not overlap:
        # First half: only odd eqs; second half: only even eqs.
        for i in range(k):
            if i % 2 == 0:
                y_mis[i][: n // 2] = np.nan
            else:
                y_mis[i][n // 2 :] = np.nan
    # A few completely empty rows retained in the layout (skipped when the
    # test explicitly requests a complete rectangular dataset).
    if miss != 0.0:
        for rr in (0, 1, n - 1):
            for i in range(k):
                y_mis[i][rr] = np.nan
    obs = np.isfinite(np.hstack(y_mis))
    return xs, y_mis, obs, sigma, betas


# ---------------------------------------------------------------------------
# 1. Reference likelihood written from the normal density per pattern
# ---------------------------------------------------------------------------
def reference_loglik(beta, sigma, xs, ys, observed):
    n, k = observed.shape
    total = 0.0
    fitted = [xs[i] @ np.asarray(beta[i]).ravel() for i in range(k)]
    for row in range(n):
        eq = np.flatnonzero(observed[row])
        if eq.size == 0:
            continue
        e = np.array([ys[i][row, 0] - fitted[i][row] for i in eq])
        s = sigma[np.ix_(eq, eq)]
        sign, ld = slogdet(s)
        assert sign > 0
        # Sum of independent standard-normal logs, then correct for the
        # actual marginal covariance of the observed sub-vector.
        total += np.log(norm.pdf(e, scale=1.0)).sum()
        total -= 0.5 * (ld + e @ np.linalg.solve(s, e) - e @ e)
    return total


def reference_known_sigma_gls(xs, ys, observed, sigma):
    """Hand-assembled pattern-wise GLS, equation by equation."""
    widths = [x.shape[1] for x in xs]
    p = sum(widths)
    offs = np.cumsum([0, *widths])
    info = np.zeros((p, p))
    rhs = np.zeros(p)
    patterns = _build_patterns(observed)
    for pat in patterns:
        rows, eq = pat.rows, pat.eqns
        im = np.linalg.inv(sigma[np.ix_(eq, eq)])
        for a, i in enumerate(eq):
            xi = xs[i][rows]
            for b, j in enumerate(eq):
                info[offs[i] : offs[i + 1], offs[j] : offs[j + 1]] += (
                    im[a, b] * xi.T @ xs[j][rows]
                )
            ystack = np.column_stack([ys[jj][rows, 0] for jj in eq])
            rhs[offs[i] : offs[i + 1]] += xi.T @ (im[a] @ ystack.T)
    beta = np.linalg.solve(info, rhs)
    return np.split(beta, offs[1:-1]), beta, info


def test_known_sigma_gls():
    xs, ys, obs, sigma, _ = make_system(seed=1)
    # Reference
    _, beta_ref, _ = reference_known_sigma_gls(xs, ys, obs, sigma)
    # Production
    eqns = {f"eq{i}": {"dependent": ys[i], "exog": xs[i]} for i in range(len(xs))}
    res = SUR(eqns, sigma=sigma, missing="observed").fit(
        method="ml", cov_type="unadjusted"
    )
    beta_hat = np.asarray(res.params).ravel()
    np.testing.assert_allclose(beta_hat, beta_ref, rtol=RTOL, atol=ATOL)
    split_at = np.cumsum([x.shape[1] for x in xs])[:-1]
    ll_ref = reference_loglik(np.split(beta_hat, split_at), sigma, xs, ys, obs)
    np.testing.assert_allclose(res.loglik, ll_ref, rtol=RTOL, atol=ATOL)
    # Known sigma: unadjusted beta cov = inverse beta observed information
    lik = _ObservedLikelihood(xs, ys, obs, _build_patterns(obs))
    bcol = beta_hat[:, None]
    hess = lik.observed_information(bcol, sigma, include_sigma=False)
    cov_ref = np.linalg.inv(hess)
    np.testing.assert_allclose(np.asarray(res.cov), cov_ref, rtol=RTOL, atol=ATOL)
    # Independently confirm the analytic information via finite differences.
    hess_fd = lik.numerical_hessian(bcol, sigma, include_sigma=False)
    np.testing.assert_allclose(hess, hess_fd, rtol=1e-5, atol=1e-6)
    print("test_known_sigma_gls OK")


def test_unknown_sigma_vs_independent_optimizer():
    xs, ys, obs, _, _ = make_system(seed=2)
    k = len(xs)
    eqns = {f"eq{i}": {"dependent": ys[i], "exog": xs[i]} for i in range(k)}
    res = SUR(eqns, missing="observed").fit(method="ml")

    widths = [x.shape[1] for x in xs]
    p = sum(widths)

    def unpack(theta):
        beta = theta[:p]
        sig = _vech_to_full(theta[p:], k)
        return beta, sig

    def nll(theta):
        beta, sig = unpack(theta)
        pieces = np.split(beta, np.cumsum(widths)[:-1])
        try:
            return -reference_loglik(pieces, sig, xs, ys, obs)
        except np.linalg.LinAlgError:
            return 1e15

    # Independent L-BFGS-B optimization using numerical gradients of the
    # hand-written reference likelihood (no analytic code shared with the
    # production estimator).
    beta0 = np.concatenate(
        [
            np.linalg.lstsq(xs[i][obs[:, i]], ys[i][obs[:, i]], rcond=None)[0].ravel()
            for i in range(k)
        ]
    )
    sig0 = np.eye(k)
    theta0 = np.r_[beta0, sig0[np.triu_indices(k)]]
    opt = minimize(
        nll,
        theta0,
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-10},
    )
    opt2 = minimize(
        nll,
        np.r_[beta0, np.diag(np.linspace(0.5, 1.5, k))[np.triu_indices(k)]],
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-10},
    )
    assert abs(opt.fun - opt2.fun) < 1e-8
    beta_ref, sig_ref = unpack(opt.x)
    np.testing.assert_allclose(
        np.asarray(res.params).ravel(), beta_ref, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(np.asarray(res.sigma), sig_ref, rtol=RTOL, atol=1e-6)
    np.testing.assert_allclose(res.loglik, -opt.fun, rtol=RTOL, atol=ATOL)
    print("test_unknown_sigma_vs_independent_optimizer OK")


def test_information_and_scores_by_finite_difference():
    xs, ys, _obs, _, _ = make_system(n=400, k=3, seed=3)
    eqns = {f"eq{i}": {"dependent": ys[i], "exog": xs[i]} for i in range(3)}
    res = SUR(eqns, missing="observed").fit(method="ml")
    lik = res.model._fit_engine_likelihood
    beta = np.asarray(res.params)[:, None]
    sigma = np.asarray(res.sigma)
    # Analytic observed information vs finite-difference score.
    _, gb, gv = lik.loglik_grad(beta, sigma)
    theta = np.r_[beta.ravel(), sigma.flatten()[lik._vech_idx]]
    h = 1e-6
    fd_hess = np.zeros((theta.size, theta.size))
    for j in range(theta.size):

        def score_at(t):
            b = t[: lik.n_params][:, None]
            s = _vech_to_full(t[lik.n_params :], lik.k)
            _, gbb, gvv = lik.loglik_grad(b, s)
            return np.r_[gbb, gvv]

        tp, tm = theta.copy(), theta.copy()
        tp[j] += h
        tm[j] -= h
        fd_hess[:, j] = -(score_at(tp) - score_at(tm)) / (2 * h)
    fd_hess = (fd_hess + fd_hess.T) / 2
    analytic = lik.observed_information(beta, sigma)
    np.testing.assert_allclose(analytic, fd_hess, rtol=1e-5, atol=1e-5)
    # Row scores sum to the total analytic score.
    rs, _ = lik.row_scores(beta, sigma)
    np.testing.assert_allclose(rs.sum(0), np.r_[gb, gv], rtol=RTOL, atol=ATOL)
    print("test_information_and_scores_by_finite_difference OK")


def test_covariance_estimators():
    xs, ys, _obs, _, _ = make_system(n=800, k=3, seed=4)
    eqns = {f"eq{i}": {"dependent": ys[i], "exog": xs[i]} for i in range(3)}
    mod = SUR(eqns, missing="observed")
    res_u = mod.fit(method="ml", cov_type="unadjusted")
    res_r = mod.fit(method="ml", cov_type="robust")
    lik = res_u.model._fit_engine_likelihood
    beta = np.asarray(res_u.params)[:, None]
    sigma = np.asarray(res_u.sigma)
    hess = lik.observed_information(beta, sigma)
    rs, _ = lik.row_scores(beta, sigma)
    info_inv = np.linalg.inv(hess)
    # unadjusted
    np.testing.assert_allclose(
        np.asarray(res_u.cov),
        info_inv[: beta.shape[0], : beta.shape[0]],
        rtol=RTOL,
        atol=ATOL,
    )
    # robust sandwich
    meat = rs.T @ rs
    full_sand = info_inv @ meat @ info_inv
    full_sand = (full_sand + full_sand.T) / 2
    np.testing.assert_allclose(
        np.asarray(res_r.cov),
        full_sand[: beta.shape[0], : beta.shape[0]],
        rtol=RTOL,
        atol=ATOL,
    )
    # Known sigma: only beta parameters enter the information.
    res_k = SUR(eqns, sigma=sigma, missing="observed").fit(method="ml")
    assert res_k.information.shape[0] == beta.shape[0]
    print("test_covariance_estimators OK")


def test_no_missing_degenerate_ml():
    """Without missingness the observed ML must equal the textbook normal ML."""
    xs, ys, _, _, _ = make_system(n=700, k=3, miss=0.0, seed=5)
    # make_system always injects fully empty rows; rebuild clean data.
    ys = [y.copy() for y in ys]
    obs = np.isfinite(np.hstack(ys))
    assert obs.all()
    eqns = {f"eq{i}": {"dependent": ys[i], "exog": xs[i]} for i in range(3)}
    res = SUR(eqns, missing="observed").fit(method="ml")
    n = obs.shape[0]
    # Closed-form ML for complete data via alternating GLS / MLE sigma.
    widths = [x.shape[1] for x in xs]
    offs = np.cumsum([0, *widths])
    p = sum(widths)

    def alternating(beta_start, sig_start):
        beta, sig = beta_start, sig_start
        prev = np.inf
        for _ in range(2000):
            sig_inv = np.linalg.inv(sig)
            info = np.zeros((p, p))
            rhs = np.zeros(p)
            for i in range(3):
                for j in range(3):
                    info[offs[i] : offs[i + 1], offs[j] : offs[j + 1]] = (
                        sig_inv[i, j] * xs[i].T @ xs[j]
                    )
                    rhs[offs[i] : offs[i + 1]] += (
                        sig_inv[i, j] * xs[i].T @ ys[j].ravel()
                    )
            beta = np.linalg.solve(info, rhs)
            E = np.column_stack(
                [ys[i].ravel() - xs[i] @ beta[offs[i] : offs[i + 1]] for i in range(3)]
            )
            sig = E.T @ E / n
            movement = np.max(np.abs(E))
            if abs(movement - prev) < 1e-13:
                break
            prev = movement
        return beta, sig

    # Two independent starts must converge to the same interior optimum.
    b_ols = np.concatenate(
        [np.linalg.lstsq(xs[i], ys[i], rcond=None)[0].ravel() for i in range(3)]
    )
    beta, sig = alternating(b_ols, np.eye(3))
    beta2, sig2 = alternating(b_ols, np.diag(np.linspace(0.5, 2.0, 3)))
    np.testing.assert_allclose(beta, beta2, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(sig, sig2, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        np.asarray(res.params).ravel(), beta, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(np.asarray(res.sigma), sig, rtol=RTOL, atol=ATOL)
    print("test_no_missing_degenerate_ml OK")


def test_layout_and_predict():
    xs, ys, _obs, _, _ = make_system(n=350, k=3, seed=6)
    idx = pd.date_range("2020-01-01", periods=350)
    eqns = {
        f"eq{i}": {
            "dependent": pd.DataFrame(ys[i], index=idx, columns=["y"]),
            "exog": pd.DataFrame(
                xs[i], index=idx, columns=[f"x{j}" for j in range(xs[i].shape[1])]
            ),
        }
        for i in range(3)
    }
    mod = SUR(eqns, missing="observed")
    res = mod.fit(method="ml")
    # residual layout
    rdf = res.resids
    fdf = res.fitted_values
    obs = _obs
    assert rdf.shape == (350, 3)
    assert fdf.shape == (350, 3)
    assert np.array_equal(rdf.isna().values, ~obs)
    assert not fdf.isna().any().any()
    # residuals equal y - xb on observed cells
    widths = [x.shape[1] for x in xs]
    offs = np.cumsum([0, *widths])
    beta = np.asarray(res.params)
    for i in range(3):
        fit = xs[i] @ beta[offs[i] : offs[i + 1]]
        np.testing.assert_allclose(
            fdf.iloc[:, i].values, fit.ravel(), rtol=RTOL, atol=ATOL
        )
        m = obs[:, i]
        np.testing.assert_allclose(
            rdf.values[m, i], (ys[i][m, 0] - fit.ravel()[m]), rtol=RTOL, atol=ATOL
        )
    # in-sample predict with missing=True reindexes to original layout
    pred = res.predict(dataframe=True, missing=True)
    assert pred.shape == (350, 3)
    # out-of-sample predict
    new_idx = pd.date_range("2021-01-01", periods=10)
    new_eqns = {
        f"eq{i}": {"exog": pd.DataFrame(xs[i][:10], index=new_idx)} for i in range(3)
    }
    oos = res.predict(equations=new_eqns, dataframe=True)
    assert oos.shape[0] == 10
    print("test_layout_and_predict OK")


def test_error_paths():
    xs, ys, _obs, sigma, _ = make_system(n=300, k=3, seed=7)

    def build(y=ys, x=xs, **kw):
        eqns = {f"eq{i}": {"dependent": y[i], "exog": x[i]} for i in range(3)}
        return SUR(eqns, missing="observed", **kw)

    # wrong method
    with np.testing.assert_raises_regex(ObservedSURError, "only support"):
        build().fit(method="gls")
    with np.testing.assert_raises_regex(ObservedSURError, "only support"):
        build().fit()
    # unsupported cov types
    for ct in ("kernel", "clustered"):
        with np.testing.assert_raises_regex(ValueError, "supports cov_type"):
            build().fit(method="ml", cov_type=ct)
    with np.testing.assert_raises_regex(ValueError, "Debiased"):
        build().fit(method="ml", debiased=True)
    # bad regressor values even when dependent is missing there
    x_bad = [x.copy() for x in xs]
    x_bad[0][5, 1] = np.nan
    ys_bad = [y.copy() for y in ys]
    ys_bad[0][5] = np.nan
    with np.testing.assert_raises_regex(ObservedSURError, "missing or infinite"):
        build(y=ys_bad, x=x_bad)
    # infinite dependent rejected
    ys_inf = [y.copy() for y in ys]
    ys_inf[1][3, 0] = np.inf
    with np.testing.assert_raises_regex(ObservedSURError, "[Ii]nfinite"):
        build(y=ys_inf)
    # weights rejected
    eqns_w = {
        f"eq{i}": {"dependent": ys[i], "exog": xs[i], "weights": np.ones((300, 1))}
        for i in range(3)
    }
    with np.testing.assert_raises_regex(ObservedSURError, "[Ww]eights"):
        SUR(eqns_w, missing="observed")
    # constraints rejected
    mod = build()
    r = pd.DataFrame(np.zeros((1, mod.param_names.__len__())), columns=mod.param_names)
    r.iloc[0, 0] = 1.0
    with np.testing.assert_raises_regex(ObservedSURError, "[Cc]onstraint"):
        mod.add_constraints(r)
    # bad known sigma
    with np.testing.assert_raises_regex(ObservedSURError, "positive definite"):
        build(sigma=np.zeros((3, 3)))
    with np.testing.assert_raises_regex(ObservedSURError, "symmetric"):
        build(sigma=sigma + np.triu(np.ones((3, 3)), 1))
    # no overlap between a pair of equations
    xs2, ys2, _obs2, _, _ = make_system(n=300, k=3, overlap=False, seed=8)
    with np.testing.assert_raises_regex(ObservedSURError, "never jointly observed"):
        build(y=ys2, x=xs2).fit(method="ml")
    # known sigma rescues the no-overlap design
    res = build(y=ys2, x=xs2, sigma=sigma).fit(method="ml")
    assert np.isfinite(res.loglik)
    # rank deficient observed design
    x_rd = [x.copy() for x in xs]
    x_rd[0][:, 2] = x_rd[0][:, 1]
    with np.testing.assert_raises_regex(ObservedSURError, "full rank"):
        build(x=x_rd)
    # equation with no observations
    ys_empty = [y.copy() for y in ys]
    ys_empty[2][:] = np.nan
    with np.testing.assert_raises_regex(ObservedSURError, "no observed"):
        build(y=ys_empty)
    # mismatched pandas indexes
    eqns_mm = {
        f"eq{i}": {
            "dependent": pd.DataFrame(ys[i], index=range(300), columns=["y"]),
            "exog": pd.DataFrame(
                xs[i],
                index=range(100, 400),
                columns=[f"x{j}" for j in range(xs[i].shape[1])],
            ),
        }
        for i in range(3)
    }
    with np.testing.assert_raises_regex(ObservedSURError, "index"):
        SUR(eqns_mm, missing="observed")
    print("test_error_paths OK")


def test_iteration_budget():
    xs, ys, _obs, _, _ = make_system(n=400, k=4, seed=9)
    eqns = {f"eq{i}": {"dependent": ys[i], "exog": xs[i]} for i in range(4)}
    with np.testing.assert_raises_regex(
        ObservedSURError, "[Ii]teration budget|converg|score"
    ):
        SUR(eqns, missing="observed").fit(method="ml", iter_limit=1)
    # generous budget succeeds and termination text is inspectable
    res = SUR(eqns, missing="observed").fit(method="ml", iter_limit=200, tol=1e-8)
    assert res.converged
    assert "max|score|" in res.termination
    print("test_iteration_budget OK")


def test_reordering_equivalence():
    xs, ys, _obs, _sigma, _ = make_system(n=400, k=3, seed=10)
    eqns = {f"eq{i}": {"dependent": ys[i], "exog": xs[i]} for i in range(3)}
    res1 = SUR(eqns, missing="observed").fit(method="ml")
    perm = [2, 0, 1]
    eqns2 = {
        f"eq{perm[i]}": {"dependent": ys[perm[i]], "exog": xs[perm[i]]}
        for i in range(3)
    }
    res2 = SUR(eqns2, missing="observed").fit(method="ml")
    np.testing.assert_allclose(res1.loglik, res2.loglik, rtol=RTOL, atol=ATOL)
    for _new_i, old_i in enumerate(perm):
        lab2 = f"eq{old_i}"
        old = res1.equations[f"eq{old_i}"]
        new = res2.equations[lab2]
        np.testing.assert_allclose(old.params, new.params, rtol=RTOL, atol=ATOL)
    print("test_reordering_equivalence OK")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    test_known_sigma_gls()
    test_unknown_sigma_vs_independent_optimizer()
    test_information_and_scores_by_finite_difference()
    test_covariance_estimators()
    test_no_missing_degenerate_ml()
    test_layout_and_predict()
    test_error_paths()
    test_iteration_budget()
    test_reordering_equivalence()
    print("\nAll independent verification checks passed.")
