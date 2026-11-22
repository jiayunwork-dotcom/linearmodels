"""
Tests for IV2SLS with absorbed high-dimensional categorical effects

The reference implementation is an IV2SLS model where the effects are
included as explicit dummy variables.  All comparisons between the
absorbing implementation and the explicit dummy-variable model use
rtol=1e-7 and atol=1e-9 unless noted otherwise.
"""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from linearmodels.iv import IV2SLS
from linearmodels.iv._absorbing import (
    EffectProjector,
    effect_space_rank,
    validate_absorb_frame,
)
from linearmodels.iv.results import AbsorbedFirstStageResults, IVAbsorbingResults
from linearmodels.shared.exceptions import MissingValueWarning
from linearmodels.shared.hypotheses import InvalidTestStatistic
from linearmodels.shared.utility import AttrDict

RTOL = 1e-7
ATOL = 1e-9

COV_TYPES = ["unadjusted", "robust", "clustered"]


def connected_components(codes1, codes2):
    """Connected components of the bipartite category graph"""
    g1 = int(codes1.max()) + 1
    g2 = int(codes2.max()) + 1
    parent = list(range(g1 + g2))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(codes1.shape[0]):
        r1, r2 = find(int(codes1[i])), find(g1 + int(codes2[i]))
        if r1 != r2:
            parent[r2] = r1
    return len({find(v) for v in range(g1 + g2)})


def full_rank_dummies(cats):
    """
    Explicit full-rank dummy design for one or two categorical columns

    Uses all dummies of the first effect and, for the second effect, drops
    one category per connected component of the bipartite graph.
    """
    parts = [pd.get_dummies(cats.iloc[:, 0], prefix="eff0", dtype=float)]
    if cats.shape[1] == 2:
        codes1 = cats.iloc[:, 0].cat.codes.to_numpy()
        codes2 = cats.iloc[:, 1].cat.codes.to_numpy()
        g1 = int(codes1.max()) + 1
        g2 = int(codes2.max()) + 1
        parent = list(range(g1 + g2))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i in range(codes1.shape[0]):
            r1, r2 = find(int(codes1[i])), find(g1 + int(codes2[i]))
            if r1 != r2:
                parent[r2] = r1
        drop = set()
        for j in range(g2):
            root = find(g1 + j)
            if root not in drop:
                drop.add(root)
                # drop the first category of each component
        drop_cats = set()
        seen = set()
        for j in range(g2):
            root = find(g1 + j)
            if root not in seen:
                seen.add(root)
                drop_cats.add(j)
        d2 = pd.get_dummies(cats.iloc[:, 1], prefix="eff1", dtype=float)
        keep = [j for j in range(g2) if j not in drop_cats]
        parts.append(d2.iloc[:, keep])
    return pd.concat(parts, axis=1)


def make_iv_data(seed=12345, nobs=2000, g1=100, g2=17, *, weighted=True, k_exog=2):
    """Well-conditioned synthetic IV data with two unbalanced effects"""
    rs = np.random.RandomState(seed)
    # Unbalanced category sizes
    p1 = rs.dirichlet(np.ones(g1))
    p2 = rs.dirichlet(np.ones(g2))
    c1 = rs.choice(g1, size=nobs, p=p1)
    c2 = rs.choice(g2, size=nobs, p=p2)
    absorb = pd.DataFrame({"firm": pd.Categorical(c1), "year": pd.Categorical(c2)})
    fe = rs.standard_normal(g1)[c1] + 0.7 * rs.standard_normal(g2)[c2]
    z = rs.standard_normal((nobs, 3))
    x = rs.standard_normal((nobs, k_exog))
    endog = 0.4 * z.sum(1) + x.sum(1) + rs.standard_normal(nobs)
    beta_x = np.arange(1, k_exog + 1) * 0.5
    y = 0.0 + x @ beta_x + 0.8 * endog + fe + rs.standard_normal(nobs)
    y = y + 2.0  # level shift absorbed by the effects
    w = rs.uniform(0.5, 2.0, nobs) if weighted else np.ones(nobs)
    clusters = rs.randint(0, 40, nobs)
    cols_x = [f"x{i}" for i in range(k_exog)]
    out = AttrDict(
        y=pd.Series(y, name="y"),
        x=pd.DataFrame(x, columns=cols_x),
        endog=pd.DataFrame(endog, columns=["endog"]),
        z=pd.DataFrame(z, columns=[f"z{i}" for i in range(3)]),
        absorb=absorb,
        w=pd.Series(w, name="w"),
        clusters=clusters,
        fe=fe,
    )
    return out


def explicit_model(data, mask=None):
    """Explicit dummy-variable IV2SLS on the same effective sample"""
    if mask is None:
        mask = np.ones(data.y.shape[0], dtype=bool)
    absorb = data.absorb.loc[mask].copy()
    for col in absorb:
        absorb[col] = absorb[col].cat.remove_unused_categories()
    dummies = full_rank_dummies(absorb)
    exog = pd.concat(
        [data.x.loc[mask].reset_index(drop=True), dummies.reset_index(drop=True)],
        axis=1,
    )
    mod = IV2SLS(
        data.y.loc[mask].to_numpy(),
        exog.to_numpy(),
        data.endog.loc[mask].to_numpy(),
        data.z.loc[mask].to_numpy(),
        weights=data.w.loc[mask].to_numpy(),
    )
    k_struct = data.x.shape[1] + data.endog.shape[1]
    return mod, k_struct


def structural(arr, k_exog, k_endog):
    """Select the structural coefficients from the explicit parameter vector"""
    return np.r_[arr[:k_exog], arr[arr.shape[0] - k_endog :]]


def structural_cov(cov, k_exog, k_endog):
    idx = list(range(k_exog)) + list(range(cov.shape[0] - k_endog, cov.shape[0]))
    return cov[np.ix_(idx, idx)]


@pytest.fixture
def data():
    return make_iv_data()


def fit_both(data, cov_type, debiased):
    """Fit the absorbing model and the explicit reference"""
    kw = {}
    if cov_type == "clustered":
        kw["clusters"] = data.clusters
    mod = IV2SLS(data.y, data.x, data.endog, data.z, weights=data.w, absorb=data.absorb)
    res = mod.fit(cov_type=cov_type, debiased=debiased, **kw)
    mod_ref, k_struct = explicit_model(data)
    res_ref = mod_ref.fit(cov_type=cov_type, debiased=debiased, **kw)
    return res, res_ref, k_struct


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("debiased", [False, True])
def test_against_explicit(data, cov_type, debiased):
    res, res_ref, k_struct = fit_both(data, cov_type, debiased)
    k_exog = data.x.shape[1]
    k_endog = data.endog.shape[1]
    assert isinstance(res, IVAbsorbingResults)
    # Structural coefficients
    assert_allclose(
        res.params.to_numpy(),
        structural(res_ref.params.to_numpy(), k_exog, k_endog),
        rtol=RTOL,
        atol=ATOL,
    )
    # Residuals and fitted values (original units, effects included)
    assert_allclose(
        res.resids.to_numpy(), res_ref.resids.to_numpy(), rtol=RTOL, atol=ATOL
    )
    assert_allclose(
        res.fitted_values.to_numpy(),
        res_ref.fitted_values.to_numpy(),
        rtol=RTOL,
        atol=ATOL,
    )
    # Covariance of the structural block
    assert_allclose(
        res.cov.to_numpy(),
        structural_cov(res_ref.cov.to_numpy(), k_exog, k_endog),
        rtol=RTOL,
        atol=ATOL,
    )
    # Degrees of freedom
    assert res.df_model == res_ref.df_model
    assert res.df_resid == res_ref.df_resid
    assert res.nobs == res_ref.nobs
    assert res.df_absorbed == res_ref.df_model - k_struct
    # Summary statistics
    assert_allclose(res.s2, res_ref.s2, rtol=RTOL, atol=ATOL)
    assert_allclose(res.rsquared, res_ref.rsquared, rtol=RTOL, atol=ATOL)
    assert_allclose(res.total_ss, res_ref.total_ss, rtol=RTOL, atol=ATOL)
    assert_allclose(res.resid_ss, res_ref.resid_ss, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("cov_type", COV_TYPES)
@pytest.mark.parametrize("debiased", [False, True])
def test_first_stage_against_explicit(data, cov_type, debiased):
    res, res_ref, _ = fit_both(data, cov_type, debiased)
    assert isinstance(res.first_stage, AbsorbedFirstStageResults)
    diag = res.first_stage.diagnostics
    diag_ref = res_ref.first_stage.diagnostics
    for col in ["rsquared", "partial.rsquared", "shea.rsquared", "f.stat", "f.pval"]:
        assert_allclose(
            diag[col].to_numpy(dtype=float),
            diag_ref[col].to_numpy(dtype=float),
            rtol=RTOL,
            atol=ATOL,
            err_msg=col,
        )
    assert list(diag["f.dist"]) == list(diag_ref["f.dist"])
    # Individual first-stage regressions
    k_exog = data.x.shape[1]
    for var in data.endog.columns:
        ind = res.first_stage.individual[var]
        ind_ref = res_ref.first_stage.individual[var]
        params_ref = np.r_[
            ind_ref.params.to_numpy()[:k_exog],
            ind_ref.params.to_numpy()[-data.z.shape[1] :],
        ]
        assert_allclose(ind.params.to_numpy(), params_ref, rtol=RTOL, atol=ATOL)
        assert ind.df_resid == ind_ref.df_resid
        assert_allclose(ind.rsquared, ind_ref.rsquared, rtol=RTOL, atol=ATOL)


def test_single_effect(data):
    absorb = data.absorb[["firm"]]
    mod = IV2SLS(data.y, data.x, data.endog, data.z, weights=data.w, absorb=absorb)
    res = mod.fit(cov_type="robust", debiased=True)
    data1 = data.copy()
    data1.absorb = absorb
    mod_ref, _ = explicit_model(data1)
    res_ref = mod_ref.fit(cov_type="robust", debiased=True)
    k_exog, k_endog = data.x.shape[1], data.endog.shape[1]
    assert_allclose(
        res.params.to_numpy(),
        structural(res_ref.params.to_numpy(), k_exog, k_endog),
        rtol=RTOL,
        atol=ATOL,
    )
    assert_allclose(
        res.cov.to_numpy(),
        structural_cov(res_ref.cov.to_numpy(), k_exog, k_endog),
        rtol=RTOL,
        atol=ATOL,
    )
    assert res.df_absorbed == absorb["firm"].nunique()


def test_no_exog(data):
    mod = IV2SLS(data.y, None, data.endog, data.z, weights=data.w, absorb=data.absorb)
    res = mod.fit(cov_type="robust")
    assert res.params.shape[0] == 1
    dummies = full_rank_dummies(data.absorb)
    mod_ref = IV2SLS(
        data.y.to_numpy(),
        dummies.to_numpy(),
        data.endog.to_numpy(),
        data.z.to_numpy(),
        weights=data.w.to_numpy(),
    )
    res_ref = mod_ref.fit(cov_type="robust")
    assert_allclose(
        res.params.to_numpy(),
        res_ref.params.to_numpy()[-1:],
        rtol=RTOL,
        atol=ATOL,
    )


def test_unweighted(data):
    mod = IV2SLS(data.y, data.x, data.endog, data.z, absorb=data.absorb)
    res = mod.fit(cov_type="robust", debiased=True)
    mod_ref, _ = explicit_model(
        AttrDict(dict(data.items(), w=pd.Series(np.ones(data.y.shape[0]))))
    )
    res_ref = mod_ref.fit(cov_type="robust", debiased=True)
    k_exog, k_endog = data.x.shape[1], data.endog.shape[1]
    assert_allclose(
        res.params.to_numpy(),
        structural(res_ref.params.to_numpy(), k_exog, k_endog),
        rtol=RTOL,
        atol=ATOL,
    )


def test_fitted_contains_effects(data):
    res, _, _ = fit_both(data, "robust", False)
    fitted = res.fitted_values.to_numpy().squeeze()
    # The structural-only prediction differs from fitted
    structural_fit = (
        data.x.to_numpy() @ res.params.to_numpy()[: data.x.shape[1]]
        + data.endog.to_numpy() @ res.params.to_numpy()[data.x.shape[1] :]
    )
    assert np.abs(structural_fit - fitted).max() > 0.1
    # The difference between fitted and the structural terms recovers the
    # contribution of the absorbed effects
    assert np.corrcoef(fitted - structural_fit, data.fe)[0, 1] > 0.9


def test_missing_data(data):
    rs = np.random.RandomState(777)
    n = data.y.shape[0]
    y = data.y.copy()
    x = data.x.copy()
    endog = data.endog.copy()
    z = data.z.copy()
    absorb = data.absorb.copy()
    w = data.w.copy()
    y.iloc[rs.choice(n, 17, replace=False)] = np.nan
    x.iloc[rs.choice(n, 13, replace=False), 0] = np.nan
    endog.iloc[rs.choice(n, 11, replace=False)] = np.nan
    z.iloc[rs.choice(n, 7, replace=False), 1] = np.nan
    w.iloc[rs.choice(n, 5, replace=False)] = np.nan
    absorb.iloc[rs.choice(n, 9, replace=False), 0] = np.nan
    with pytest.warns(MissingValueWarning):
        mod = IV2SLS(y, x, endog, z, weights=w, absorb=absorb)
    res = mod.fit(cov_type="robust", debiased=True)
    mask = mod.notnull
    assert mask.shape[0] == n
    assert mod.isnull.sum() > 0
    assert res.nobs == mask.sum()
    # Reference on the same effective sample
    data_m = AttrDict(dict(data.items()))
    data_m.y, data_m.x, data_m.endog, data_m.z, data_m.w, data_m.absorb = (
        y,
        x,
        endog,
        z,
        w,
        absorb,
    )
    mod_ref, _ = explicit_model(data_m, mask)
    res_ref = mod_ref.fit(cov_type="robust", debiased=True)
    k_exog, k_endog = x.shape[1], 1
    assert_allclose(
        res.params.to_numpy(),
        structural(res_ref.params.to_numpy(), k_exog, k_endog),
        rtol=RTOL,
        atol=ATOL,
    )
    assert_allclose(
        res.resids.to_numpy(), res_ref.resids.to_numpy(), rtol=RTOL, atol=ATOL
    )
    # Index and labels preserved
    assert list(res.resids.index) == list(y.index[mask])
    # In-sample predictions with missing locations filled back
    pred = res.predict(missing=True)
    assert pred.shape[0] == n
    assert pred["fitted_values"].isnull().sum() == int((~mask).sum())
    assert_allclose(
        pred["fitted_values"].dropna().to_numpy(),
        res.fitted_values.to_numpy().squeeze(),
        rtol=RTOL,
        atol=ATOL,
    )


def test_category_rename_and_shuffle(data):
    mod = IV2SLS(data.y, data.x, data.endog, data.z, weights=data.w, absorb=data.absorb)
    res = mod.fit(cov_type="robust")
    # Rename categories: labels must not matter
    absorb_renamed = data.absorb.copy()
    for col in absorb_renamed:
        s = absorb_renamed[col]
        s = s.cat.rename_categories([f"lvl-{c}" for c in s.cat.categories])
        absorb_renamed[col] = s
    mod_rn = IV2SLS(
        data.y, data.x, data.endog, data.z, weights=data.w, absorb=absorb_renamed
    )
    res_rn = mod_rn.fit(cov_type="robust")
    assert_allclose(
        res.params.to_numpy(), res_rn.params.to_numpy(), rtol=RTOL, atol=ATOL
    )
    assert_allclose(
        res.resids.to_numpy(), res_rn.resids.to_numpy(), rtol=RTOL, atol=ATOL
    )
    # Shuffle rows consistently
    rs = np.random.RandomState(31)
    perm = rs.permutation(data.y.shape[0])
    idx = data.y.index[perm]
    mod_sh = IV2SLS(
        data.y.iloc[perm],
        data.x.iloc[perm],
        data.endog.iloc[perm],
        data.z.iloc[perm],
        weights=data.w.iloc[perm],
        absorb=data.absorb.iloc[perm],
    )
    res_sh = mod_sh.fit(cov_type="robust")
    assert_allclose(
        res.params.to_numpy(), res_sh.params.to_numpy(), rtol=RTOL, atol=ATOL
    )
    assert_allclose(
        res.resids.to_numpy()[perm],
        res_sh.resids.to_numpy(),
        rtol=RTOL,
        atol=ATOL,
    )
    assert list(res_sh.resids.index) == list(idx)


def test_nested_effects():
    rs = np.random.RandomState(101)
    n = 1500
    c1 = rs.randint(0, 60, n)
    c2 = c1 * 3 + rs.randint(0, 3, n)  # nested in c1
    absorb = pd.DataFrame({"a": pd.Categorical(c1), "b": pd.Categorical(c2)})
    codes = [absorb[c].cat.codes.to_numpy() for c in absorb]
    assert effect_space_rank(codes) == absorb["b"].nunique()
    fe = rs.standard_normal(60)[c1] + 0.4 * rs.standard_normal(180)[c2]
    z1 = rs.standard_normal(n)
    x1 = rs.standard_normal(n)
    endog = 0.5 * z1 + x1 + rs.standard_normal(n)
    y = 0.6 * x1 + 0.9 * endog + fe + rs.standard_normal(n)
    mod = IV2SLS(y, x1[:, None], endog[:, None], z1[:, None], absorb=absorb)
    res = mod.fit(cov_type="robust", debiased=True)
    d = AttrDict(
        y=pd.Series(y),
        x=pd.DataFrame(x1[:, None], columns=["x0"]),
        endog=pd.DataFrame(endog[:, None], columns=["endog"]),
        z=pd.DataFrame(z1[:, None], columns=["z0"]),
        absorb=absorb,
        w=pd.Series(np.ones(n)),
    )
    mod_ref, _ = explicit_model(d)
    res_ref = mod_ref.fit(cov_type="robust", debiased=True)
    assert_allclose(
        res.params.to_numpy(),
        structural(res_ref.params.to_numpy(), 1, 1),
        rtol=RTOL,
        atol=ATOL,
    )
    assert res.df_model == res_ref.df_model


def test_disconnected_effects():
    rs = np.random.RandomState(202)
    n = 1500
    block = np.arange(n) // 500
    c1 = block * 10 + rs.randint(0, 10, n)
    c2 = block * 7 + rs.randint(0, 7, n)
    absorb = pd.DataFrame({"a": pd.Categorical(c1), "b": pd.Categorical(c2)})
    codes = [absorb[c].cat.codes.to_numpy() for c in absorb]
    ncomp = connected_components(codes[0], codes[1])
    assert ncomp == 3
    assert effect_space_rank(codes) == 30 + 21 - 3
    fe = rs.standard_normal(30)[c1] + 0.4 * rs.standard_normal(21)[c2]
    z1 = rs.standard_normal(n)
    x1 = rs.standard_normal(n)
    endog = 0.5 * z1 + x1 + rs.standard_normal(n)
    y = 0.6 * x1 + 0.9 * endog + fe + rs.standard_normal(n)
    mod = IV2SLS(y, x1[:, None], endog[:, None], z1[:, None], absorb=absorb)
    res = mod.fit(cov_type="robust", debiased=True)
    assert res.df_absorbed == 30 + 21 - 3
    d = AttrDict(
        y=pd.Series(y),
        x=pd.DataFrame(x1[:, None], columns=["x0"]),
        endog=pd.DataFrame(endog[:, None], columns=["endog"]),
        z=pd.DataFrame(z1[:, None], columns=["z0"]),
        absorb=absorb,
        w=pd.Series(np.ones(n)),
    )
    mod_ref, _ = explicit_model(d)
    res_ref = mod_ref.fit(cov_type="robust", debiased=True)
    assert_allclose(
        res.params.to_numpy(),
        structural(res_ref.params.to_numpy(), 1, 1),
        rtol=RTOL,
        atol=ATOL,
    )
    assert_allclose(
        res.cov.to_numpy(),
        structural_cov(res_ref.cov.to_numpy(), 1, 1),
        rtol=RTOL,
        atol=ATOL,
    )


def test_singletons_retained(data):
    n = data.y.shape[0]
    # Rebuild the first effect so that many categories are singletons
    firm = data.absorb["firm"].cat.codes.to_numpy().astype(object)
    nsingletons = 200
    firm[:nsingletons] = [f"singleton-{i}" for i in range(nsingletons)]
    absorb = data.absorb.copy()
    absorb["firm"] = pd.Categorical(firm)
    counts = absorb["firm"].value_counts()
    assert (counts == 1).sum() >= nsingletons
    assert (counts.loc[[f"singleton-{i}" for i in range(nsingletons)]] == 1).all()
    mod = IV2SLS(data.y, data.x, data.endog, data.z, weights=data.w, absorb=absorb)
    res = mod.fit(cov_type="robust")
    # Singletons are kept: no observation is dropped
    assert res.nobs == n
    assert mod.notnull.all()
    # And the fit coincides with the explicit dummy model
    d = AttrDict(dict(data.items()))
    d.absorb = absorb
    mod_ref, _ = explicit_model(d)
    res_ref = mod_ref.fit(cov_type="robust")
    k_exog, k_endog = data.x.shape[1], data.endog.shape[1]
    assert_allclose(
        res.params.to_numpy(),
        structural(res_ref.params.to_numpy(), k_exog, k_endog),
        rtol=RTOL,
        atol=ATOL,
    )


def test_repeated_and_interleaved_fit(data):
    mod_a = IV2SLS(
        data.y, data.x, data.endog, data.z, weights=data.w, absorb=data.absorb
    )
    absorb_b = data.absorb[["year"]]
    w_b = 1.0 / data.w
    mod_b = IV2SLS(data.y, data.x, data.endog, data.z, weights=w_b, absorb=absorb_b)
    res_a1 = mod_a.fit(cov_type="robust", debiased=True)
    res_b1 = mod_b.fit(cov_type="robust", debiased=True)
    res_a2 = mod_a.fit(cov_type="robust", debiased=True)
    res_b2 = mod_b.fit(cov_type="robust", debiased=True)
    # Repeated fits are identical and interleaving does not cross-contaminate
    assert_allclose(res_a1.params.to_numpy(), res_a2.params.to_numpy())
    assert_allclose(res_b1.params.to_numpy(), res_b2.params.to_numpy())
    assert_allclose(res_a1.cov.to_numpy(), res_a2.cov.to_numpy())
    assert_allclose(res_b1.cov.to_numpy(), res_b2.cov.to_numpy())
    # Different weights/effects give different answers (sanity)
    assert not np.allclose(
        res_a1.params.to_numpy(), res_b1.params.to_numpy(), rtol=1e-4
    )


def test_constructor_formula_equivalence(data):
    df = pd.concat([data.y, data.x, data.endog, data.z], axis=1)
    formula = "y ~ x0 + x1 + [endog ~ z0 + z1 + z2]"
    mod_f = IV2SLS.from_formula(formula, df, weights=data.w, absorb=data.absorb)
    res_f = mod_f.fit(cov_type="robust", debiased=True)
    mod_c = IV2SLS(
        data.y, data.x, data.endog, data.z, weights=data.w, absorb=data.absorb
    )
    res_c = mod_c.fit(cov_type="robust", debiased=True)
    assert_allclose(
        res_f.params.to_numpy(), res_c.params.to_numpy(), rtol=RTOL, atol=ATOL
    )
    assert_allclose(res_f.cov.to_numpy(), res_c.cov.to_numpy(), rtol=RTOL, atol=ATOL)
    assert list(res_f.params.index) == list(res_c.params.index)
    assert mod_f.formula == formula


def test_cluster_alignment(data):
    rs = np.random.RandomState(55)
    n = data.y.shape[0]
    y = data.y.copy()
    y.iloc[rs.choice(n, 25, replace=False)] = np.nan
    clusters = pd.Series(rs.randint(0, 30, n), dtype=float)
    # NaN in clusters on dropped rows is fine
    drop_locs = y.isnull().to_numpy()
    clusters.iloc[drop_locs] = np.nan
    with pytest.warns(MissingValueWarning):
        mod = IV2SLS(y, data.x, data.endog, data.z, weights=data.w, absorb=data.absorb)
    res = mod.fit(cov_type="clustered", clusters=clusters)
    assert res.nobs == int((~drop_locs).sum())
    # NaN in clusters on retained rows raises
    clusters_bad = clusters.copy()
    clusters_bad.iloc[~drop_locs] = clusters_bad.iloc[~drop_locs].fillna(0)
    clusters_bad.iloc[0] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        mod.fit(cov_type="clustered", clusters=clusters_bad)
    # Wrong length raises
    with pytest.raises(ValueError, match="wrong number of rows"):
        mod.fit(cov_type="clustered", clusters=clusters.iloc[:-3])
    # Clusters aligned with the estimation sample are accepted
    res2 = mod.fit(cov_type="clustered", clusters=clusters.loc[mod.notnull])
    assert_allclose(res.params.to_numpy(), res2.params.to_numpy(), rtol=RTOL, atol=ATOL)
    assert_allclose(res.cov.to_numpy(), res2.cov.to_numpy(), rtol=RTOL, atol=ATOL)


def test_out_of_sample_prediction_rejected(data):
    mod = IV2SLS(data.y, data.x, data.endog, data.z, absorb=data.absorb)
    res = mod.fit()
    with pytest.raises(ValueError, match="Out-of-sample prediction"):
        mod.predict(res.params, exog=data.x, endog=data.endog)
    with pytest.raises(ValueError, match="Out-of-sample prediction"):
        res.predict(exog=data.x, endog=data.endog)
    df = pd.concat([data.y, data.x, data.endog, data.z], axis=1)
    mod_f = IV2SLS.from_formula(
        "y ~ x0 + x1 + [endog ~ z0 + z1 + z2]", df, absorb=data.absorb
    )
    res_f = mod_f.fit()
    with pytest.raises(ValueError, match="Out-of-sample prediction"):
        res_f.predict(data=df)


def test_in_sample_prediction(data):
    mod = IV2SLS(data.y, data.x, data.endog, data.z, absorb=data.absorb)
    res = mod.fit()
    pred = res.predict()
    assert list(pred.columns) == ["fitted_values"]
    assert_allclose(
        pred["fitted_values"].to_numpy(),
        res.fitted_values.to_numpy().squeeze(),
        rtol=RTOL,
        atol=ATOL,
    )
    pred2 = res.predict(fitted=True, idiosyncratic=True)
    assert list(pred2.columns) == ["fitted_values", "residual"]


def test_absorb_validation(data):
    # Not a DataFrame
    with pytest.raises(TypeError, match="pandas DataFrame"):
        IV2SLS(data.y, data.x, data.endog, data.z, absorb=data.absorb.to_numpy())
    # Too many columns
    absorb3 = data.absorb.copy()
    absorb3["extra"] = absorb3["firm"]
    with pytest.raises(ValueError, match="one or two categorical"):
        IV2SLS(data.y, data.x, data.endog, data.z, absorb=absorb3)
    # Zero columns
    with pytest.raises(ValueError, match="one or two categorical"):
        IV2SLS(
            data.y,
            data.x,
            data.endog,
            data.z,
            absorb=data.absorb.iloc[:, :0],
        )
    # Non-categorical column
    absorb_nc = data.absorb.copy()
    absorb_nc["firm"] = absorb_nc["firm"].cat.codes
    with pytest.raises(TypeError, match="categorical dtype"):
        IV2SLS(data.y, data.x, data.endog, data.z, absorb=absorb_nc)
    # Wrong number of rows
    with pytest.raises(ValueError, match="same number of rows"):
        IV2SLS(data.y, data.x, data.endog, data.z, absorb=data.absorb.iloc[:-1])


def test_weight_validation(data):
    w_neg = data.w.copy()
    w_neg.iloc[3] = -1.0
    with pytest.raises(ValueError, match="strictly positive"):
        IV2SLS(data.y, data.x, data.endog, data.z, weights=w_neg, absorb=data.absorb)
    w_zero = data.w.copy()
    w_zero.iloc[3] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        IV2SLS(data.y, data.x, data.endog, data.z, weights=w_zero, absorb=data.absorb)
    w_inf = data.w.copy()
    w_inf.iloc[3] = np.inf
    with pytest.raises(ValueError, match="finite"):
        IV2SLS(data.y, data.x, data.endog, data.z, weights=w_inf, absorb=data.absorb)
    # NaN weights are treated as missing and dropped
    w_nan = data.w.copy()
    w_nan.iloc[3] = np.nan
    with pytest.warns(MissingValueWarning):
        mod = IV2SLS(
            data.y, data.x, data.endog, data.z, weights=w_nan, absorb=data.absorb
        )
    assert mod.dependent.shape[0] == data.y.shape[0] - 1


def test_structural_requirements(data):
    # endog required
    with pytest.raises(ValueError, match="endog must contain at least one"):
        IV2SLS(data.y, data.x, None, data.z, absorb=data.absorb)
    # instruments required
    with pytest.raises(ValueError, match="instruments must contain at least one"):
        IV2SLS(data.y, data.x, data.endog, None, absorb=data.absorb)
    # order condition
    with pytest.raises(ValueError, match="at least"):
        IV2SLS(
            data.y,
            data.x,
            data.endog,
            data.z.iloc[:, :0],
            absorb=data.absorb,
        )


def test_exog_constant_rejected(data):
    x_const = data.x.copy()
    x_const["const"] = 1.0
    with pytest.raises(ValueError, match="must not contain a constant"):
        IV2SLS(data.y, x_const, data.endog, data.z, absorb=data.absorb)
    # Implicit constant: two columns summing to one
    x_impl = data.x.copy()
    x_impl["x1c"] = 1.0 - x_impl["x0"]
    with pytest.raises(ValueError, match="must not contain a constant"):
        IV2SLS(data.y, x_impl, data.endog, data.z, absorb=data.absorb)


def test_fully_absorbed_rejected(data):
    # exog fully absorbed: exog column is a category-level mean
    x_abs = data.x.copy()
    means = data.endog["endog"].groupby(data.absorb["firm"]).transform("mean")
    x_abs["abs"] = means.to_numpy()
    with pytest.raises(ValueError, match="fully absorbed"):
        IV2SLS(data.y, x_abs[["abs"]], data.endog, data.z, absorb=data.absorb)
    # instrument fully absorbed
    z_abs = data.z.copy()
    z_abs["z_abs"] = means.to_numpy()
    with pytest.raises(ValueError, match="fully absorbed"):
        IV2SLS(data.y, data.x, data.endog, z_abs[["z_abs"]], absorb=data.absorb)
    # endog fully absorbed
    with pytest.raises(ValueError, match="fully absorbed"):
        IV2SLS(data.y, data.x, means.to_frame("endog"), data.z, absorb=data.absorb)


def test_identification_failure(data):
    # Collinear instruments are detected on the original arrays
    z_coll = data.z.copy()
    z_coll["z2"] = z_coll["z0"] - z_coll["z1"]
    with pytest.raises(ValueError, match="full column rank"):
        IV2SLS(data.y, data.x, data.endog, z_coll, absorb=data.absorb)
    # An endogenous regressor explained only by the effects is fully absorbed
    endog_fe = pd.DataFrame({"endog": data.fe + 0.0})
    with pytest.raises(ValueError, match="fully absorbed"):
        IV2SLS(data.y, data.x, endog_fe, data.z, absorb=data.absorb)


def test_not_identified_after_absorption(data):
    # Directly exercise the identification check with an endogenous
    # regressor that is orthogonal to the instruments after absorption
    mod = IV2SLS(data.y, data.x, data.endog, data.z, absorb=data.absorb)
    nexog = data.x.shape[1]
    z_abs = mod._z
    rs = np.random.RandomState(9)
    v = rs.standard_normal((z_abs.shape[0], 1))
    # Residual of v on the absorbed instruments: orthogonal to span(z)
    orth = v - z_abs @ np.linalg.lstsq(z_abs, v, rcond=None)[0]
    assert np.abs(z_abs.T @ orth).max() < 1e-8
    mod._x[:, nexog:] = orth
    with pytest.raises(ValueError, match="not identified"):
        mod._absorb_validate_inputs()


def test_kernel_covariance_rejected(data):
    mod = IV2SLS(data.y, data.x, data.endog, data.z, absorb=data.absorb)
    with pytest.raises(ValueError, match="not supported"):
        mod.fit(cov_type="kernel")


def test_unavailable_diagnostics(data):
    mod = IV2SLS(data.y, data.x, data.endog, data.z, absorb=data.absorb)
    res = mod.fit()
    for attr in [
        "sargan",
        "basmann",
        "wooldridge_score",
        "wooldridge_regression",
        "wooldridge_overid",
        "anderson_rubin",
        "basmann_f",
    ]:
        assert isinstance(getattr(res, attr), InvalidTestStatistic), attr
    assert isinstance(res.durbin(), InvalidTestStatistic)
    assert isinstance(res.wu_hausman(), InvalidTestStatistic)


def test_summary_smoke(data):
    mod = IV2SLS(data.y, data.x, data.endog, data.z, absorb=data.absorb)
    res = mod.fit(cov_type="clustered", clusters=data.clusters, debiased=True)
    s = str(res.summary)
    assert "IV-2SLS Estimation Summary" in s
    assert "Absorbed Effects" in s
    fs = str(res.first_stage.summary)
    assert "First Stage Estimation Results" in fs
    assert "Partial R-squared" in fs


def test_effect_projector_nonconvergence():
    rs = np.random.RandomState(0)
    n = 500
    codes = [rs.randint(0, 50, n), rs.randint(0, 10, n)]
    proj = EffectProjector(codes, np.ones(n), maxiter=1)
    with pytest.raises(RuntimeError, match="did not converge"):
        proj.residualize(rs.standard_normal((n, 1)))


def test_effect_projector_single_vs_lsmr():
    # The exact one-effect demeaning must agree with an LSMR projection
    rs = np.random.RandomState(0)
    n = 800
    codes1 = rs.randint(0, 60, n)
    w = rs.uniform(0.5, 2.0, n)
    mat = rs.standard_normal((n, 3))
    p1 = EffectProjector([codes1], w)
    p2 = EffectProjector([codes1, np.zeros(n, dtype=int)], w)
    assert_allclose(p1.residualize(mat), p2.residualize(mat), rtol=1e-7, atol=1e-9)


def test_validate_absorb_frame(data):
    out = validate_absorb_frame(data.absorb, data.y.shape[0])
    assert out is not data.absorb  # defensive copy
    with pytest.raises(TypeError):
        validate_absorb_frame("not a frame", data.y.shape[0])


def test_absorb_property_and_no_absorb_unchanged(data):
    mod = IV2SLS(data.y, data.x, data.endog, data.z, absorb=data.absorb)
    assert mod.absorb is not None
    assert list(mod.absorb.columns) == ["firm", "year"]
    mod_plain = IV2SLS(data.y, data.x, data.endog, data.z)
    assert mod_plain.absorb is None


def _peak_rss_mib():
    """Peak RSS of this process in MiB, or None if unavailable"""
    try:
        import resource  # noqa: PLC0415
    except ImportError:
        return None
    # Linux reports ru_maxrss in KiB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


@pytest.mark.slow
def test_memory_large_scale():
    """
    Memory check: 100k observations, 20k + 50 effects, few regressors

    The peak memory added by constructing and fitting the model must stay
    below 512 MiB, excluding the input data and the Python interpreter
    overhead (both allocated before the measurement starts).
    """
    baseline = _peak_rss_mib()
    if baseline is None:
        pytest.skip("resource module not available on this platform")
    rs = np.random.RandomState(20240101)
    nobs, g1, g2 = 100_000, 20_000, 50
    c1 = rs.randint(0, g1, nobs)
    c2 = rs.randint(0, g2, nobs)
    absorb = pd.DataFrame({"firm": pd.Categorical(c1), "year": pd.Categorical(c2)})
    fe = rs.standard_normal(g1)[c1] + 0.5 * rs.standard_normal(g2)[c2]
    z = rs.standard_normal((nobs, 3))
    x = rs.standard_normal((nobs, 2))
    endog = 0.4 * z.sum(1) + x.sum(1) + rs.standard_normal(nobs)
    y = x @ np.array([0.5, 1.0]) + 0.8 * endog + fe + rs.standard_normal(nobs)
    w = rs.uniform(0.5, 2.0, nobs)
    clusters = rs.randint(0, 500, nobs)
    before = _peak_rss_mib()
    mod = IV2SLS(y, x, endog[:, None], z, weights=w, absorb=absorb)
    res = mod.fit(cov_type="clustered", debiased=True, clusters=clusters)
    after = _peak_rss_mib()
    added = after - before
    # Sanity checks on the fit itself
    assert res.nobs == nobs
    assert res.df_absorbed == absorb["firm"].nunique() + g2 - 1
    assert added < 512, f"fit added {added:.1f} MiB of peak memory (limit 512)"


@pytest.mark.parametrize("cov_type", ["homoskedastic", "heteroskedastic", "one-way"])
def test_cov_type_aliases(data, cov_type):
    kw = {"clusters": data.clusters} if cov_type == "one-way" else {}
    mod = IV2SLS(data.y, data.x, data.endog, data.z, weights=data.w, absorb=data.absorb)
    res = mod.fit(cov_type=cov_type, **kw)
    assert res.params.shape[0] == data.x.shape[1] + 1


def test_clusters_nested_in_effects(data):
    # Clusters identical to one absorbed effect: the same finite-sample
    # adjustment is used as in the explicit dummy model (no PanelOLS-style
    # automatic degree-of-freedom policy)
    clusters = data.absorb["firm"].cat.codes.to_numpy()
    kw = {"clusters": clusters}
    mod = IV2SLS(data.y, data.x, data.endog, data.z, weights=data.w, absorb=data.absorb)
    res = mod.fit(cov_type="clustered", debiased=True, **kw)
    mod_ref, _ = explicit_model(data)
    res_ref = mod_ref.fit(cov_type="clustered", debiased=True, **kw)
    k_exog, k_endog = data.x.shape[1], data.endog.shape[1]
    assert_allclose(
        res.cov.to_numpy(),
        structural_cov(res_ref.cov.to_numpy(), k_exog, k_endog),
        rtol=RTOL,
        atol=ATOL,
    )


def test_unused_categories_in_input(data):
    absorb = data.absorb.copy()
    absorb["firm"] = absorb["firm"].cat.add_categories("never-observed")
    assert (
        len(absorb["firm"].cat.categories)
        == len(data.absorb["firm"].cat.categories) + 1
    )
    mod = IV2SLS(data.y, data.x, data.endog, data.z, weights=data.w, absorb=absorb)
    res = mod.fit(cov_type="robust")
    # Unused categories do not count towards the absorbed degrees of freedom
    assert (
        res.df_absorbed
        == data.absorb["firm"].nunique() + data.absorb["year"].nunique() - 1
    )
    mod2 = IV2SLS(
        data.y, data.x, data.endog, data.z, weights=data.w, absorb=data.absorb
    )
    res2 = mod2.fit(cov_type="robust")
    assert_allclose(res.params.to_numpy(), res2.params.to_numpy())


def test_formula_no_exog(data):
    df = pd.concat([data.y, data.endog, data.z], axis=1)
    mod = IV2SLS.from_formula("y ~ [endog ~ z0 + z1 + z2]", df, absorb=data.absorb)
    res = mod.fit(cov_type="robust")
    assert res.params.shape[0] == 1
    assert "endog" in res.params.index


def test_absorb_duplicate_columns(data):
    absorb_dup = data.absorb.copy()
    absorb_dup.columns = ["firm", "firm"]
    with pytest.raises(ValueError, match="duplicate"):
        IV2SLS(data.y, data.x, data.endog, data.z, absorb=absorb_dup)
