"""Regression tests for unit invariance of IVSystemGMM.

These tests cover the reported failure in which re-scaling instrument columns
made a well-identified model appear rank deficient.  They are deliberately
written without calling any of the new internal helpers so that they can be
used to detect a regression in the public behaviour.
"""

import numpy as np
from numpy.testing import assert_allclose
from pandas import DataFrame, Series
import pytest

from linearmodels.system import IVSystemGMM

RTOL = 1e-8
ATOL = 1e-10


def _system(n=300, seed=7):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 3))
    e = rng.standard_normal((n, 2))
    x = z[:, 0] + 0.5 * z[:, 1] + e[:, 0]
    const = np.ones((n, 1))
    return {
        "z": z, "x": x, "const": const,
        "y0": 1.0 + 2.0 * x + e[:, 0],
        "y1": -2.0 + 0.5 * x + e[:, 1],
    }


def _eqns(d, z_scale=None, endog_scale=(1.0, 1.0), order=(0, 1, 2)):
    if z_scale is None:
        z_scale = np.ones((2, 3))
    order = np.asarray(order)
    eqns = {}
    for i, key in enumerate(("e0", "e1")):
        z = d["z"][:, order] * np.asarray(z_scale)[i, order][None, :]
        eqns[key] = {
            "dependent": d[f"y{i}"],
            "exog": d["const"],
            "endog": (d["x"] * endog_scale[i])[:, None],
            "instruments": z,
        }
    return eqns


def test_original_reported_failure():
    # The exact reproduction from the bug report.
    rng = np.random.default_rng(42)
    z = rng.standard_normal((200, 3))
    e = rng.standard_normal((200, 2))
    x = z[:, 0] + 0.5 * z[:, 1] + e[:, 0]
    const = np.ones((200, 1))

    def build(zz):
        return {
            "e0": {"dependent": 1 + 2 * x + e[:, 0], "exog": const,
                   "endog": x[:, None], "instruments": zz},
            "e1": {"dependent": -2 + 0.5 * x + e[:, 1], "exog": const,
                   "endog": x[:, None], "instruments": zz},
        }

    base = IVSystemGMM(build(z), weight_type="robust").fit(
        iter_limit=2, cov_type="robust"
    )
    # This used to raise "Equation e0 instrument array is full rank".
    scaled = IVSystemGMM(build(z * np.array([1e-8, 1.0, 1e8])),
                         weight_type="robust").fit(iter_limit=2, cov_type="robust")
    assert_allclose(scaled.params, base.params, rtol=RTOL, atol=ATOL)
    assert_allclose(np.asarray(scaled.cov), np.asarray(base.cov),
                    rtol=RTOL, atol=ATOL)
    assert_allclose(scaled.j_stat.stat, base.j_stat.stat, rtol=RTOL, atol=ATOL)
    assert_allclose(np.asarray(scaled.resids), np.asarray(base.resids),
                    rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("weight_type", ["robust", "unadjusted", "kernel"])
@pytest.mark.parametrize("cov_type", ["robust", "unadjusted"])
def test_unit_and_order_invariance(weight_type, cov_type):
    d = _system()
    wconfig = {}
    if weight_type == "kernel":
        wconfig["bandwidth"] = 4
    base = IVSystemGMM(_eqns(d), weight_type=weight_type, **wconfig).fit(
        iter_limit=2, cov_type=cov_type
    )
    # Extreme, sign-flipping per-equation units and a column permutation.
    scales = np.array([[1e-8, 1.0, 1e8], [-1e8, -1.0, 1e-8]])
    scaled = IVSystemGMM(
        _eqns(d, scales, order=(2, 0, 1)), weight_type=weight_type, **wconfig
    ).fit(iter_limit=2, cov_type=cov_type)
    assert_allclose(scaled.params, base.params, rtol=RTOL, atol=ATOL)
    assert_allclose(np.asarray(scaled.cov), np.asarray(base.cov),
                    rtol=RTOL, atol=ATOL)
    assert_allclose(scaled.j_stat.stat, base.j_stat.stat, rtol=RTOL, atol=ATOL)
    # ``w`` is returned in the caller's instrument coordinates, so it is
    # mapped (not invariant) under a unit change; that mapping is checked in
    # test_initial_weight_equivariance_and_user_coordinate_w.


def test_structural_unit_mapping_and_inference():
    d = _system()
    base = IVSystemGMM(_eqns(d), weight_type="robust").fit(
        iter_limit=2, cov_type="robust"
    )
    scale = (1e-6, 1e6)
    scaled = IVSystemGMM(_eqns(d, endog_scale=scale), weight_type="robust").fit(
        iter_limit=2, cov_type="robust"
    )
    expected = np.asarray(base.params).copy()
    expected[[1, 3]] /= np.asarray(scale)
    assert_allclose(scaled.params, expected, rtol=RTOL, atol=ATOL)
    std_expected = np.asarray(base.std_errors).copy()
    std_expected[[1, 3]] /= np.asarray(scale)
    assert_allclose(scaled.std_errors, std_expected, rtol=RTOL, atol=ATOL)
    # Residuals and fitted values do not depend on structural units.
    assert_allclose(np.asarray(scaled.resids), np.asarray(base.resids),
                    rtol=RTOL, atol=ATOL)
    assert_allclose(np.asarray(scaled.fitted_values),
                    np.asarray(base.fitted_values), rtol=RTOL, atol=ATOL)


def test_convergence_iterations_invariant_to_units():
    d = _system()
    iterations = []
    params = []
    for sc in (1.0, 1e-8, 1e8, -1e-12):
        z = d["z"] * np.array([sc, 1.0, 1.0 / sc])
        res = IVSystemGMM(_with_z(d, z), weight_type="robust").fit(
            iter_limit=20, cov_type="robust", tol=1e-10
        )
        iterations.append(res.iterations)
        params.append(np.asarray(res.params))
    assert len(set(iterations)) == 1
    for p in params:
        assert_allclose(p, params[0], rtol=1e-10, atol=1e-12)


def _with_z(d, z):
    return {
        "e0": {"dependent": d["y0"], "exog": d["const"],
               "endog": d["x"][:, None], "instruments": z},
        "e1": {"dependent": d["y1"], "exog": d["const"],
               "endog": d["x"][:, None], "instruments": z},
    }


def test_initial_weight_equivariance_and_user_coordinate_w():
    d = _system()
    rng = np.random.default_rng(1)
    a = rng.standard_normal((8, 8))
    w0 = a @ a.T + 8 * np.eye(8)
    base = IVSystemGMM(_eqns(d), weight_type="robust").fit(
        iter_limit=1, cov_type="robust", initial_weight=w0
    )

    scale = np.array([[1e-8, 1.0, 1e8], [-1e8, -1.0, 1e-8]])
    order = (2, 0, 1)
    P = np.zeros((8, 8))
    for eq in range(2):
        Pi = np.zeros((4, 4))
        Pi[0, 0] = 1.0
        for j in range(3):
            Pi[order[j] + 1, j + 1] = scale[eq, order[j]]
        P[eq * 4:(eq + 1) * 4, eq * 4:(eq + 1) * 4] = Pi
    w_scaled = P.T @ w0 @ P
    scaled = IVSystemGMM(_eqns(d, scale, order=order),
                         weight_type="robust").fit(
        iter_limit=1, cov_type="robust", initial_weight=w_scaled
    )
    assert_allclose(scaled.params, base.params, rtol=RTOL, atol=ATOL)
    assert_allclose(np.asarray(scaled.cov), np.asarray(base.cov),
                    rtol=RTOL, atol=ATOL)
    # The returned weight matrix is in the caller's instrument coordinates.
    assert_allclose(np.asarray(scaled.w), P.T @ np.asarray(base.w) @ P,
                    rtol=RTOL, atol=ATOL)


def test_nonzero_cross_equation_constraints_and_rewriting():
    d = _system()
    mod = IVSystemGMM(_eqns(d), weight_type="robust")
    names = mod.param_names
    r = np.zeros((2, len(names)))
    r[0, 0] = 1.0
    r[0, 2] = 1.0
    r[1, 3] = 1.0
    q = np.array([0.5, 0.3])
    mod.add_constraints(DataFrame(r, columns=names),
                        Series(q, index=names[:2]))
    res = mod.fit(iter_limit=2, cov_type="robust")
    params = np.asarray(res.params)
    assert abs(params[0] + params[2] - 0.5) < 1e-10
    assert abs(params[3] - 0.3) < 1e-10

    # Rewrite the constraints after a structural unit change and confirm the
    # estimates and standard errors match the original.
    scale = (1e-8, 1e8)
    mod_s = IVSystemGMM(_eqns(d, endog_scale=scale), weight_type="robust")
    r_s = r.copy()
    r_s[1, 3] = scale[1]  # beta_new = beta_old / scale
    mod_s.add_constraints(DataFrame(r_s, columns=names),
                          Series(q, index=names[:2]))
    res_s = mod_s.fit(iter_limit=2, cov_type="robust")
    p_s = np.asarray(res_s.params)
    assert_allclose(p_s[1] * scale[0], params[1], rtol=1e-7, atol=1e-9)
    assert_allclose(p_s[3] * scale[1], params[3], rtol=1e-7, atol=1e-9)
    se_s = np.asarray(res_s.std_errors)
    se = np.asarray(res.std_errors)
    assert_allclose(se_s[[1, 3]] * np.asarray(scale), se[[1, 3]],
                    rtol=1e-7, atol=1e-9)


def test_rank_deficiency_and_underidentification_still_fail():
    d = _system()
    n = d["z"].shape[0]
    # Duplicate instrument column: genuine rank failure.
    z_dup = np.column_stack([d["z"], d["z"][:, 0]])
    eqns = {
        "e0": {"dependent": d["y0"], "exog": d["const"],
               "endog": d["x"][:, None], "instruments": z_dup},
        "e1": {"dependent": d["y1"], "exog": d["const"],
               "endog": d["x"][:, None], "instruments": d["z"]},
    }
    with pytest.raises(ValueError, match="full rank"):
        IVSystemGMM(eqns)

    # Duplicated structural regressor.
    exog = np.column_stack([np.ones(n), d["z"][:, 0], d["z"][:, 0]])
    with pytest.raises(ValueError, match="regressor array is not full rank"):
        IVSystemGMM({
            "e0": {"dependent": d["y0"], "exog": exog},
            "e1": {"dependent": d["y1"], "exog": d["const"]},
        })

    # Under-identified: no excluded instrument for an endogenous regressor.
    with pytest.raises(ValueError, match="fewer instruments"):
        IVSystemGMM({
            "e0": {"dependent": d["y0"], "exog": d["const"],
                   "endog": d["x"][:, None]},
            "e1": {"dependent": d["y1"], "exog": d["const"]},
        })


def test_repeated_fit_and_input_immutability():
    d = _system()
    eqns = _eqns(d)
    snapshot = {k: {f: np.array(v, copy=True) for f, v in e.items()}
                for k, e in eqns.items()}
    mod = IVSystemGMM(eqns, weight_type="robust")
    r1 = mod.fit(iter_limit=3, cov_type="robust")
    r2 = mod.fit(iter_limit=3, cov_type="robust")
    assert_allclose(r1.params, r2.params, rtol=1e-12, atol=1e-12)
    assert_allclose(np.asarray(r1.cov), np.asarray(r2.cov),
                    rtol=1e-12, atol=1e-12)
    for k, e in eqns.items():
        for f, v in e.items():
            assert np.array_equal(np.asarray(v), snapshot[k][f])


def test_direct_and_formula_agree():
    d = _system()
    n = d["z"].shape[0]
    frame = DataFrame({
        "y0": d["y0"], "y1": d["y1"], "x0": d["x"], "x1": d["x"],
        "z0": d["z"][:, 0], "z1": d["z"][:, 1], "z2": d["z"][:, 2],
    })
    formula = {
        "e0": "y0 ~ 1 + [x0 ~ z0 + z1 + z2]",
        "e1": "y1 ~ 1 + [x1 ~ z0 + z1 + z2]",
    }
    eqns = {
        k: {"dependent": frame[dep],
            "exog": DataFrame({"const": np.ones(n)}),
            "endog": frame[[en]],
            "instruments": frame[["z0", "z1", "z2"]]}
        for k, dep, en in (("e0", "y0", "x0"), ("e1", "y1", "x1"))
    }
    direct = IVSystemGMM(eqns, weight_type="robust").fit(
        iter_limit=2, cov_type="robust"
    )
    via_formula = IVSystemGMM.from_formula(
        formula, frame, weight_type="robust"
    ).fit(iter_limit=2, cov_type="robust")
    assert_allclose(via_formula.params, direct.params, rtol=RTOL, atol=ATOL)
    assert_allclose(np.asarray(via_formula.cov), np.asarray(direct.cov),
                    rtol=RTOL, atol=ATOL)
    assert list(via_formula.equation_labels) == list(direct.equation_labels)
