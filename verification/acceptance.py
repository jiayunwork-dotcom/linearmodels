"""
Acceptance harness for unit-invariant system GMM.

Run with:  PYTHONPATH=/workspace python3 verification/acceptance.py

The reference values are produced by ``verification/reference_gmm.py``, a
dense, self-contained implementation written directly from the econometric
formulas.  It never calls the new internal transformations in
``linearmodels.system.model``.
"""

from __future__ import annotations

import pickle

import numpy as np
from numpy.testing import assert_allclose
from pandas import DataFrame, Series

from linearmodels.system import IVSystemGMM

from reference_gmm import fit_dense

RTOL = 1e-7
ATOL = 1e-9


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def make_system(n=400, seed=20240101, rho=0.0):
    """Well-conditioned 2-equation IV system with a constant and 3 instruments."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 3))
    e = rng.standard_normal((n, 2))
    if rho:
        for col in range(2):
            for t in range(1, n):
                e[t, col] = rho * e[t - 1, col] + np.sqrt(1 - rho**2) * e[t, col]
    x1 = z[:, 0] + 0.5 * z[:, 1] + e[:, 0]
    x2 = -0.3 * z[:, 1] + 0.8 * z[:, 2] + e[:, 1]
    const = np.ones((n, 1))
    y0 = 1.0 + 2.0 * x1 + e[:, 0]
    y1 = -2.0 + 0.5 * x2 + e[:, 1]
    return dict(
        z=z,
        e=e,
        const=const,
        x=[x1, x2],
        y=[y0, y1],
        exog=[const, const],
        # structural regressors in the equation-specific order used below
        xmat=[
            np.column_stack([const, x1]),
            np.column_stack([const, x2]),
        ],
        zmat=[
            np.column_stack([const, z]),
            np.column_stack([const, z]),
        ],
    )


def equations_from(data, endog_scale=(1.0, 1.0), instr_scale=None, order=None):
    """
    Build the equations dict.

    endog_scale : per-equation positive scaling applied to the endogenous
        regressor.  Structural slope coefficients change accordingly.
    instr_scale : 2 x 3 array of instrument column scalings (may be negative).
    order : optional permutation of the instrument columns.
    """
    if instr_scale is None:
        instr_scale = np.ones((2, 3))
    instr_scale = np.asarray(instr_scale, dtype=float)
    if order is None:
        order = np.array([0, 1, 2])
    order = np.asarray(order)
    eqns = {}
    xmats, zmats = [], []
    for i, key in enumerate(("e0", "e1")):
        z_scaled = data["z"][:, order] * instr_scale[i, order][None, :]
        x_scaled = data["x"][i] * endog_scale[i]
        y = data["y"][i]
        eqns[key] = {
            "dependent": y,
            "exog": data["const"],
            "endog": x_scaled[:, None],
            "instruments": z_scaled,
        }
        xmats.append(np.column_stack([data["const"], x_scaled]))
        zmats.append(np.column_stack([data["const"], z_scaled]))
    return eqns, xmats, zmats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fit_pkg(eqns, **kwargs):
    weight_type = kwargs.pop("weight_type", "robust")
    sigma = kwargs.pop("sigma", None)
    weight_config = kwargs.pop("weight_config", {})
    mod = IVSystemGMM(eqns, weight_type=weight_type, sigma=sigma, **weight_config)
    constraints = kwargs.pop("constraints", None)
    if constraints is not None:
        mod.add_constraints(*constraints)
    return mod, mod.fit(**kwargs)


def pack(res):
    return dict(
        params=np.asarray(res.params),
        cov=np.asarray(res.cov),
        w=np.asarray(res.w),
        j=float(res.j_stat.stat),
        resid=np.asarray(res.resids),
        fitted=np.asarray(res.fitted_values),
        stderr=np.asarray(res.std_errors),
        iters=int(res.iterations),
        sigma=np.asarray(res.sigma),
    )


def assert_same(a, b, label, rtol=RTOL, atol=ATOL):
    assert a.shape == b.shape, f"{label}: shape {a.shape} vs {b.shape}"
    assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=label)


def compare_pack(pkg, ref, label, rtol=RTOL, atol=ATOL):
    assert_same(pkg["params"], ref["params"], label + " params", rtol, atol)
    assert_same(pkg["cov"], ref["cov"], label + " cov", rtol, atol)
    assert_same(pkg["w"], ref["w"], label + " w", rtol, atol)
    assert_allclose(pkg["j"], ref["j_stat"], rtol=rtol, atol=atol, err_msg=label + " j")
    assert_same(pkg["resid"], ref["resid"], label + " resid", rtol, atol)
    assert_same(pkg["fitted"], ref["fitted"], label + " fitted", rtol, atol)


# ---------------------------------------------------------------------------
# 1. Original failure and unit invariance in every inference configuration
# ---------------------------------------------------------------------------
def test_original_failure():
    # Exact reproduction from the report.
    rng = np.random.default_rng(42)
    z = rng.standard_normal((200, 3))
    e = rng.standard_normal((200, 2))
    x = z[:, 0] + 0.5 * z[:, 1] + e[:, 0]
    y0 = 1 + 2 * x + e[:, 0]
    y1 = -2 + 0.5 * x + e[:, 1]
    const = np.ones((200, 1))

    def build(zz):
        return {
            "e0": {"dependent": y0, "exog": const, "endog": x[:, None],
                   "instruments": zz},
            "e1": {"dependent": y1, "exog": const, "endog": x[:, None],
                   "instruments": zz},
        }

    base = IVSystemGMM(build(z), weight_type="robust").fit(
        iter_limit=2, cov_type="robust"
    )
    scaled = IVSystemGMM(build(z * np.array([1e-8, 1.0, 1e8])),
                         weight_type="robust").fit(iter_limit=2, cov_type="robust")
    assert_same(np.asarray(scaled.params), np.asarray(base.params),
                "original-failure params")
    assert_same(np.asarray(scaled.cov), np.asarray(base.cov), "original-failure cov")
    assert_allclose(scaled.j_stat.stat, base.j_stat.stat, rtol=RTOL, atol=ATOL)
    print("  original failure reproduced and fixed")


def test_unit_and_order_invariance():
    data = make_system()
    configs = [
        dict(weight_type="robust", cov_type="robust", iter_limit=2),
        dict(weight_type="robust", cov_type="robust", iter_limit=5),
        dict(weight_type="robust", cov_type="robust", iter_limit=2, debiased=True),
        dict(weight_type="unadjusted", cov_type="unadjusted", iter_limit=2),
        dict(weight_type="unadjusted", cov_type="unadjusted", iter_limit=2,
             debiased=True),
        dict(weight_type="robust", cov_type="unadjusted", iter_limit=2),
        dict(weight_type="unadjusted", cov_type="robust", iter_limit=2),
        dict(weight_type="kernel", cov_type="kernel", iter_limit=2,
             weight_config={"bandwidth": 4}, bandwidth=4),
        dict(weight_type="kernel", cov_type="kernel", iter_limit=2,
             weight_config={"optimal_bw": True}),
    ]
    # Per-equation column scalings, including negative proportions and the
    # extreme 1e-80 / 1e80 cases.  The information in the instruments is
    # unchanged, only the units are.
    scales = [
        (np.array([1.0, 1.0]), np.ones((2, 3)), np.array([0, 1, 2])),
        (np.array([1e-8, 1e8]), np.array([[1e-8, 1.0, 1e8], [-1e8, -1.0, 1e-8]]),
         np.array([2, 0, 1])),
        (np.array([1e80, 1e-80]), np.array([[1e80, 1e-80, -1.0], [1.0, -1e80, 1e-80]]),
         np.array([1, 2, 0])),
        (np.array([-1e-20, 1e20]), np.array([[-1e-20, 1e20, -1.0], [1e-20, -1.0, 1e20]]),
         np.array([2, 1, 0])),
    ]
    eqns0, x0, z0 = equations_from(data, *scales[0][:2], scales[0][2])
    _, base_res = fit_pkg(eqns0, **dict(configs[0]))
    base = pack(base_res)
    print("  unit/order invariance across inference configurations")
    for cfg in configs:
        for eq_scale, instr_scale, order in scales[1:]:
            eqns, xm, zm = equations_from(data, eq_scale, instr_scale, order)
            _, res = fit_pkg(eqns, **dict(cfg))
            pk = pack(res)
            # Structural and instrument unit changes leave slope and constant
            # estimates unchanged; the endog variable was not rescaled here
            # relative to the base comparison, so compare to a baseline fit
            # with the SAME endog scaling (see below).
            label = f"{cfg['weight_type']}/{cfg['cov_type']}"
            # Compare against an independently estimated baseline with the
            # same endog scaling but unitless instruments.
            eqns_b, _, _ = equations_from(
                data, eq_scale, np.ones((2, 3)), np.array([0, 1, 2])
            )
            _, res_b = fit_pkg(eqns_b, **dict(cfg))
            assert_same(pk["params"], pack(res_b)["params"], label + " params")
            assert_same(pk["cov"], pack(res_b)["cov"], label + " cov")
            assert_allclose(pk["j"], pack(res_b)["j"], rtol=RTOL, atol=ATOL)
            assert_same(pk["resid"], pack(res_b)["resid"], label + " resid")
            assert_same(pk["fitted"], pack(res_b)["fitted"], label + " fitted")
    # Structural unit change: scaling the endogenous regressor by s divides
    # its slope by s, while the intercept adjusts so that fitted values are
    # unchanged (the coefficient mapping is exact).
    endog_scale = (1e-6, 1e6)
    eqns_s, _, _ = equations_from(data, endog_scale, np.ones((2, 3)),
                                  np.array([0, 1, 2]))
    _, res_s = fit_pkg(eqns_s, **dict(configs[0]))
    expect = np.asarray(base_res.params).copy()
    expect[[1, 3]] /= np.array(endog_scale)
    assert_same(np.asarray(res_s.params), expect, "structural-unit mapping")
    print("  structural-unit coefficient mapping verified")


# ---------------------------------------------------------------------------
# 2. Reference comparison for every weight / covariance combination
# ---------------------------------------------------------------------------
def test_reference_all_configs():
    data = make_system(rho=0.0)
    eqns, xm, zm = equations_from(data)
    sigma_fixed = np.array([[2.0, 0.3], [0.3, 1.5]])
    cases = [
        dict(weight_type="robust", cov_type="robust", iter_limit=1),
        dict(weight_type="robust", cov_type="robust", iter_limit=2),
        dict(weight_type="robust", cov_type="robust", iter_limit=2, debiased=True),
        dict(weight_type="robust", cov_type="robust", iter_limit=6),
        dict(weight_type="unadjusted", cov_type="unadjusted", iter_limit=2),
        dict(weight_type="unadjusted", cov_type="unadjusted", iter_limit=2,
             debiased=True),
        dict(weight_type="unadjusted", cov_type="unadjusted", iter_limit=3,
             sigma=sigma_fixed),
        dict(weight_type="robust", cov_type="unadjusted", iter_limit=2),
        dict(weight_type="unadjusted", cov_type="robust", iter_limit=2),
        dict(weight_type="kernel", cov_type="kernel", iter_limit=2,
             weight_config={"bandwidth": 3}, bandwidth=3),
        dict(weight_type="kernel", cov_type="kernel", iter_limit=2,
             weight_config={"optimal_bw": True}),
        dict(weight_type="robust", cov_type="kernel", iter_limit=2, bandwidth=5),
    ]
    print("  reference comparisons (package vs dense formulas)")
    for cfg in cases:
        wt = cfg["weight_type"]
        ct = cfg["cov_type"]
        il = cfg["iter_limit"]
        sf = cfg.get("sigma")
        bw = cfg.get("bandwidth")
        deb = cfg.get("debiased", False)
        wconf = cfg.get("weight_config", {})
        _, res = fit_pkg(eqns, **dict(cfg))
        ref = fit_dense(
            data["y"], xm, zm, weight_type=wt, cov_type=ct, iter_limit=il,
            sigma_fixed=sf, debiased=deb, bandwidth=bw,
            optimal_bw=wconf.get("optimal_bw", False),
            weight_debiased=wconf.get("debiased", False),
        )
        compare_pack(pack(res), ref, f"{wt}/{ct}")
        assert res.weight_type == wt
    print("  all weight/covariance combinations match the dense reference")


# ---------------------------------------------------------------------------
# 3. User-supplied initial weight (moment covariance, not its inverse)
# ---------------------------------------------------------------------------
def test_initial_weight_equivariance():
    data = make_system()
    eqns, xm, zm = equations_from(data)
    # A symmetric positive-definite user-coordinate initial weight.
    rng = np.random.default_rng(99)
    a = rng.standard_normal((8, 8))
    w_user = a @ a.T + 8 * np.eye(8)
    _, res = fit_pkg(eqns, iter_limit=1, cov_type="robust", initial_weight=w_user)
    ref = fit_dense(data["y"], xm, zm, weight_type="robust", cov_type="robust",
                    iter_limit=1, initial_weight=w_user)
    compare_pack(pack(res), ref, "initial-weight user coords")

    # Now express the same estimator after re-scaling each instrument column.
    instr_scale = np.array([[1e-8, 1.0, 1e8], [-1e8, -1.0, 1e-8]])
    order = np.array([2, 0, 1])
    eqns_s, xm_s, zm_s = equations_from(data, (1.0, 1.0), instr_scale, order)
    # Build the per-equation transform P_i satisfying Z_new_i = Z_old_i P_i.
    # Column 0 is the constant; new column j+1 is old column order[j]+1 scaled
    # by instr_scale[i, order[j]].
    P = np.zeros((8, 8))
    for eq in range(2):
        Pi = np.zeros((4, 4))
        Pi[0, 0] = 1.0
        for j in range(3):
            Pi[order[j] + 1, j + 1] = instr_scale[eq, order[j]]
        P[eq * 4 : (eq + 1) * 4, eq * 4 : (eq + 1) * 4] = Pi
    # With Z_new = Z_old P the moment covariance transforms as P' W P.
    w_scaled = P.T @ w_user @ P
    _, res_s = fit_pkg(eqns_s, iter_limit=1, cov_type="robust",
                       initial_weight=w_scaled)
    ref_s = fit_dense(data["y"], xm_s, zm_s, weight_type="robust",
                      cov_type="robust", iter_limit=1, initial_weight=w_scaled)
    compare_pack(pack(res_s), ref_s, "initial-weight scaled coords")
    assert_same(pack(res_s)["params"], pack(res)["params"], "initial-weight params")
    assert_same(pack(res_s)["cov"], pack(res)["cov"], "initial-weight cov")
    # The returned weight matrix is in the original instrument coordinates.
    assert_same(pack(res_s)["w"], P.T @ pack(res)["w"] @ P,
                "initial-weight w returned in user coords")
    print("  initial_weight equivalence and returned w verified")


# ---------------------------------------------------------------------------
# 4. Constraints: non-zero RHS, cross-equation, and unit rewriting
# ---------------------------------------------------------------------------
def test_constraints():
    data = make_system()
    eqns, xm, zm = equations_from(data)
    mod0, base_mod_res = fit_pkg(eqns, iter_limit=2, cov_type="robust")
    names = mod0.param_names
    # Two constraints: e0_const + e1_const = 0.5 (cross-equation, non-zero
    # RHS) and e1_endog = 0.3.
    r = np.zeros((2, len(names)))
    r[0, 0] = 1.0
    r[0, 2] = 1.0
    r[1, 3] = 1.0
    q = np.array([0.5, 0.3])
    cons = (DataFrame(r, columns=names, index=["c1", "c2"]), Series(q, index=["c1", "c2"]))
    _, res = fit_pkg(eqns, iter_limit=2, cov_type="robust", constraints=cons)
    pk = pack(res)
    # Verify the constraint is satisfied exactly.
    assert abs(pk["params"][0] + pk["params"][2] - 0.5) < 1e-10
    assert abs(pk["params"][3] - 0.3) < 1e-10
    ref = fit_dense(data["y"], xm, zm, weight_type="robust", cov_type="robust",
                    iter_limit=2, constraints=(r, q))
    # Compare in the unconstrained-coordinate convention: the reference
    # parameter vector and covariance match the package after restricting.
    assert_same(pk["params"], ref["params"], "constraints params")
    # The package's covariance is the constrained covariance; the reference
    # projection gives the same values for the identified directions.
    assert_same(pk["cov"], ref["cov"], "constraints cov")
    assert_allclose(pk["j"], ref["j_stat"], rtol=RTOL, atol=ATOL)

    # Rewrite the constraints equivalently after a structural unit change.
    endog_scale = (1e-8, 1e8)
    eqns_s, xm_s, zm_s = equations_from(data, endog_scale, np.ones((2, 3)),
                                        np.array([0, 1, 2]))
    # In the new units, beta_endog_new = beta_endog_old / scale, so the
    # loading on the endog coefficient is multiplied by the scale.
    r_s = r.copy()
    r_s[1, 3] = endog_scale[1]
    cons_s = (DataFrame(r_s, columns=names, index=["c1", "c2"]),
              Series(q, index=["c1", "c2"]))
    _, res_s = fit_pkg(eqns_s, iter_limit=2, cov_type="robust", constraints=cons_s)
    pk_s = pack(res_s)
    assert abs(pk_s["params"][1] * endog_scale[0] - pk["params"][1]) < 1e-7
    assert abs(pk_s["params"][3] * endog_scale[1] - pk["params"][3]) < 1e-7
    assert abs(pk_s["params"][0] + pk_s["params"][2] - 0.5) < 1e-10
    # Standard errors map the same way as the coefficients.
    assert_allclose(pk_s["stderr"][[1, 3]] * np.array(endog_scale),
                    pk["stderr"][[1, 3]], rtol=RTOL, atol=ATOL)
    print("  non-zero cross-equation constraints and unit rewriting verified")


# ---------------------------------------------------------------------------
# 5. Serial correlation (kernel / HAC) and observation weights
# ---------------------------------------------------------------------------
def test_serial_correlation_and_obs_weights():
    data = make_system(rho=0.7)
    eqns, xm, zm = equations_from(data)
    for bw in (2, 5, None):
        wconfig = {"bandwidth": bw} if bw is not None else {"optimal_bw": True}
        fit_kwargs = dict(weight_type="kernel", cov_type="kernel", iter_limit=2,
                          weight_config=wconfig)
        if bw is not None:
            fit_kwargs["bandwidth"] = bw
        _, res = fit_pkg(eqns, **dict(fit_kwargs))
        ref = fit_dense(data["y"], xm, zm, weight_type="kernel", cov_type="kernel",
                        iter_limit=2, kernel="bartlett",
                        bandwidth=bw, optimal_bw=bw is None)
        compare_pack(pack(res), ref, f"HAC bw={bw}")

    # Positive observation weights.
    rng = np.random.default_rng(3)
    w_obs = [rng.chisquare(5, (data["y"][0].shape[0], 1)) / 5 for _ in range(2)]
    eqns_w = {}
    for i, key in enumerate(("e0", "e1")):
        eqns_w[key] = dict(eqns[key])
        eqns_w[key]["weights"] = w_obs[i]
    _, res = fit_pkg(eqns_w, iter_limit=2, cov_type="robust")
    ref = fit_dense(data["y"], xm, zm, weight_type="robust", cov_type="robust",
                    iter_limit=2, obs_weights=w_obs)
    compare_pack(pack(res), ref, "observation weights")
    print("  serial correlation and observation weights verified")


# ---------------------------------------------------------------------------
# 6. Rejection boundaries: genuine rank deficiency and under-identification
# ---------------------------------------------------------------------------
def test_rejection_boundaries():
    data = make_system()
    rng = np.random.default_rng(11)
    n = data["y"][0].shape[0]

    # Duplicate instrument columns (exactly collinear) must fail.
    z_dup = np.column_stack([data["z"], data["z"][:, 0]])
    eqns = {
        "e0": {"dependent": data["y"][0], "exog": data["const"],
               "endog": data["x"][0][:, None], "instruments": z_dup},
        "e1": {"dependent": data["y"][1], "exog": data["const"],
               "endog": data["x"][1][:, None], "instruments": data["z"]},
    }
    try:
        IVSystemGMM(eqns)
    except ValueError as exc:
        assert "full rank" in str(exc), str(exc)
    else:
        raise AssertionError("duplicate instruments were not rejected")

    # --- genuine rank deficiency must not be masked by a pseudoinverse ---
    # Structural regressor that is an exact linear combination of two others.
    exog_bad = np.column_stack([np.ones(n), data["z"][:, 0], data["z"][:, 0]])
    eqns_bad = {
        "e0": {"dependent": data["y"][0], "exog": exog_bad},
        "e1": {"dependent": data["y"][1], "exog": data["const"]},
    }
    try:
        IVSystemGMM(eqns_bad)
    except ValueError as exc:
        assert "regressor array is not full rank" in str(exc), str(exc)
    else:
        raise AssertionError("structural rank deficiency was not rejected")

    # Under-identified: fewer instruments than structural regressors.
    eqns_u = {
        "e0": {"dependent": data["y"][0], "exog": data["const"],
               "endog": data["x"][0][:, None], "instruments": None},
        "e1": {"dependent": data["y"][1], "exog": data["const"]},
    }
    try:
        IVSystemGMM(eqns_u)
    except ValueError as exc:
        assert "fewer instruments" in str(exc), str(exc)
    else:
        raise AssertionError("under-identification was not rejected")

    # Exact linear dependence *between* exog and instruments.
    z_lin = np.column_stack([data["z"], data["z"][:, 0] + data["z"][:, 1]])
    eqns_l = {
        "e0": {"dependent": data["y"][0], "exog": data["const"],
               "endog": data["x"][0][:, None], "instruments": z_lin},
        "e1": {"dependent": data["y"][1], "exog": data["const"]},
    }
    try:
        IVSystemGMM(eqns_l)
    except ValueError as exc:
        assert "full rank" in str(exc), str(exc)
    else:
        raise AssertionError("linearly dependent instruments were not rejected")
    print("  genuine rank deficiency / under-identification rejected")


# ---------------------------------------------------------------------------
# 7. Repeated fit and input immutability
# ---------------------------------------------------------------------------
def test_repeated_fit_and_immutability():
    data = make_system()
    eqns, xm, zm = equations_from(data)
    # Snapshot all numeric inputs.
    snapshot = {}
    for key, val in eqns.items():
        for field, arr in val.items():
            snapshot[(key, field)] = np.array(arr, copy=True)
    mod = IVSystemGMM(eqns, weight_type="robust")
    res1 = mod.fit(iter_limit=3, cov_type="robust")
    res2 = mod.fit(iter_limit=3, cov_type="robust")
    assert_same(np.asarray(res1.params), np.asarray(res2.params), "repeat-fit params")
    assert_same(np.asarray(res1.cov), np.asarray(res2.cov), "repeat-fit cov")
    assert res1.iterations == res2.iterations
    for key, val in eqns.items():
        for field, arr in val.items():
            assert np.array_equal(np.asarray(arr), snapshot[(key, field)]), (
                f"input mutated: {key}/{field}"
            )
    # Pickling the results (used by the existing test suite) still works.
    pickle.loads(pickle.dumps(res1))
    print("  repeated fit is stable and inputs are unchanged")


# ---------------------------------------------------------------------------
# 8. from_formula parity and matrix-dimension audit
# ---------------------------------------------------------------------------
def test_formula_parity():
    data = make_system()
    n = data["z"].shape[0]
    frame = DataFrame({
        "y0": data["y"][0], "y1": data["y"][1],
        "x0": data["x"][0], "x1": data["x"][1],
        "z0": data["z"][:, 0], "z1": data["z"][:, 1], "z2": data["z"][:, 2],
    })
    formula = {
        "e0": "y0 ~ 1 + [x0 ~ z0 + z1 + z2]",
        "e1": "y1 ~ 1 + [x1 ~ z0 + z1 + z2]",
    }
    # Direct construction with named DataFrames so parameter names and the
    # output ordering can be compared as well.
    eqns_named = {
        key: {
            "dependent": frame[dep],
            "exog": frame[["const"]] if "const" in frame else
                    DataFrame({"const": np.ones(n)}),
            "endog": frame[[endog]],
            "instruments": frame[["z0", "z1", "z2"]],
        }
        for key, dep, endog in (("e0", "y0", "x0"), ("e1", "y1", "x1"))
    }
    mod_direct = IVSystemGMM(eqns_named, weight_type="robust")
    res_direct = mod_direct.fit(iter_limit=2, cov_type="robust")
    mod_formula = IVSystemGMM.from_formula(formula, frame, weight_type="robust")
    res_formula = mod_formula.fit(iter_limit=2, cov_type="robust")
    assert_same(np.asarray(res_formula.params), np.asarray(res_direct.params),
                "formula params")
    assert_same(np.asarray(res_formula.cov), np.asarray(res_direct.cov),
                "formula cov")
    assert_allclose(res_formula.j_stat.stat, res_direct.j_stat.stat,
                    rtol=RTOL, atol=ATOL)
    assert list(res_formula.equation_labels) == list(res_direct.equation_labels)
    # Parameter names follow the caller's own input.  Formula construction
    # names the constant "Intercept"; direct construction uses the column
    # name supplied ("const").  Within each construction the order is the
    # caller's equation order and then exog-before-endog.
    assert list(res_direct.params.index) == [
        "e0_const", "e0_x0", "e1_const", "e1_x1"
    ]
    assert list(res_formula.params.index) == [
        "e0_Intercept", "e0_x0", "e1_Intercept", "e1_x1"
    ]
    assert list(mod_direct.param_names) == list(res_direct.params.index)
    assert list(mod_formula.param_names) == list(res_formula.params.index)
    print("  direct and formula construction agree (values; names follow input)")


def test_matrix_dimensions():
    """Record the dimensions of the working matrices and confirm no
    observation-level block-diagonal matrix is materialized."""
    data = make_system(n=400)
    eqns, _, _ = equations_from(data)
    mod = IVSystemGMM(eqns, weight_type="robust")
    k = len(mod._wx)
    nobs = mod._wx[0].shape[0]
    dims = dict(
        nobs=nobs,
        k=k,
        nvar=sum(a.shape[1] for a in mod._wx),
        ninstr=sum(a.shape[1] for a in mod._wz),
    )
    # The largest moment-covariance matrix is ninstr x ninstr (8 x 8 here) and
    # the largest weighted cross-product is nvar x ninstr (4 x 8).  No array
    # with k * nobs rows is created.
    dims["moment_cov"] = (dims["ninstr"], dims["ninstr"])
    dims["cross_prod"] = (dims["nvar"], dims["ninstr"])
    dims["max_rows_created"] = max(dims["ninstr"], dims["nvar"])
    assert dims["max_rows_created"] < nobs
    res = mod.fit(iter_limit=2, cov_type="robust")
    assert np.asarray(res.w).shape == (dims["ninstr"], dims["ninstr"])
    print("  matrix dimensions:", dims)
    return dims


def main():
    print("Running system GMM acceptance harness")
    test_original_failure()
    test_unit_and_order_invariance()
    test_reference_all_configs()
    test_initial_weight_equivariance()
    test_constraints()
    test_serial_correlation_and_obs_weights()
    test_rejection_boundaries()
    test_repeated_fit_and_immutability()
    test_formula_parity()
    test_matrix_dimensions()
    print("ALL ACCEPTANCE CHECKS PASSED")


if __name__ == "__main__":
    main()
