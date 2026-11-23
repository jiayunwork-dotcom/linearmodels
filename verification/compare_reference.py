"""Compare the package estimator with the dense reference on well-conditioned data."""
import numpy as np
from numpy.testing import assert_allclose
import sys
sys.path.insert(0, "/workspace/verification")
from reference_gmm import fit_dense
from linearmodels.system import IVSystemGMM


def make_data(n=300, seed=7, rho=0.0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 3))
    # optional serially correlated errors
    e = rng.standard_normal((n, 2))
    if rho:
        for j in range(e.shape[1]):
            for t in range(1, n):
                e[t, j] = rho * e[t - 1, j] + np.sqrt(1 - rho**2) * e[t, j]
    x = z[:, 0] + 0.5 * z[:, 1] + e[:, 0]
    const = np.ones((n, 1))
    y0 = 1 + 2 * x + e[:, 0]
    y1 = -2 + 0.5 * x + e[:, 1]
    eqns = {
        "e0": {"dependent": y0, "exog": const, "endog": x[:, None], "instruments": z},
        "e1": {"dependent": y1, "exog": const, "endog": x[:, None], "instruments": z},
    }
    y = [y0, y1]
    xx = [np.column_stack([const, x]), np.column_stack([const, x])]
    zz = [np.column_stack([const, z]), np.column_stack([const, z])]
    return eqns, y, xx, zz


def main():
    cases = [
        dict(weight_type="robust", cov_type="robust", iter_limit=1),
        dict(weight_type="robust", cov_type="robust", iter_limit=2),
        dict(weight_type="robust", cov_type="robust", iter_limit=2, debiased=True),
        dict(weight_type="unadjusted", cov_type="unadjusted", iter_limit=2),
        dict(weight_type="unadjusted", cov_type="unadjusted", iter_limit=2, debiased=True),
        dict(weight_type="unadjusted", cov_type="unadjusted", iter_limit=3, sigma_fixed=np.array([[2.0, 0.3], [0.3, 1.5]])),
        dict(weight_type="robust", cov_type="unadjusted", iter_limit=2),
        dict(weight_type="unadjusted", cov_type="robust", iter_limit=2),
        dict(weight_type="kernel", cov_type="kernel", iter_limit=2, bandwidth=3),
        dict(weight_type="kernel", cov_type="kernel", iter_limit=2, optimal_bw=True),
        dict(weight_type="robust", cov_type="kernel", iter_limit=2, bandwidth=4),
        dict(weight_type="robust", cov_type="robust", iter_limit=5),
        dict(weight_type="robust", cov_type="robust", iter_limit=5, tol=1e-12),
    ]

    for rho in (0.0, 0.6):
        eqns, y, xx, zz = make_data(rho=rho)
        for i, c in enumerate(cases):
            kw = dict(c)
            wt = kw.pop("weight_type"); ct = kw.pop("cov_type")
            il = kw.pop("iter_limit"); sf = kw.pop("sigma_fixed", None)
            bw = kw.pop("bandwidth", None); obw = kw.pop("optimal_bw", False)
            deb = kw.pop("debiased", False); tol = kw.pop("tol", 1e-6)
            wconf = {}
            if wt == "kernel":
                if bw is not None:
                    wconf["bandwidth"] = bw
                if obw:
                    wconf["optimal_bw"] = True
            mod = IVSystemGMM(eqns, weight_type=wt, sigma=sf, **wconf)
            res = mod.fit(iter_limit=il, cov_type=ct, debiased=deb, tol=tol, bandwidth=bw) if bw is not None or ct == "kernel" else mod.fit(iter_limit=il, cov_type=ct, debiased=deb, tol=tol)
            ref = fit_dense(y, xx, zz, weight_type=wt, cov_type=ct, iter_limit=il,
                            sigma_fixed=sf, debiased=deb, bandwidth=bw,
                            optimal_bw=obw, tol=tol, kernel="bartlett")
            assert_allclose(res.params.values, ref["params"], rtol=1e-7, atol=1e-9)
            assert_allclose(np.asarray(res.cov), ref["cov"], rtol=1e-7, atol=1e-9)
            assert_allclose(np.asarray(res.w), ref["w"], rtol=1e-7, atol=1e-9)
            assert_allclose(res.j_stat.stat, ref["j_stat"], rtol=1e-7, atol=1e-9)
            assert_allclose(np.asarray(res.resids), ref["resid"], rtol=1e-7, atol=1e-9)
            assert_allclose(np.asarray(res.fitted_values), ref["fitted"], rtol=1e-7, atol=1e-9)
            print(f"rho={rho} case {i} {wt}/{ct} it={il} OK  iters={res.iterations} "
                  f"(ref {ref['iterations']}) j={res.j_stat.stat:.6f}")
    print("ALL REFERENCE COMPARISONS PASSED")

if __name__ == '__main__':
    main()
