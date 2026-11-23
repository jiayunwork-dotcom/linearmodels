import numpy as np
from linearmodels.system import IVSystemGMM

rng = np.random.default_rng(42)
z = rng.standard_normal((200, 3))
e = rng.standard_normal((200, 2))
x = z[:, 0] + 0.5 * z[:, 1] + e[:, 0]
y0 = 1 + 2 * x + e[:, 0]
y1 = -2 + 0.5 * x + e[:, 1]
const = np.ones((200, 1))

def fit(z):
    eqns = {
        "e0": {"dependent": y0, "exog": const, "endog": x[:, None], "instruments": z},
        "e1": {"dependent": y1, "exog": const, "endog": x[:, None], "instruments": z},
    }
    mod = IVSystemGMM(eqns, weight_type="robust")
    res = mod.fit(iter_limit=2, cov_type="robust")
    return res

res = fit(z)
print("base params", res.params.values, res.j_stat.stat)
zs = z * np.array([1e-8, 1, 1e8])
try:
    res2 = fit(zs)
    print("scaled params", res2.params.values)
except Exception as ex:
    print("FAILED:", type(ex).__name__, ex)
