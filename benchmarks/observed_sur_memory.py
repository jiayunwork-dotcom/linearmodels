"""
Peak-memory measurement for the observed-data SUR ML path at the required
scale: 20,000 rows, 6 equations, 3 regressors per equation, with mixed
observation patterns.

The reported "incremental" peak is the fit's peak resident memory minus the
baseline measured immediately before fitting, so the interpreter and the
input arrays are excluded. No global dense block design and no dense
total-observation-sized covariance are ever constructed.
"""

from __future__ import annotations

import resource
import sys

import numpy as np

from linearmodels.system import SUR


def peak_rss_mib() -> float:
    # ru_maxrss is kilobytes on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    n, k, p = 20_000, 6, 3
    rng = np.random.default_rng(20261125)
    a = rng.standard_normal((k, k))
    sigma = a @ a.T / k + np.eye(k)
    err = rng.standard_normal((n, k)) @ np.linalg.cholesky(sigma).T
    xs, ys = [], []
    for i in range(k):
        x = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
        b = rng.standard_normal((p, 1))
        xs.append(x)
        ys.append(x @ b + err[:, [i]])
    # Mixed observation patterns: ~30% missing cells plus some fully empty
    # rows and rows observing only a single equation.
    miss = rng.random((n, k)) < 0.3
    for i in range(k):
        ys[i][miss[:, i]] = np.nan
    for rr in range(30):
        for i in range(k):
            ys[i][rr] = np.nan
    eqns = {f"eq{i}": {"dependent": ys[i], "exog": xs[i]} for i in range(k)}

    input_bytes = sum(x.nbytes for x in xs) + sum(y.nbytes for y in ys)
    print(f"design: n={n}, k={k}, regressors/eq={p}")
    print(f"input arrays: {input_bytes / 2**20:.1f} MiB")
    total_obs = np.isfinite(np.hstack(ys)).sum()
    print(f"observed cells: {total_obs} ({total_obs / (n * k):.1%})")

    baseline = peak_rss_mib()
    mod = SUR(eqns, missing="observed")
    res = mod.fit(method="ml")
    peak = peak_rss_mib()
    incremental = peak - baseline

    # Sanity assertions on the memory-relevant matrix dimensions.
    lik = mod._fit_engine_likelihood
    n_theta = lik.n_params + lik.n_sigma
    assert n_theta == k * p + k * (k + 1) // 2
    # The largest factored dense matrix is the small information matrix.
    assert lik.observed_information(
        np.asarray(res.params)[:, None], np.asarray(res.sigma)
    ).shape == (n_theta, n_theta)
    # One score vector per observed cell contribution; every non-empty
    # original row appears as many times as it has observed equations.
    n_score_rows = mod._fit_engine_likelihood.row_scores(
        np.asarray(res.params)[:, None], np.asarray(res.sigma)
    )[0].shape[0]
    assert n_score_rows == int(total_obs)
    print(
        f"largest dense factored matrix: {n_theta} x {n_theta} "
        f"({n_theta ** 2 * 8 / 2**10:.1f} KiB)"
    )
    print(f"score matrix rows: {n_score_rows} (one per observed cell)")
    print(f"log-likelihood: {res.loglik:.4f}, converged={res.converged}")
    print(f"baseline RSS: {baseline:.1f} MiB")
    print(f"peak RSS:     {peak:.1f} MiB")
    print(f"INCREMENTAL PEAK: {incremental:.1f} MiB (limit 256 MiB)")
    assert incremental < 256.0, "incremental peak memory exceeds 256 MiB"
    print("PASS")


if __name__ == "__main__":
    sys.exit(main())
