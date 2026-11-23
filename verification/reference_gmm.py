"""
Independent dense reference implementation of the system GMM estimator.

This module is deliberately written from the econometric formulas using
dense, observation-stacked block-diagonal matrices.  It does **not** import or
call any of the new internal coordinate transformations added to
``linearmodels.system.model``.  It exists only to produce ground-truth values
for the verification harness.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import solve

from linearmodels.iv.covariance import KERNEL_LOOKUP, kernel_optimal_bandwidth
from linearmodels.shared.covariance import cov_kernel


def _stack(x, y, z):
    """Build dense block-diagonal X, Z and stacked Y (observations stacked)."""
    k = len(x)
    nobs = x[0].shape[0]
    nvar = [a.shape[1] for a in x]
    ninstr = [a.shape[1] for a in z]
    big_x = np.zeros((k * nobs, sum(nvar)))
    big_z = np.zeros((k * nobs, sum(ninstr)))
    big_y = np.zeros((k * nobs, 1))
    cx = cz = 0
    for i in range(k):
        r = slice(i * nobs, (i + 1) * nobs)
        big_x[r, cx : cx + nvar[i]] = x[i]
        big_z[r, cz : cz + ninstr[i]] = z[i]
        big_y[r] = np.asarray(y[i]).reshape(-1, 1)
        cx += nvar[i]
        cz += ninstr[i]
    return big_x, big_y, big_z


def _debias(nobs, x, z):
    nvar = np.concatenate([np.full(a.shape[1], x[i].shape[1]) for i, a in enumerate(z)])
    nvar = np.sqrt(nvar)[:, None]
    return nobs / (nobs - nvar @ nvar.T)


def _kernel_cov(moments, kernel, bandwidth):
    weights = KERNEL_LOOKUP[kernel](bandwidth, moments.shape[0] - 1)
    out = cov_kernel(moments, weights)
    return (out + out.T) / 2


def _signed_column_scale(a):
    """Signed column scale: norm with orientation fixed by largest entry."""
    a = np.asarray(a, dtype=float)
    norms = np.sqrt(np.sum(a * a, axis=0))
    signs = _column_signs(a)
    return norms * signs


def _column_signs(a):
    """Sign of the largest-magnitude entry of each column."""
    idx = np.argmax(np.abs(a), axis=0)
    signs = np.sign(a[idx, np.arange(a.shape[1])])
    signs[signs == 0] = 1.0
    return signs


def _orientation(z):
    """Orient instrument columns so the largest-magnitude entry is positive"""
    return [np.asarray(a) * _column_signs(a)[None, :] for a in z]


def _optimal_bandwidth(z, eps, kernel):
    # The library's plug-in bandwidth sums standardized moment series across
    # columns.  It is invariant to column scaling but not to the arbitrary
    # sign (orientation) of a column.  Fix the orientation from the
    # instrument columns themselves, so the bandwidth does not depend on the
    # caller's units or sign conventions.
    z_oriented = _orientation(z)
    moments = np.concatenate(
        [z_oriented[i] * eps[:, i : i + 1] for i in range(len(z))], 1
    )
    m = moments / moments.std(0)[None, :]
    m = m.sum(1)
    return kernel_optimal_bandwidth(m, kernel=kernel)


def _weight_matrix(kind, x, z, eps, sigma, center, debiased, kernel, bandwidth,
                   optimal_bw):
    nobs = x[0].shape[0]
    if kind == "unadjusted":
        big_z = _stack(z, [a[:, :1] for a in z], z)[2]
        kron = np.kron(sigma, np.eye(nobs))
        return big_z.T @ kron @ big_z / nobs
    ze = np.concatenate([z[i] * eps[:, i : i + 1] for i in range(len(z))], 1)
    if center:
        ze = ze - ze.mean(0)
    if kind == "robust":
        w = ze.T @ ze / nobs
    else:  # kernel
        if bandwidth is not None:
            bw = bandwidth
        elif not optimal_bw:
            bw = nobs - 2
        else:
            bw = _optimal_bandwidth(z, eps, kernel)
        w = _kernel_cov(ze, kernel, bw)
    if debiased:
        w = w * _debias(nobs, x, z)
    return w


def _param_cov(kind, x, z, eps, sigma, w, cov_type, debiased, kernel, bandwidth,
               constraints=None):
    nobs = x[0].shape[0]
    k = len(x)
    big_x, _, big_z = _stack(x, [a[:, :1] for a in x], z)
    xpz = big_x.T @ big_z / nobs
    wi = np.linalg.inv(w)
    a = xpz @ wi @ xpz.T
    if cov_type == "unadjusted":
        omega = big_z.T @ np.kron(sigma, np.eye(nobs)) @ big_z / nobs
    else:
        ze = np.concatenate([z[i] * eps[:, i : i + 1] for i in range(k)], 1)
        if cov_type == "robust":
            omega = ze.T @ ze / nobs
        else:
            bw = bandwidth if bandwidth is not None else _optimal_bandwidth(z, eps, kernel)
            omega = _kernel_cov(ze, kernel, bw)
    middle = xpz @ wi @ omega @ wi @ xpz.T
    ainv = np.linalg.inv(a)
    cov = ainv @ middle @ ainv / nobs
    if constraints is not None:
        r = np.asarray(constraints[0], dtype=float)
        # Projection onto the constraint null space (sandwich projection).
        ainv_rt = ainv @ r.T
        proj = np.eye(a.shape[0]) - ainv_rt @ np.linalg.inv(r @ ainv_rt) @ r
        cov = proj @ cov @ proj.T
    cov = (cov + cov.T) / 2
    if debiased:
        nvar = np.concatenate([np.full(x[i].shape[1], x[i].shape[1]) for i in range(k)])
        adj = np.sqrt(nobs / (nobs - nvar))[:, None]
        cov = cov * (adj @ adj.T)
    return cov


def fit_dense(
    y,
    x,
    z,
    *,
    weight_type="robust",
    cov_type="robust",
    iter_limit=2,
    tol=1e-6,
    initial_weight=None,
    sigma_fixed=None,
    constraints=None,  # (r, q) in user parameter units
    debiased=False,
    center=False,
    kernel="bartlett",
    bandwidth=None,
    optimal_bw=False,
    obs_weights=None,
    weight_debiased=False,
):
    """Return a dict with params, cov, sigma, w, j_stat, resid, iterations."""
    k = len(x)
    nobs = x[0].shape[0]
    nvar = [a.shape[1] for a in x]
    ninstr = [a.shape[1] for a in z]
    kx = sum(nvar)
    kz = sum(ninstr)

    if obs_weights is not None:
        wy, wx, wz = [], [], []
        y = [np.asarray(a).reshape(-1, 1) for a in y]
        for i in range(k):
            w = np.asarray(obs_weights[i], dtype=float).ravel()
            w = w / np.nanmean(w)
            s = np.sqrt(w)
            wy.append(y[i] * s[:, None])
            wx.append(x[i] * s[:, None])
            wz.append(z[i] * s[:, None])
    else:
        wy, wx, wz = list(y), list(x), list(z)

    big_x, big_y, big_z = _stack(wx, wy, wz)
    xpz = big_x.T @ big_z / nobs
    zpy = big_z.T @ big_y / nobs

    # Canonical, unit-free parameter coordinates used only for the convergence
    # test.  Each structural column is divided by a signed scale (its norm with
    # the orientation fixed by the largest-magnitude entry).  This is a
    # diagonal re-parameterization that is *exactly* invariant to a change of
    # units, so ``tol`` always measures the same thing.
    x_scale = np.concatenate([_signed_column_scale(a) for a in wx])
    x_scale = np.where(x_scale == 0, 1.0, x_scale)

    def estimate(w):
        a = xpz @ np.linalg.inv(w) @ xpz.T
        b = xpz @ np.linalg.inv(w) @ zpy
        if constraints is not None:
            r, q = constraints
            r = np.asarray(r, dtype=float)
            q = np.asarray(q, dtype=float)
            kkt = np.block(
                [[a, r.T], [r, np.zeros((r.shape[0], r.shape[0]))]]
            )
            rhs = np.concatenate([b, q[:, None]])
            sol = solve(kkt, rhs)
            return sol[:kx]
        return solve(a, b)

    if initial_weight is None:
        w = big_z.T @ big_z / nobs
    else:
        w = np.asarray(initial_weight, dtype=float)
    beta = estimate(w)
    eps = big_y - big_x @ beta
    eps_mat = eps.reshape(k, nobs).T

    sigma = sigma_fixed
    if sigma is None:
        em = eps_mat - eps_mat.mean(0)
        sigma = em.T @ em / nobs
        if weight_type == "unadjusted" and weight_debiased:
            nv = np.sqrt(np.asarray(nvar))[:, None]
            sigma = sigma * (nobs / (nobs - nv @ nv.T))

    iters = 1
    norm = 10 * tol + 1
    vinv = None
    beta_last = beta
    while iters < iter_limit and norm > tol:
        sigma = sigma_fixed
        if sigma is None:
            em = eps_mat - eps_mat.mean(0)
            sigma = em.T @ em / nobs
            if weight_type == "unadjusted" and weight_debiased:
                nv = np.sqrt(np.asarray(nvar))[:, None]
                sigma = sigma * (nobs / (nobs - nv @ nv.T))
        w = _weight_matrix(
            weight_type, wx, wz, eps_mat, sigma, center, weight_debiased,
            kernel, bandwidth, optimal_bw,
        )
        beta = estimate(w)
        delta = beta_last - beta
        if vinv is None:
            wi = np.linalg.inv(w)
            a0 = xpz @ wi @ xpz.T / nobs
            # Map to the canonical, unit-free parameter coordinates.
            a0 = a0 / (x_scale[:, None] * x_scale[None, :])
            vinv = np.linalg.inv(a0)
        delta_c = delta * x_scale[:, None]
        norm = float(np.squeeze(delta_c.T @ vinv @ delta_c))
        beta_last = beta
        eps = big_y - big_x @ beta
        eps_mat = eps.reshape(k, nobs).T
        iters += 1

    cov = _param_cov(
        weight_type, wx, wz, eps_mat, sigma, w, cov_type, debiased, kernel, bandwidth,
        constraints=constraints,
    )

    # J statistic using the moment covariance w
    g_bar = big_z.T @ eps / nobs
    j_stat = float(np.squeeze(nobs * g_bar.T @ np.linalg.inv(w) @ g_bar))

    fitted = big_x @ beta
    eps = big_y - fitted
    resid = eps
    # Report residuals / fitted values in the *original* (unweighted) units.
    if obs_weights is not None:
        unweighted_y = np.zeros_like(big_y)
        unweighted_x = np.zeros_like(big_x)
        cx = 0
        for i in range(k):
            r = slice(i * nobs, (i + 1) * nobs)
            unweighted_y[r] = np.asarray(y[i]).reshape(-1, 1)
            nv = x[i].shape[1]
            unweighted_x[r, cx : cx + nv] = x[i]
            cx += nv
        fitted = unweighted_x @ beta
        resid = unweighted_y - fitted
    return {
        "params": beta.ravel(),
        "cov": cov,
        "sigma": sigma,
        "w": w,
        "j_stat": j_stat,
        "resid": resid.reshape(k, nobs).T,
        "fitted": fitted.reshape(k, nobs).T,
        "iterations": iters - 1,
        "eps": eps_mat,
    }
