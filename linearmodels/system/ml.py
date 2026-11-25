"""
Maximum-likelihood estimation for SUR models with partially observed
dependent variables (``missing="observed"``).

The engine works directly from the observed data without imputing missing
dependent values.  Rows are independent, within a row the equation errors
are jointly normal with a constant, equation-level covariance matrix
:math:`\\Sigma`, and a missing dependent variable simply contributes no
density term for that equation.

Rows are grouped by their observation pattern so that the Gaussian
quantities for every pattern are evaluated on dense sub-matrices.  The
implementation never forms a global dense block design or a dense
covariance of the size of the total number of observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.linalg import LinAlgError, cholesky, slogdet, solve
from scipy.optimize import minimize

import linearmodels.typing.data

__all__ = [
    "ObservedSURConvergenceError",
    "ObservedSURError",
    "ObservedSURResult",
    "fit_observed_sur",
]


class ObservedSURError(ValueError):
    """Error raised when an observed-data SUR model is not estimable."""


class ObservedSURConvergenceError(ObservedSURError):
    """Error raised when the ML iteration does not converge."""


@dataclass
class _Pattern:
    """Observation pattern: rows where exactly ``eqns`` are observed."""

    key: tuple[int, ...]
    rows: linearmodels.typing.data.IntArray
    eqns: linearmodels.typing.data.IntArray
    n: int = 0


@dataclass
class ObservedSURResult:
    """Container for a successful observed-data ML fit."""

    params: linearmodels.typing.data.Float64Array
    sigma: linearmodels.typing.data.Float64Array
    loglik: float
    iter_count: int
    converged: bool
    termination: str
    tol: float
    score_norm: float
    score_rms: float
    nobs_total: int
    nobs_per_eq: linearmodels.typing.data.IntArray
    patterns: list[_Pattern]
    residuals: linearmodels.typing.data.Float64Array
    # Per-row (beta, vech(sigma)) scores on rows with at least one obs.
    row_scores: linearmodels.typing.data.Float64Array
    score_rows: linearmodels.typing.data.IntArray
    hessian: linearmodels.typing.data.Float64Array
    known_sigma: bool
    extra: dict = field(default_factory=dict)


def _chol(
    sigma: linearmodels.typing.data.Float64Array,
) -> linearmodels.typing.data.Float64Array:
    try:
        return cholesky(sigma)
    except LinAlgError as exc:
        raise ObservedSURError(
            "Sigma is not positive definite (Cholesky factorization failed)."
        ) from exc


def check_known_sigma(
    sigma: linearmodels.typing.data.ArrayLike, k: int
) -> linearmodels.typing.data.Float64Array:
    """Validate a caller-supplied error covariance matrix."""
    arr = np.asarray(sigma, dtype=np.float64)
    if arr.shape != (k, k):
        raise ObservedSURError(
            "sigma must be a square matrix with dimensions equal to the "
            f"number of equations ({k}); got shape {arr.shape}."
        )
    if not np.isfinite(arr).all():
        raise ObservedSURError("sigma must contain only finite values.")
    if not np.allclose(arr, arr.T, rtol=1e-10, atol=1e-12):
        raise ObservedSURError("sigma must be symmetric.")
    # Symmetrize to kill tiny asymmetry before the definiteness check.
    arr = (arr + arr.T) / 2.0
    eig = np.linalg.eigvalsh(arr)
    if eig.min() <= 0.0:
        raise ObservedSURError(
            "sigma must be strictly positive definite; smallest eigenvalue "
            f"is {eig.min():.3e}."
        )
    _chol(arr)
    return arr


def _build_patterns(
    observed: linearmodels.typing.data.BoolArray,
) -> list[_Pattern]:
    """Group row indices by their equation observation pattern."""
    n = observed.shape[0]
    pattern_map: dict[tuple[int, ...], list[int]] = {}
    for i in range(n):
        key = tuple(int(j) for j in np.flatnonzero(observed[i]))
        pattern_map.setdefault(key, []).append(i)
    patterns: list[_Pattern] = []
    # Deterministic order, empty patterns (all-missing rows) are skipped.
    for key in sorted(pattern_map, key=lambda t: (len(t), t)):
        if not key:
            continue
        rows = np.asarray(pattern_map[key], dtype=np.int64)
        eqns = np.asarray(key, dtype=np.int64)
        patterns.append(_Pattern(key=key, rows=rows, eqns=eqns, n=rows.shape[0]))
    return patterns


def _vech_indices(k: int) -> linearmodels.typing.data.IntArray:
    """Flat upper-triangle positions in row-major (C order) flattening."""
    idx = np.empty(k * (k + 1) // 2, dtype=np.int64)
    loc = 0
    for i in range(k):
        for j in range(i, k):
            idx[loc] = i * k + j
            loc += 1
    return idx


def _vech_position(i: int, j: int, k: int) -> int:
    """Position of the (i, j) element in the canonical vech ordering."""
    if i > j:
        i, j = j, i
    return int(i * k - i * (i - 1) // 2 + (j - i))


def _vech_to_full(
    vech: linearmodels.typing.data.Float64Array, k: int
) -> linearmodels.typing.data.Float64Array:
    out = np.zeros((k, k))
    loc = 0
    for i in range(k):
        for j in range(i, k):
            out[i, j] = vech[loc]
            out[j, i] = vech[loc]
            loc += 1
    return out


def _full_to_vech(
    sigma: linearmodels.typing.data.Float64Array, k: int
) -> linearmodels.typing.data.Float64Array:
    return np.asarray(
        [sigma[i, j] for i in range(k) for j in range(i, k)], dtype=np.float64
    )


class _ObservedLikelihood:
    """
    Observed-data Gaussian likelihood for a SUR system.

    Parameters
    ----------
    x : list of ndarray
        Per-equation regressor arrays, all with the same row count ``n``.
    y : list of ndarray
        Per-equation dependent arrays (n by 1).
    observed : ndarray
        Boolean n by k array marking observed dependent cells.
    patterns : list[_Pattern]
        Precomputed observation patterns.
    """

    def __init__(
        self,
        x: list[linearmodels.typing.data.Float64Array],
        y: list[linearmodels.typing.data.Float64Array],
        observed: linearmodels.typing.data.BoolArray,
        patterns: list[_Pattern],
    ) -> None:
        self._x = x
        self._y = y
        self._observed = observed
        self._patterns = patterns
        self._k = len(x)
        self._n = x[0].shape[0]
        self._widths = [int(a.shape[1]) for a in x]
        self._p = int(sum(self._widths))
        self._offsets = np.cumsum([0, *self._widths])
        self._vech_idx = _vech_indices(self._k)
        self._n_sigma = self._vech_idx.shape[0]
        # n_i per equation
        self._nobs_per_eq = observed.sum(axis=0).astype(np.int64)
        self._nobs_total = int(observed.sum())

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    @property
    def k(self) -> int:
        return self._k

    @property
    def n_params(self) -> int:
        return self._p

    @property
    def n_sigma(self) -> int:
        return self._n_sigma

    @property
    def nobs_per_eq(self) -> linearmodels.typing.data.IntArray:
        return self._nobs_per_eq

    @property
    def nobs_total(self) -> int:
        return self._nobs_total

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _split_beta(
        self, beta: linearmodels.typing.data.Float64Array
    ) -> list[linearmodels.typing.data.Float64Array]:
        return [beta[self._offsets[i] : self._offsets[i + 1]] for i in range(self._k)]

    def _residuals(
        self, beta: linearmodels.typing.data.Float64Array
    ) -> linearmodels.typing.data.Float64Array:
        betas = self._split_beta(beta)
        eps = np.full((self._n, self._k), np.nan)
        for i in range(self._k):
            eps[:, i] = (self._y[i] - self._x[i] @ betas[i]).squeeze(1)
        eps[~self._observed] = np.nan
        return eps

    def _pattern_data(
        self,
        pattern: _Pattern,
        eps: linearmodels.typing.data.Float64Array,
        sigma: linearmodels.typing.data.Float64Array | None = None,
    ) -> tuple[
        linearmodels.typing.data.Float64Array,
        linearmodels.typing.data.Float64Array,
        linearmodels.typing.data.Float64Array,
        float,
    ]:
        """Return (e, Sigma_m^-1, Sigma_m^-1 e, logdet) for one pattern."""
        rows, eqns = pattern.rows, pattern.eqns
        e = eps[np.ix_(rows, eqns)]
        assert sigma is not None
        sig_m = sigma[np.ix_(eqns, eqns)]
        sign, logdet = slogdet(sig_m)
        if sign <= 0.0 or not np.isfinite(logdet):
            raise LinAlgError("Non positive-definite pattern covariance")
        # The marginal precision of the observed sub-vector is the inverse of
        # the covariance SUB-matrix, not the sub-matrix of the full inverse.
        inv_m = np.linalg.inv(sig_m)
        se = inv_m @ e.T
        return e, inv_m, se, logdet

    # ------------------------------------------------------------------
    # Likelihood and full gradients (beta and vech sigma)
    # ------------------------------------------------------------------
    def loglik_grad(
        self,
        beta: linearmodels.typing.data.Float64Array,
        sigma: linearmodels.typing.data.Float64Array,
    ) -> tuple[
        float,
        linearmodels.typing.data.Float64Array,
        linearmodels.typing.data.Float64Array,
    ]:
        """
        Observed log-likelihood and its analytic gradient.

        Returns
        -------
        loglik : float
            Joint log-likelihood of all observed cells.
        grad_beta : ndarray
            Gradient with respect to the stacked coefficient vector.
        grad_vech : ndarray
            Gradient with respect to the free elements of sigma
            (column-wise upper triangle, matching :func:`_vech_indices`).
        """
        eps = self._residuals(beta)
        loglik = 0.0
        grad_beta = np.zeros(self._p)
        # Gradient with respect to each *free* (vech) element of sigma.
        grad_vech = np.zeros(self._n_sigma)
        for pattern in self._patterns:
            e, inv_m, se, logdet = self._pattern_data(pattern, eps, sigma=sigma)
            m = eqns = pattern.eqns
            n_m = pattern.n
            loglik += -0.5 * (
                n_m * m.shape[0] * np.log(2.0 * np.pi)
                + n_m * logdet
                + float(np.sum(e * se.T))
            )
            # beta gradient: x_i' (Sigma_m^-1 E)_i over the pattern rows;
            # ``se`` already contains Sigma_m^-1 e for every row.
            se_t = se.T  # n_m by m
            for pos, i in enumerate(eqns):
                loc = self._offsets[i]
                grad_beta[loc : loc + self._widths[i]] += (
                    self._x[i][pattern.rows].T @ se_t[:, pos]
                )
            # d ell / d Sigma_m treating Sigma_m as symmetric.
            et_inv_e = inv_m @ (e.T @ e) @ inv_m  # m by m
            g_m = -0.5 * (n_m * inv_m - et_inv_e)
            for a, ia in enumerate(eqns):
                for b in range(a, m.shape[0]):
                    ib = eqns[b]
                    # Free elements are the upper triangle; off-diagonal
                    # derivatives of a symmetric matrix pair collect the
                    # (a, b) and (b, a) differentials.
                    value = g_m[a, b] if a == b else g_m[a, b] + g_m[b, a]
                    grad_vech[_vech_position(int(ia), int(ib), self._k)] += value

        return loglik, grad_beta, grad_vech

    def loglik(
        self,
        beta: linearmodels.typing.data.Float64Array,
        sigma: linearmodels.typing.data.Float64Array,
    ) -> float:
        return self.loglik_grad(beta, sigma)[0]

    # ------------------------------------------------------------------
    # Closed form beta update for fixed sigma (pattern-wise GLS)
    # ------------------------------------------------------------------
    def gls_beta(
        self, sigma: linearmodels.typing.data.Float64Array
    ) -> linearmodels.typing.data.Float64Array:
        """GLS coefficient estimate conditional on a known sigma."""
        info = np.zeros((self._p, self._p))
        score = np.zeros((self._p, 1))
        for pattern in self._patterns:
            rows, eqns = pattern.rows, pattern.eqns
            sig_m = sigma[np.ix_(eqns, eqns)]
            inv_m = np.linalg.inv(sig_m)
            for pos_i, i in enumerate(eqns):
                xi = self._x[i][rows]
                loc_i = self._offsets[i]
                ki = self._widths[i]
                for pos_j, j in enumerate(eqns):
                    xj = self._x[j][rows]
                    w = inv_m[pos_i, pos_j]
                    info[
                        loc_i : loc_i + ki, self._offsets[j] : self._offsets[j + 1]
                    ] += w * (xi.T @ xj)
                score[loc_i : loc_i + ki] += (
                    xi.T
                    @ (
                        inv_m[pos_i, :][None, :]
                        @ np.hstack([self._y[jj][rows] for jj in eqns]).T
                    ).T
                )
        try:
            beta = solve(info, score)
        except LinAlgError as exc:
            raise ObservedSURError(
                "The observed design is numerically singular; the pattern-wise "
                "GLS information matrix could not be inverted."
            ) from exc
        return beta

    # ------------------------------------------------------------------
    # Per-row scores for sandwich covariance
    # ------------------------------------------------------------------
    def row_scores(
        self,
        beta: linearmodels.typing.data.Float64Array,
        sigma: linearmodels.typing.data.Float64Array,
    ) -> tuple[
        linearmodels.typing.data.Float64Array, linearmodels.typing.data.IntArray
    ]:
        """
        Per-row score vectors stacking the beta and vech(sigma) scores.

        Rows with no observed dependent variable contribute nothing and are
        omitted.  Each remaining original row is one independent unit.
        """
        eps = self._residuals(beta)
        total = self._p + self._n_sigma
        out_rows = np.empty(self._nobs_total, dtype=np.int64)
        out = np.zeros((self._nobs_total, total))
        loc = 0
        for pattern in self._patterns:
            rows, eqns = pattern.rows, pattern.eqns
            e = eps[np.ix_(rows, eqns)]  # n_m by m
            # Marginal precision of the observed sub-vector (inverse of the
            # covariance sub-matrix), matching the likelihood definition.
            inv_m = np.linalg.inv(sigma[np.ix_(eqns, eqns)])
            se = (inv_m @ e.T).T  # n_m by m
            n_m = pattern.n
            for r in range(n_m):
                row_idx = rows[r]
                s = np.zeros(total)
                for pos, i in enumerate(eqns):
                    lo = self._offsets[i]
                    s[lo : lo + self._widths[i]] = self._x[i][row_idx] * se[r, pos]
                # vech sigma score, single row
                er = e[r : r + 1].T  # m by 1
                g_m = -0.5 * (inv_m - inv_m @ er @ er.T @ inv_m)
                for a, ia in enumerate(eqns):
                    for b in range(a, eqns.shape[0]):
                        ib = eqns[b]
                        val = g_m[a, b] if a == b else g_m[a, b] + g_m[b, a]
                        s[self._p + _vech_position(int(ia), int(ib), self._k)] = val
                out[loc] = s
                out_rows[loc] = row_idx
                loc += 1
        order = np.argsort(out_rows)
        return out[order], out_rows[order]

    # ------------------------------------------------------------------
    # Observed information (negative Hessian of the log-likelihood)
    # ------------------------------------------------------------------
    def _pattern_sigma_derivatives(
        self,
        v_m: linearmodels.typing.data.Float64Array,
        e: linearmodels.typing.data.Float64Array,
        n_m: int,
        m: int,
    ) -> tuple[
        linearmodels.typing.data.Float64Array,
        linearmodels.typing.data.Float64Array,
    ]:
        r"""
        Gradient and observed information of a pattern log-likelihood with
        respect to the free (vech) elements of its marginal covariance.

        Let :math:`V=\Sigma_m^{-1}`, :math:`Q=E'E` and :math:`E_l` be the
        symmetric selector of free element :math:`l`. Then
        :math:`\dot V_l=-V E_l V` and

        .. math::

            g_l=-\tfrac12\,\operatorname{tr}\!\left(
                E_l(n_m V - V Q V)\right),

        .. math::

            H_{lj}=\tfrac12\,\operatorname{tr}\!\left(
                E_l(n_m \dot V_j - \dot V_j Q V - V Q \dot V_j)\right),

        with :math:`H=-\partial^2\ell` the observed information.
        """
        nfree = m * (m + 1) // 2
        pairs = [(a, b) for a in range(m) for b in range(a, m)]
        selectors = np.zeros((nfree, m, m))
        for loc, (a, b) in enumerate(pairs):
            selectors[loc, a, b] = 1.0
            selectors[loc, b, a] = 1.0
        q = e.T @ e
        vqv = v_m @ q @ v_m
        vdot = [-v_m @ selectors[li] @ v_m for li in range(nfree)]
        g = np.empty(nfree)
        info = np.empty((nfree, nfree))
        for li in range(nfree):
            g[li] = -0.5 * np.trace(selectors[li] @ (n_m * v_m - vqv))
            for j in range(nfree):
                dv = vdot[j]
                d_vqv = dv @ q @ v_m + v_m @ q @ dv
                info[li, j] = 0.5 * np.trace(selectors[li] @ (n_m * dv - d_vqv))
        return g, info

    def observed_information(
        self,
        beta: linearmodels.typing.data.Float64Array,
        sigma: linearmodels.typing.data.Float64Array,
        *,
        include_sigma: bool = True,
    ) -> linearmodels.typing.data.Float64Array:
        r"""
        Analytic observed information (negative Hessian) of the
        observed-data log-likelihood assembled pattern by pattern.

        The beta-beta block is the pattern-wise GLS information. The sigma
        blocks use the exact symmetric-matrix derivatives of the normal
        log-likelihood, and the cross blocks follow
        :math:`\partial V_m/\partial S = -V_m E V_m`.
        """
        p, k = self._p, self._k
        n_vech = self._n_sigma
        size = p + (n_vech if include_sigma else 0)
        H = np.zeros((size, size))
        eps = self._residuals(beta)
        for pattern in self._patterns:
            rows, eqns = pattern.rows, pattern.eqns
            e = eps[np.ix_(rows, eqns)]
            sig_m = sigma[np.ix_(eqns, eqns)]
            v_m = np.linalg.inv(sig_m)
            m = eqns.shape[0]

            # beta-beta block
            for a, i in enumerate(eqns):
                xi = self._x[i][rows]
                lo_i = self._offsets[i]
                for b, j in enumerate(eqns):
                    H[
                        lo_i : lo_i + self._widths[i],
                        self._offsets[j] : self._offsets[j + 1],
                    ] += (
                        v_m[a, b] * xi.T @ self._x[j][rows]
                    )
            if not include_sigma:
                continue

            _, info_m = self._pattern_sigma_derivatives(v_m, e, pattern.n, m)
            # Map pattern-local free coords onto global vech coords.
            local_pairs = [(a, b) for a in range(m) for b in range(a, m)]
            selectors_local = np.zeros((len(local_pairs), m, m))
            for loc, (a, b) in enumerate(local_pairs):
                selectors_local[loc, a, b] = 1.0
                selectors_local[loc, b, a] = 1.0
            global_cols = [
                p + _vech_position(int(eqns[a]), int(eqns[b]), k)
                for a, b in local_pairs
            ]
            for li, ci in enumerate(global_cols):
                for lj, cj in enumerate(global_cols):
                    H[ci, cj] += info_m[li, lj]

            # beta-vech cross information block. The information entry is
            # -d(score_beta)/ds where score_beta = x'(V_m e); under the vech
            # parameterization every free element's symmetric differential
            # contributes twice (two symmetric slots for off-diagonal
            # elements, two equal one-sided terms for diagonal elements).
            # beta-vech cross information block, verified against central
            # finite differences of the analytic score. Diagonal free
            # elements contribute the one-sided T_aa; off-diagonal elements
            # contribute the symmetric sum T_ab + T_ba (two distinct terms).
            for li, (a, b) in enumerate(local_pairs):
                t_ab = v_m[:, [a]] @ v_m[[b], :]
                if a == b:
                    dv = t_ab
                else:
                    dv = t_ab + v_m[:, [b]] @ v_m[[a], :]
                ci = global_cols[li]
                for ii, i_eq in enumerate(eqns):
                    lo = self._offsets[i_eq]
                    col = self._x[i_eq][rows].T @ (dv[ii] @ e.T)
                    H[lo : lo + self._widths[i_eq], ci] += col.ravel()
        # Mirror the cross block explicitly (never average it: averaging
        # would halve the rectangular beta-sigma entries). The beta-beta and
        # sigma-sigma blocks are accumulated symmetrically by construction.
        if include_sigma:
            H[p:, :p] = H[:p, p:].T
        return H

    def numerical_hessian(
        self,
        beta: linearmodels.typing.data.Float64Array,
        sigma: linearmodels.typing.data.Float64Array,
        eps_fd: float = 1e-6,
        *,
        include_sigma: bool = True,
    ) -> linearmodels.typing.data.Float64Array:
        """
        Central finite-difference negative Hessian of the analytic gradient,
        retained as an independent check of :meth:`observed_information`.
        """
        if include_sigma:
            vech0 = sigma.flatten()[self._vech_idx]
            theta0 = np.r_[beta.squeeze(1), vech0]
        else:
            theta0 = beta.squeeze(1).copy()
        total = theta0.shape[0]

        def grad(
            theta: linearmodels.typing.data.Float64Array,
        ) -> linearmodels.typing.data.Float64Array:
            if include_sigma:
                b = theta[: self._p][:, None]
                s = _vech_to_full(theta[self._p :], self._k)
                _, gb, gv = self.loglik_grad(b, s)
                return np.r_[gb, gv]
            _, gb, _ = self.loglik_grad(theta[:, None], sigma)
            return gb

        hess = np.zeros((total, total))
        for j in range(total):
            h = eps_fd * max(1.0, abs(theta0[j]))
            tp = theta0.copy()
            tm = theta0.copy()
            tp[j] += h
            tm[j] -= h
            gp = grad(tp)
            gm = grad(tm)
            hess[:, j] = -(gp - gm) / (2.0 * h)
        # Guard against a failed component producing NaNs.
        if not np.isfinite(hess).all():
            raise ObservedSURError(
                "Numerical observed information contains non-finite values; "
                "the solution is numerically singular."
            )
        hess = (hess + hess.T) / 2.0
        return hess

    # ------------------------------------------------------------------
    # Unconstrained L-BFGS-B over (beta, vech sigma)
    # ------------------------------------------------------------------
    def _sigma_barrier(
        self, sigma: linearmodels.typing.data.Float64Array
    ) -> tuple[float, linearmodels.typing.data.Float64Array]:
        """Quadratic eigenvalue barrier used outside the PD cone."""
        eig, vecs = np.linalg.eigh((sigma + sigma.T) / 2.0)
        gap = max(0.0, 1e-6 - eig.min())
        pen = 1e8 * gap**2
        v = vecs[:, [0]]
        d_sigma = 2e8 * gap * (v @ v.T)
        return pen, _full_to_vech(d_sigma, self._k)

    def _objective_vech(
        self, theta: linearmodels.typing.data.Float64Array
    ) -> tuple[float, linearmodels.typing.data.Float64Array]:
        beta = theta[: self._p][:, None]
        sigma = _vech_to_full(theta[self._p :], self._k)
        try:
            loglik, gb, gv = self.loglik_grad(beta, sigma)
        except LinAlgError:
            pen, d_vech = self._sigma_barrier(sigma)
            return pen, np.r_[np.zeros(self._p), d_vech]
        if not np.isfinite(loglik):
            return 1e12, np.zeros_like(theta)
        return -loglik, -np.r_[gb, gv]

    def newton_polish(
        self,
        beta0: linearmodels.typing.data.Float64Array,
        sigma0: linearmodels.typing.data.Float64Array,
        *,
        max_steps: int = 25,
        tol: float = 1e-10,
    ) -> dict:
        """
        Damped Newton refinement using the exact analytic observed
        information. A backtracking line search keeps every iterate inside
        the positive-definite cone and strictly increases the likelihood.
        """
        beta, sigma = beta0, (sigma0 + sigma0.T) / 2.0
        ll, gb, gv = self.loglik_grad(beta, sigma)
        for _ in range(max_steps):
            score = np.r_[gb, gv]
            if np.max(np.abs(score)) <= tol:
                break
            info = self.observed_information(beta, sigma)
            step = np.linalg.solve(info, score)
            theta = np.r_[beta.ravel(), sigma.flatten()[self._vech_idx]]
            accepted = False
            for shrink in (1.0, 0.5, 0.25, 0.1, 0.01):
                theta_new = theta + shrink * step
                b_new = theta_new[: self._p][:, None]
                s_new = _vech_to_full(theta_new[self._p :], self._k)
                eig = np.linalg.eigvalsh(s_new)
                if eig.min() <= 0.0 or not np.isfinite(theta_new).all():
                    continue
                try:
                    ll_new, gb_n, gv_n = self.loglik_grad(b_new, s_new)
                except LinAlgError:
                    continue
                if np.isfinite(ll_new) and ll_new >= ll - 1e-12 * abs(ll):
                    beta, sigma, ll, gb, gv = (
                        b_new,
                        s_new,
                        ll_new,
                        gb_n,
                        gv_n,
                    )
                    accepted = True
                    break
            if not accepted:
                break
        score = np.r_[gb, gv]
        return {
            "beta": beta,
            "sigma": sigma,
            "loglik": ll,
            "score_norm": float(np.max(np.abs(score))),
            "score_rms": float(np.sqrt(score @ score / score.shape[0])),
            "iterations": 0,
            "opt": None,
        }

    def minimize_vech(
        self,
        beta0: linearmodels.typing.data.Float64Array,
        sigma0: linearmodels.typing.data.Float64Array,
        *,
        iter_limit: int,
        tol: float,
    ) -> dict:
        """Joint L-BFGS-B optimization followed by a Newton polish."""
        theta0 = np.r_[beta0.squeeze(1), sigma0.flatten()[self._vech_idx]]
        opt = minimize(
            self._objective_vech,
            theta0,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": iter_limit,
                "maxfun": max(20, 6 * iter_limit),
                "ftol": 1e-14,
                "gtol": tol * 1e-2,
                "maxls": 40,
            },
        )
        beta_lb = opt.x[: self._p][:, None]
        sigma_lb = _vech_to_full(opt.x[self._p :], self._k)
        # The Newton polish is a convergence refinement, not extra budget:
        # it only runs when L-BFGS-B stopped before the iteration limit.
        if int(opt.nit) < iter_limit:
            polished = self.newton_polish(beta_lb, sigma_lb, tol=max(tol, 1e-11))
        else:
            ll, gb, gv = self.loglik_grad(beta_lb, sigma_lb)
            score = np.r_[gb, gv]
            polished = {
                "beta": beta_lb,
                "sigma": sigma_lb,
                "loglik": ll,
                "score_norm": float(np.max(np.abs(score))),
                "score_rms": float(np.sqrt(score @ score / score.shape[0])),
                "iterations": 0,
                "opt": opt,
            }
        polished["opt"] = opt
        polished["iterations"] = int(opt.nit)
        return polished

    def zigzag(
        self,
        beta0: linearmodels.typing.data.Float64Array,
        sigma0: linearmodels.typing.data.Float64Array,
        *,
        max_sweeps: int,
        tol: float,
    ) -> tuple[
        linearmodels.typing.data.Float64Array,
        linearmodels.typing.data.Float64Array,
        int,
    ]:
        """
        Stable alternating warm start.

        Each sweep (1) updates beta by pattern-wise GLS conditional on sigma
        and (2) optimizes the observed likelihood over vech(sigma) with beta
        fixed. Both steps stay inside the positive-definite cone via the
        barrier in :meth:`_objective_vech`.
        """
        beta, sigma = beta0, (sigma0 + sigma0.T) / 2.0
        sweeps = 0
        last_ll = -np.inf
        for sweep in range(1, max_sweeps + 1):
            beta = self.gls_beta(sigma)
            v0 = sigma.flatten()[self._vech_idx]
            opt = minimize(
                lambda v, b=beta: self._fixed_beta_sigma_objective(b, v),
                v0,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": 200, "gtol": tol * 1e-2, "ftol": tol * 1e-4},
            )
            sigma = _vech_to_full(opt.x, self._k)
            sigma = (sigma + sigma.T) / 2.0
            ll = self.loglik(beta, sigma)
            if abs(ll - last_ll) <= tol * max(1.0, abs(ll)):
                sweeps = sweep
                break
            last_ll = ll
            sweeps = sweep
        return beta, sigma, sweeps

    def _fixed_beta_sigma_objective(
        self,
        beta_fixed: linearmodels.typing.data.Float64Array,
        v: linearmodels.typing.data.Float64Array,
    ) -> tuple[float, linearmodels.typing.data.Float64Array]:
        """Negative log-likelihood over vech(sigma) with beta held fixed."""
        s = _vech_to_full(v, self._k)
        try:
            ll, _, gv = self.loglik_grad(beta_fixed, s)
        except LinAlgError:
            return self._sigma_barrier(s)
        return -ll, -gv


def _initial_sigma(
    lik: _ObservedLikelihood,
    beta: linearmodels.typing.data.Float64Array,
) -> linearmodels.typing.data.Float64Array:
    """Positive-(semi)definite sigma start from observed residual moments."""
    eps = lik._residuals(beta)
    k = lik.k
    sigma = np.eye(k)
    for i in range(k):
        ei = eps[:, i]
        m_i = np.isfinite(ei)
        vi = float(ei[m_i] @ ei[m_i] / max(m_i.sum(), 1))
        sigma[i, i] = max(vi, 1e-8)
    for i in range(k):
        for j in range(i + 1, k):
            m = np.isfinite(eps[:, i]) & np.isfinite(eps[:, j])
            if m.any():
                c = float(eps[m, i] @ eps[m, j] / m.sum())
                sigma[i, j] = c
                sigma[j, i] = c
    # Guarantee strict positive definiteness for the Cholesky start.
    eig = np.linalg.eigvalsh((sigma + sigma.T) / 2.0)
    gap = max(0.0, 1e-8 - eig.min()) + 1e-10
    return sigma + gap * np.eye(k)


def _ols_start(
    lik: _ObservedLikelihood,
) -> linearmodels.typing.data.Float64Array:
    beta = np.zeros((lik.n_params, 1))
    for i in range(lik.k):
        rows = np.flatnonzero(lik._observed[:, i])
        xi = lik._x[i][rows]
        yi = lik._y[i][rows]
        beta[lik._offsets[i] : lik._offsets[i + 1]] = np.linalg.lstsq(
            xi, yi, rcond=None
        )[0]
    return beta


def fit_observed_sur(
    x: list[linearmodels.typing.data.Float64Array],
    y: list[linearmodels.typing.data.Float64Array],
    observed: linearmodels.typing.data.BoolArray,
    *,
    sigma: linearmodels.typing.data.Float64Array | None,
    iter_limit: int,
    tol: float,
) -> ObservedSURResult:
    """
    Fit the observed-data Gaussian SUR model.

    Parameters
    ----------
    x, y : list of ndarray
        Per-equation design and dependent arrays sharing ``n`` rows.
    observed : ndarray
        Boolean n by k mask of observed dependent cells.
    sigma : ndarray or None
        Known error covariance; when ``None`` it is estimated jointly with
        the coefficients by maximum likelihood.
    iter_limit : int
        Maximum number of L-BFGS-B iterations.
    tol : float
        Convergence tolerance on the scaled gradient norm and objective
        changes.
    """
    observed = np.asarray(observed, dtype=bool)
    k = observed.shape[1]
    if observed.sum() == 0:
        raise ObservedSURError(
            "No dependent observations are available; the model cannot be fit."
        )
    patterns = _build_patterns(observed)
    lik = _ObservedLikelihood(x, y, observed, patterns)
    nobs_per_eq = lik.nobs_per_eq

    # Per-equation identification on each equation's own observed design.
    widths = lik._widths
    for i in range(k):
        n_i = int(nobs_per_eq[i])
        if n_i == 0:
            raise ObservedSURError(
                f"Equation {i} has no observed dependent values and cannot "
                "be estimated."
            )
        if n_i <= widths[i]:
            raise ObservedSURError(
                f"Equation {i} has {n_i} observed rows but {widths[i]} "
                "regressors; it is not identified."
            )
        rows = np.flatnonzero(observed[:, i])
        rank = int(np.linalg.matrix_rank(x[i][rows]))
        if rank < widths[i]:
            raise ObservedSURError(
                f"Equation {i}'s observed design matrix is rank deficient "
                f"(rank {rank} for {widths[i]} regressors)."
            )

    known_sigma = sigma is not None
    if known_sigma:
        assert sigma is not None
        sigma_hat = check_known_sigma(sigma, k)
        beta_hat = lik.gls_beta(sigma_hat)
        loglik, gb, _ = lik.loglik_grad(beta_hat, sigma_hat)
        # With known sigma the beta score must be zero at the GLS solution.
        score_norm = float(np.max(np.abs(gb)))
        score_rms = float(np.sqrt(np.mean(gb**2)))
        termination = (
            f"closed-form GLS with known sigma; "
            f"max|score|={score_norm:.3e}, rms|score|={score_rms:.3e}"
        )
        iter_count = 0
        if not np.isfinite(loglik) or score_norm > max(tol, 1e-8):
            raise ObservedSURConvergenceError(
                "Known-sigma GLS solution did not satisfy the score tolerance "
                f"(max abs score {score_norm:.3e})."
            )
    else:
        # Joint identification of covariance terms: every pair must share
        # at least one row where both dependents are observed.
        overlap = observed.T.astype(np.int64) @ observed.astype(np.int64)
        for i in range(k):
            for j in range(i + 1, k):
                if overlap[i, j] == 0:
                    raise ObservedSURError(
                        f"Equations {i} and {j} are never jointly observed; "
                        "their error covariance is not identified. Provide a "
                        "known sigma or revise the observation pattern."
                    )

        beta_ols = _ols_start(lik)
        sig_mom = _initial_sigma(lik, beta_ols)
        shrink = 0.5 * sig_mom + 0.5 * np.diag(np.diag(sig_mom)) + 1e-8 * np.eye(k)
        starts: list[
            tuple[
                str,
                linearmodels.typing.data.Float64Array,
                linearmodels.typing.data.Float64Array,
            ]
        ] = [
            ("OLS residual moments", beta_ols, sig_mom),
            ("identity covariance", beta_ols, np.eye(k)),
            ("shrunken moments", beta_ols, shrink),
        ]

        attempts: list[dict] = []
        best: dict | None = None
        warm_sweeps = max(2, min(25, iter_limit // 4))
        for name, b0, s0 in starts:
            try:
                # Stable alternating GLS/sigma sweeps move into the basin of
                # the interior optimum before the joint polishing step.
                bw, sw, _ = lik.zigzag(b0, s0, max_sweeps=warm_sweeps, tol=tol)
                res = lik.minimize_vech(bw, sw, iter_limit=iter_limit, tol=tol)
            except (ObservedSURConvergenceError, LinAlgError, ValueError):
                continue
            res["start"] = name
            attempts.append(res)
            if best is None or res["loglik"] > best["loglik"]:
                best = res
        if best is None:
            raise ObservedSURConvergenceError(
                "All likelihood optimizations failed numerically."
            )

        beta_hat = best["beta"]
        sigma_hat = best["sigma"]
        loglik = best["loglik"]
        score_norm = best["score_norm"]
        score_rms = best["score_rms"]
        iter_count = best["iterations"]
        opt = best["opt"]

        # Multi-start agreement: distinct interior optima with materially
        # different likelihoods indicate an ill-conditioned surface.
        ll_values = np.asarray([a["loglik"] for a in attempts])
        if ll_values.shape[0] > 1 and (ll_values.max() - ll_values.min()) > 1e-6 * max(
            1.0, abs(ll_values.max())
        ):
            beta_spread = max(
                float(np.max(np.abs(a["beta"] - beta_hat))) for a in attempts
            )
            if beta_spread > 1e-5:
                raise ObservedSURConvergenceError(
                    "Likelihood optimizations from different starting points "
                    "did not agree; the model surface appears ill-conditioned."
                )

        eig = np.linalg.eigvalsh((sigma_hat + sigma_hat.T) / 2.0)
        if not np.isfinite(beta_hat).all() or not np.isfinite(sigma_hat).all():
            raise ObservedSURConvergenceError(
                "Non-finite parameter values at the reported solution."
            )
        if eig.min() <= 0.0:
            raise ObservedSURConvergenceError(
                "The estimated sigma is not positive definite; the likelihood "
                "approached the singular boundary."
            )

        if iter_count >= iter_limit and score_norm > tol:
            raise ObservedSURConvergenceError(
                f"Iteration budget ({iter_limit}) exhausted before the "
                f"gradient tolerance was reached (max abs score "
                f"{score_norm:.3e} > tol {tol:.3e})."
            )
        # The score condition is the authoritative numerical stopping rule;
        # L-BFGS-B's own status may report precision limits even when the
        # analytic score meets the requested tolerance.
        score_threshold = max(tol, 1e-8 * (1.0 + np.abs(loglik)))
        if not bool(opt.success) and score_norm > score_threshold:
            raise ObservedSURConvergenceError(
                f"L-BFGS-B did not report convergence: {opt.message!r}. "
                f"Completed {iter_count} iterations and the max absolute "
                f"score is {score_norm:.3e}."
            )
        if score_norm > score_threshold:
            raise ObservedSURConvergenceError(
                "The score at the solution "
                f"(max abs component {score_norm:.3e}) exceeds the stopping "
                f"threshold ({score_threshold:.3e}); the solution is not "
                "converged."
            )
        termination = (
            f"L-BFGS-B: {opt.message}; "
            f"max|score|={score_norm:.3e} (tol={tol:.3g}), "
            f"rms|score|={best['score_rms']:.3e}, iterations={iter_count}"
        )

    # Residuals in the original layout, NaN in unobserved cells.
    residuals = lik._residuals(beta_hat)
    row_scores, score_rows = lik.row_scores(beta_hat, sigma_hat)
    hessian = lik.observed_information(
        beta_hat, sigma_hat, include_sigma=not known_sigma
    )
    if known_sigma:
        # Sigma is fixed: row scores cover beta only.
        row_scores = row_scores[:, : lik.n_params].copy()

    # Numerical information must be positive definite at a true interior ML
    # solution (beta block at minimum when sigma is known).
    eig_h = np.linalg.eigvalsh(hessian)
    if not np.isfinite(eig_h).all() or eig_h.min() <= 0.0:
        raise ObservedSURConvergenceError(
            "The observed information matrix is not positive definite "
            f"(smallest eigenvalue {eig_h.min():.3e}); the solution is "
            "numerically singular."
        )

    return ObservedSURResult(
        params=beta_hat,
        sigma=sigma_hat,
        loglik=float(loglik),
        iter_count=int(iter_count),
        converged=True,
        termination=termination,
        tol=float(tol),
        score_norm=float(score_norm),
        score_rms=float(score_rms),
        nobs_total=lik.nobs_total,
        nobs_per_eq=nobs_per_eq,
        patterns=patterns,
        residuals=residuals,
        row_scores=row_scores,
        score_rows=score_rows,
        hessian=hessian,
        known_sigma=known_sigma,
        extra={"likelihood": lik},
    )
