r"""
Absorption of high-dimensional categorical effects for IV estimation

This module contains the numerical engine used by ``IV2SLS`` when the
``absorb`` keyword is used.  It provides

* validation of the user-supplied ``absorb`` frame,
* exact computation of the rank of the effect space spanned by one or two
  categorical variables, and
* weighted residualization of a matrix with respect to the effect space
  without ever materializing a dense observation-by-category matrix or a
  dense projection.

The residualization is the weighted projection

.. math::

    M_D^w = I - D (D^\prime W D)^{-} D^\prime W, \qquad W = diag(w)

applied column-by-column.  With a single effect the projection is the exact
weighted group demeaning.  With two effects it is computed with
:func:`scipy.sparse.linalg.lsmr` on a sparse, preconditioned dummy operator
that has two non-zero entries per row, so memory grows linearly in the
number of observations and categories.
"""

from __future__ import annotations

from typing import Any

from numpy import (
    asarray,
    bincount,
    column_stack,
    empty_like,
    sqrt,
)
from pandas import CategoricalDtype, DataFrame
import scipy.sparse as sp
from scipy.sparse.linalg import lsmr

from linearmodels.panel.utility import dummy_matrix
import linearmodels.typing.data

__all__ = ["EffectProjector", "effect_space_rank", "validate_absorb_frame"]

# LSMR termination flags that indicate a converged solution.  Remaining
# values (3, 6, 7) indicate the iteration stopped because of the condition
# limit or the iteration budget and are treated as failures.
_LSMR_CONVERGED = (0, 1, 2, 4, 5)


def validate_absorb_frame(absorb: Any, nobs: int) -> DataFrame:
    """
    Validate the absorb input and return a defensive copy

    Parameters
    ----------
    absorb : DataFrame
        Candidate frame of effects.  Must contain one or two columns, all
        categorical.
    nobs : int
        Number of observations in the model, used to check alignment.

    Returns
    -------
    DataFrame
        A copy of the validated frame.
    """
    if not isinstance(absorb, DataFrame):
        raise TypeError(
            "absorb must be a pandas DataFrame containing one or two "
            "categorical columns. Interactions and continuous variables are "
            "not supported."
        )
    if absorb.shape[1] == 0 or absorb.shape[1] > 2:
        raise ValueError(
            "absorb must contain one or two categorical columns. "
            f"Got {absorb.shape[1]} columns."
        )
    if not absorb.columns.is_unique:
        raise ValueError("absorb must not contain duplicate column names.")
    bad = [
        str(col)
        for col in absorb
        if not isinstance(absorb[col].dtype, CategoricalDtype)
    ]
    if bad:
        raise TypeError(
            "All columns of absorb must have categorical dtype. The "
            "following columns do not: " + ", ".join(bad) + "."
        )
    if absorb.shape[0] != nobs:
        raise ValueError(
            "absorb must have the same number of rows as the other model "
            f"inputs ({absorb.shape[0]} != {nobs}). Rows of absorb are "
            "matched to the data by position."
        )
    return absorb.copy()


def effect_space_rank(codes: list[linearmodels.typing.data.IntArray]) -> int:
    """
    Exact rank of the space spanned by one or two categorical effects

    Parameters
    ----------
    codes : list of ndarray
        Category codes, one array per effect.  Codes must be consecutive
        integers starting at 0 with every value present in the data (i.e.,
        unused categories have been removed).

    Returns
    -------
    int
        The rank of the dummy-variable space spanned by the effects.

    Notes
    -----
    With two effects the rank is ``g1 + g2 - c`` where ``g1`` and ``g2``
    are the number of categories and ``c`` is the number of connected
    components of the bipartite graph linking observed category pairs.
    This correctly handles nested effects (``c = g1`` when the second
    effect is nested in the first) and effects that partition the sample
    into disconnected blocks (``c > 1``).
    """
    if len(codes) == 1:
        return int(codes[0].max()) + 1
    c1, c2 = codes[0], codes[1]
    g1 = int(c1.max()) + 1
    g2 = int(c2.max()) + 1
    # Union-find over the g1 + g2 category nodes
    parent = list(range(g1 + g2))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(c1.shape[0]):
        root1 = find(int(c1[i]))
        root2 = find(g1 + int(c2[i]))
        if root1 != root2:
            parent[root2] = root1
    components = len({find(v) for v in range(g1 + g2)})
    return g1 + g2 - components


class EffectProjector:
    r"""
    Weighted projection engine for one or two categorical effects

    Parameters
    ----------
    codes : list of ndarray
        Category codes, one integer array per effect.  Codes must be
        consecutive integers starting at 0 with all values observed.
    weights : ndarray
        Positive observation weights with shape (nobs,).  Weights are used
        with the same meaning as in :class:`linearmodels.iv.IV2SLS`.
    atol : float
        Absolute tolerance used by LSMR when two effects are absorbed.
    btol : float
        Relative tolerance used by LSMR when two effects are absorbed.
    maxiter : int
        Iteration budget for LSMR when two effects are absorbed.  If None,
        the LSMR default (the smaller dimension of the operator) is used.
        Exhausting the budget raises an error rather than silently
        returning a non-converged projection.

    Notes
    -----
    The projector removes the weighted best fit of the effects from each
    column, i.e. it applies :math:`M_D^w = I - D(D^\prime W D)^{-}D^\prime
    W`.  With a single effect this is the exact weighted group demeaning
    computed with :func:`numpy.bincount`.  With two effects the weighted
    normal equations are solved with LSMR on a sparse preconditioned dummy
    operator that has at most two non-zeros per row.  Neither a dense
    nobs-by-ncat matrix nor a dense projection is ever formed, and the
    category coefficients are never collected into a dense system that
    needs to be inverted.

    The sparse operator is constructed once and reused for every column
    passed to :meth:`residualize`, and the absorbed arrays produced by a
    model are stored on the model instance.  Results are never shared
    across instances, so changing the weights or the category table can
    never reuse a stale projection.
    """

    def __init__(
        self,
        codes: list[linearmodels.typing.data.IntArray],
        weights: linearmodels.typing.data.Float64Array,
        *,
        atol: float = 1e-12,
        btol: float = 1e-12,
        maxiter: int | None = None,
    ) -> None:
        self._codes = [asarray(c) for c in codes]
        self._weights = asarray(weights, dtype=float).squeeze()
        self._root_w = sqrt(self._weights)
        self._ncats = [int(c.max()) + 1 for c in self._codes]
        self._rank = effect_space_rank(self._codes)
        self._atol = float(atol)
        self._btol = float(btol)
        self._maxiter = maxiter
        self._operator: sp.csc_matrix | None = None

    @property
    def rank(self) -> int:
        """Rank of the effect space"""
        return self._rank

    @property
    def n_effects(self) -> int:
        """Number of absorbed effects"""
        return len(self._codes)

    @property
    def n_categories(self) -> list[int]:
        """Number of categories in each effect"""
        return list(self._ncats)

    def _weighted_operator(self) -> sp.csc_matrix:
        """Sparse weighted dummy operator with unit-norm columns"""
        if self._operator is None:
            code_mat = self._codes[0][:, None]
            if len(self._codes) == 2:
                code_mat = column_stack(self._codes)
            dummies = dummy_matrix(code_mat, precondition=True)[0]
            assert isinstance(dummies, sp.csc_matrix)
            weighted = sp.diags(self._root_w).dot(dummies).asformat("csc")
            assert isinstance(weighted, sp.csc_matrix)
            self._operator = weighted
        return self._operator

    def _demean_single(
        self, mat: linearmodels.typing.data.Float64Array
    ) -> linearmodels.typing.data.Float64Array:
        """Exact weighted group demeaning for a single effect"""
        codes = self._codes[0]
        ncat = self._ncats[0]
        w = self._weights
        denom = bincount(codes, weights=w, minlength=ncat)
        out = empty_like(mat, dtype=float)
        for j in range(mat.shape[1]):
            numer = bincount(codes, weights=w * mat[:, j], minlength=ncat)
            out[:, j] = mat[:, j] - (numer / denom)[codes]
        return out

    def _lsmr_residualize(
        self, mat: linearmodels.typing.data.Float64Array
    ) -> linearmodels.typing.data.Float64Array:
        """Residualize using LSMR on the sparse weighted dummy operator"""
        operator = self._weighted_operator()
        root_w = self._root_w
        out = empty_like(mat, dtype=float)
        for j in range(mat.shape[1]):
            weighted_col = root_w * mat[:, j]
            sol = lsmr(
                operator,
                weighted_col,
                atol=self._atol,
                btol=self._btol,
                maxiter=self._maxiter,
            )
            beta, istop = sol[0], sol[1]
            if istop not in _LSMR_CONVERGED:
                raise RuntimeError(
                    "Absorbing the effects did not converge: LSMR stopped "
                    f"with istop={istop} before reaching the requested "
                    f"accuracy (atol={self._atol}, btol={self._btol}) while "
                    f"absorbing column {j}. The projection was not computed "
                    "and results are not available. Consider whether the "
                    "effects are degenerate, or increase the iteration "
                    "budget."
                )
            resid = weighted_col - operator @ beta
            out[:, j] = resid / root_w
        return out

    def residualize(
        self, mat: linearmodels.typing.data.Float64Array
    ) -> linearmodels.typing.data.Float64Array:
        """
        Remove the weighted projection on the effect space

        Parameters
        ----------
        mat : ndarray
            Array with shape (nobs, nvar) to residualize.  Each column is
            residualized independently.

        Returns
        -------
        ndarray
            The residuals of each column after removing the weighted best
            fit of the absorbed effects, in the original units of ``mat``.
        """
        if self.n_effects == 1:
            return self._demean_single(mat)
        return self._lsmr_residualize(mat)
