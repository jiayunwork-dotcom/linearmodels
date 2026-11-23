# Verification notes — unit-invariant `IVSystemGMM`

Reproduce everything with:

```bash
PYTHONPATH=/workspace python3 verification/repro.py          # original failure + fix
PYTHONPATH=/workspace:/workspace/verification python3 verification/acceptance.py
PYTHONPATH=/workspace python3 -m pytest linearmodels/tests/system -q
PYTHONPATH=/workspace python3 -m pytest linearmodels/tests/iv \
    -q --ignore=linearmodels/tests/iv/test_results.py   # test_results needs optional xarray
```

The reference values come from `verification/reference_gmm.py`: a dense,
observation-stacked implementation written directly from the GMM formulas.
It never calls the new internal transformations.

## What was wrong

1. `_validate_data` checked `matrix_rank(x)` / `matrix_rank(z)` on the raw
   user arrays.  With columns spanning `1e-8 … 1e8` NumPy's default `rcond`
   classified the instrument matrix as rank deficient even though it is
   perfectly identified.
2. `_blocked_gmm` solved `(X'ZW⁻¹Z'X)⁻¹ …` in the user's coordinates, so a
   large condition number from units could destroy the solve.
3. Weight/covariance updates, the convergence norm and the J statistic were
   all computed in the user's coordinates and therefore unit-dependent.

## What changed

All of the fix lives in the real `IVSystemGMM` execution path
(`linearmodels/system/model.py`), with one optional flag added to
`LinearConstraint` (`linearmodels/system/_utility.py`).

* **Signed column scales.**  Each column of every equation's `X` and `Z`
  is divided by `_column_scales`, the column norm with its orientation
  pinned to the sign of the largest-magnitude entry.  The internal arrays
  are *exactly* unchanged by any diagonal unit change `x → x D`, `z → z D'`
  (including negative proportions), because the scale is homogeneous of
  degree one and the sign convention removes the only remaining ambiguity.
* **Rank checks on the internal arrays** (`_column_norms`).  Duplicate
  columns, collinear columns and under-identification still fail; only the
  units stop mattering.
* **Estimation, weights, covariance, J** all run on the internal arrays.
  `_blocked_gmm` also checks identification of the cross-product that is
  actually solved (including the constrained one) and raises a clear
  `ValueError` rather than silently using a pseudoinverse.
* **Mapping back.**  `_from_internal_params`, `_from_internal_cov`,
  `_from_internal_weight` return parameters, covariance/standard errors and
  `res.w` in the caller's original units.  `res.w` is a moment-condition
  covariance (not its inverse), as before.
* **Constraints.**  User constraints are re-expressed on the internal
  parameters (`r → r diag(1/scale)`), row-normalized, and validated once in
  the user's coordinates.  Non-zero right-hand sides and cross-equation
  relationships are preserved; equivalent rewrites after a structural-unit
  change give the same estimates and inference.
* **Convergence.**  `tol` still measures a covariance-normalized parameter
  change, now in the unit-free internal coordinates, so a change of units
  cannot make the loop converge earlier or later.

The covariances are still computed from the *compact* block structure.  A
hook on `numpy.zeros/empty` during a 2-equation, 500-observation fit shows
the largest array allocated has 500 rows (observation-level arrays); no
`k·nobs = 1000`-row block-diagonal matrix is ever created.  The largest
matrices are the `8×8` moment covariance and the `4×8` weighted cross product.
`max_rows_created = 8`.

## Numerical evidence (rtol = 1e-7, atol = 1e-9 unless stated)

* Original failure: with `z` columns multiplied by `1e-8, 1, 1e8`, params,
  covariance and J match the base fit (max diff ~1e-15).
* Package vs dense reference, all weight/covariance combinations
  (`robust`/`unadjusted`/`kernel` × `robust`/`unadjusted`/`kernel`,
  1-, 2- and multi-step, `debiased`, fixed `sigma`, fixed/optimal bandwidth):
  params, covariance, `w`, J, residuals and fitted values agree to ~1e-17.
  Example (serially correlated errors, `rho = 0.6`, robust 2-step):
  `params = [1.09599517, 1.92941925, -1.87951963, 0.38803594]` for both;
  `max|cov diff| = 1.2e-17`; `J = 5.2927923309` for both.
* Unit/order invariance across every inference configuration: parameters,
  covariance, J, residuals, fitted values unchanged to ~1e-16; J difference
  ~3.6e-15.
* Structural-unit mapping: scaling an endogenous regressor by `s` divides
  its slope and standard error by `s`; residuals and fitted values are
  unchanged.
* `initial_weight`: supplying the same moment covariance re-expressed in the
  new units gives an equivalent estimate; `res.w` is returned in the
  original instrument coordinates.
* Constraints: non-zero cross-equation constraints hold exactly and map
  correctly under a structural-unit rewrite (params and standard errors).
* Repeated `fit` is bit-stable and the user's input arrays are not modified.
* `from_formula` and direct construction agree on values, equation labels
  and output order; parameter names follow the caller's input
  (`Intercept` for formulas, the supplied column name for direct arrays).
* Pickling the results (used by existing tests) works.

## Rejection boundaries

* Duplicate or exactly collinear instrument columns → `ValueError`
  ("… instrument array is full rank").
* Rank-deficient structural regressors → `ValueError`.
* Fewer instruments than endogenous variables → `ValueError`.
* A system whose solved cross-product is rank deficient → `ValueError`
  ("… not identified").  No pseudoinverse is used to mask it.
* This is a numerical-rank test after removing units; a model that is
  genuinely ill-conditioned *beyond* unit differences may still report a
  rank error, which is the intended, locatable failure.

## Valid numerical range

Units may be scaled per column by any non-zero factor with magnitude between
roughly `1e-80` and `1e80`; the `1e-80`/`1e80` combination is exercised in
`acceptance.py`.  The guarantee is for models that are well-conditioned once
units are removed.  Other system estimators (`SUR`, `IV3SLS`) are untouched;
their full test suites and the IV suite (7034 passed, 2402 skipped) plus the
absorption/asset-pricing/shared tests (88 passed) still pass.
