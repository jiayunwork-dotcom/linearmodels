Next release
------------

- Added an observed-data maximum-likelihood mode to
  :class:`~linearmodels.system.model.SUR` through the new
  ``missing="observed"`` option. The estimator retains every row and fits
  the joint Gaussian likelihood of the dependent cells that are actually
  observed instead of dropping rows with any missing dependent value.
  NaNs in a dependent variable mark that cell as unobserved; missing or
  infinite regressors, infinite dependent values, non-unique or
  mismatched pandas indexes, observation weights and linear parameter
  constraints raise explicit errors. The mode supports
  ``fit(method="ml")`` with ``cov_type`` of ``"unadjusted"`` (inverse
  observed information, including uncertainty from estimating the error
  covariance) and ``"robust"`` (row-score sandwich), both without
  small-sample adjustments. Results keep the original row layout, report
  per-equation and summed observation counts, expose the log-likelihood
  and verifiable convergence diagnostics, and mark rectangular-residual
  test statistics as unavailable instead of imputing missing values.
