System Regression Models
------------------------

System regression estimates multiple regressions simultaneously.  There are
three reasons to consider system estimation instead of equation by
equation estimation

* Joint inference on parameters across models
* Efficiency gains in some circumstances using cross-model GLS
* Imposing restrictions on parameters across models

The main model is the Seemingly Unrelated Regression (:class:`~linearmodels.system.model.SUR`)
Estimator.  This estimator uses a modified syntax since the class allow
multiple models to be specified, each with it own dependent and exogenous
variables. The more structured syntax uses a :class:`dict` or preferably
an :class:`~collections.OrderedDict`, which ensures that the order of the
equations in the results is preserved, where each entry is a complete model
with a dependent and exogenous (stored as a :class:`dict` with keys
``dependent`` and ``exog``.

.. code-block:: python

  from collections import OrderedDict
  import statsmodels.api as sm
  from linearmodels.datasets import fringe
  from linearmodels.system import SUR
  data = sm.add_constant(fringe.load())
  equations = OrderedDict()
  equations['earnings'] = {'dependent': data.hrearn,
                           'exog': data[['const', 'exper', 'tenure']]}
  equations['benefits'] = {'dependent': data.hrbens,
                           'exog': data[['const', 'exper', 'tenure']]}
  mod = SUR(equations)
  mod.fit(cov_type='unadjusted')


::

                           System GLS Estimation Summary
   ==============================================================================
   Estimator:                        GLS   Overall R-squared:              0.0757
   No. Equations.:                     2   Cov. Estimator:             unadjusted
   No. Observations:                 616   Num. Constraints:                 None
   Date:                Sat, Jun 17 2017
   Time:                        23:21:18


                   Equation: earnings, Dependent Variable: hrearn
   ==============================================================================
               Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
   ------------------------------------------------------------------------------
   const          4.2839     0.3407     12.573     0.0000      3.6161      4.9517
   exper          0.1163     0.0186     6.2478     0.0000      0.0798      0.1528
   tenure        -0.0283     0.0295    -0.9598     0.3372     -0.0862      0.0295

                   Equation: benefits, Dependent Variable: hrbens
   ==============================================================================
               Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
   ------------------------------------------------------------------------------
   const          0.6390     0.0449     14.220     0.0000      0.5509      0.7270
   exper          0.0014     0.0025     0.5617     0.5743     -0.0034      0.0062
   tenure         0.0316     0.0039     8.1254     0.0000      0.0240      0.0393

   ==============================================================================
   SystemResults, id: 0x282ca8f7b70



In addition to SUR, the system module also contain an estimator for the Three-stage Least
Squares (:class:`~linearmodels.system.model.IV3SLS`) Estimator.  3SLS is a generalization of
SUR which allows variables to be either exogenous or endogenous, and when there are endogenous
variables, for instruments to be used.  SUR is a special case of 3SLS where there are no
endogenous variables or instruments. 3SLS allows systems of IV equations to be jointly estimated.

Partially observed dependent data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default any row with a missing value in any equation is dropped before
estimation (``missing="drop"``).  :class:`~linearmodels.system.model.SUR`
also supports ``missing="observed"``, which keeps every row and fits the
joint Gaussian likelihood of the dependent cells that are actually
observed.  Rows remain independent, equations observed within a row retain
their cross-equation covariance :math:`\Sigma`, and a NaN dependent simply
contributes no density rather than being imputed.  Rows where every
dependent is missing stay in the data layout but never enter the
likelihood.

.. code-block:: python

  import numpy as np
  from linearmodels.system import SUR

  # equations is an OrderedDict of {"dependent": y, "exog": x} entries;
  # NaNs in a dependent mark that cell as unobserved.
  mod = SUR(equations, missing="observed")
  res = mod.fit(method="ml")

The observed-data path is maximum-likelihood only
(``fit(method="ml")``); the OLS and GLS estimators require complete
rectangular data.  If ``sigma`` is supplied it is treated as known and
must be finite, symmetric and positive definite, otherwise the
coefficients and a positive-definite :math:`\Sigma` are estimated jointly.
Two covariance estimators are available, both without small-sample
adjustments: ``cov_type="unadjusted"`` uses the inverse observed
information of the full likelihood (including the coefficient / covariance
nuisance blocks when :math:`\Sigma` is estimated), and
``cov_type="robust"`` uses a sandwich built from row-level scores that
combine every equation observed in that row.

Only NaNs in a dependent variable denote unobserved cells.  Missing or
infinite regressors and infinite dependent values raise an error even when
the corresponding dependent would otherwise be missing, and all equations
must share the same length and, for pandas inputs, the same unique index
in the same order.  Observation weights and linear parameter constraints
are not supported in this mode.  Results preserve the original row layout:
residuals are NaN in unobserved cells while fitted values (the conditional
mean :math:`x\beta`) are reported for every row and equation, and no
information from other equations is used to fill missing values.
Test statistics that require a complete rectangular residual matrix (for
example the Breusch-Pagan and diagonal-covariance likelihood-ratio tests)
are returned as invalid rather than silently replacing missing residuals
with zeros.

.. toctree::
   :maxdepth: 1
   :glob:

   examples/examples.ipynb
   examples/formulas.ipynb
   examples/three-stage-ls.ipynb
   reference
   mathematical-formula
