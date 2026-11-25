"""
Parameter covariance estimators for SUR models fit on partially observed
dependent data (``missing="observed"``).

Two estimators are supported, both without small-sample (debiased)
adjustments:

* ``"unadjusted"`` -- the inverse observed information of the joint
  observed-data likelihood, evaluated at the maximum-likelihood solution.
  When :math:`\\Sigma` is estimated, the information includes the
  coefficient / covariance nuisance-parameter blocks so that uncertainty
  from estimating :math:`\\Sigma` is propagated to the coefficients.
* ``"robust"`` -- a sandwich estimator that treats each original row as one
  independent unit.  The row score combines every equation observed in that
  row, and again spans both coefficients and the free elements of
  :math:`\\Sigma` when it is estimated.
"""

from __future__ import annotations

from numpy.linalg import inv

from linearmodels.shared.utility import AttrDict
import linearmodels.typing.data

__all__ = ["ObservedHeteroskedasticCovariance", "ObservedHomoskedasticCovariance"]


class _ObservedCovarianceBase:
    """Shared scaffolding for observed-data ML covariance estimators."""

    _name = "Observed-data ML Covariance"

    def __init__(
        self,
        n_params: int,
        hessian: linearmodels.typing.data.Float64Array,
        row_scores: linearmodels.typing.data.Float64Array,
        *,
        known_sigma: bool,
        debiased: bool = False,
    ) -> None:
        if debiased:
            raise ValueError(
                "Debiased covariance estimation is not supported for "
                'missing="observed" SUR models.'
            )
        self._n_params = n_params
        self._hessian = hessian
        self._row_scores = row_scores
        self._known_sigma = known_sigma
        self._debiased = debiased
        self._cov_config = AttrDict(debiased=False)
        self._str_extra = AttrDict(
            Debiased=False, KnownSigma=known_sigma, ObservedData=True
        )
        self._cov: linearmodels.typing.data.Float64Array | None = None

    def __str__(self) -> str:
        extra = ", ".join(f"{k}: {v}" for k, v in self._str_extra.items())
        return f"{self._name} ({extra})"

    def __repr__(self) -> str:
        return self.__str__() + f", id: {hex(id(self))}"

    @property
    def cov_config(self) -> AttrDict:
        return self._cov_config

    @property
    def full_cov(self) -> linearmodels.typing.data.Float64Array:
        """Covariance of the full (beta, [vech sigma]) parameter vector."""
        raise NotImplementedError

    @property
    def cov(self) -> linearmodels.typing.data.Float64Array:
        """Coefficient covariance (beta block only)."""
        if self._cov is None:
            full = self.full_cov
            # When sigma is known the information covers beta alone and
            # already has the coefficient dimension.
            if full.shape[0] == self._n_params:
                self._cov = full
            else:
                self._cov = full[: self._n_params, : self._n_params]
        return self._cov


class ObservedHomoskedasticCovariance(_ObservedCovarianceBase):
    r"""
    Observed-information covariance for the observed-data SUR likelihood.

    The full parameter vector is :math:`\theta = (\beta, \mathrm{vech}\Sigma)`
    when :math:`\Sigma` is estimated, and :math:`\beta` alone when it is
    known.  The covariance is the inverse of the observed information
    matrix

    .. math::

        \mathcal{I}(\hat{\theta}) =
            -\mathbb{E}\!\left[\frac{\partial^2 \ell(\theta)}
            {\partial \theta \partial \theta^\prime}\mid \mathcal{Y}_{obs}\right]
            \approx
            -\frac{\partial^2 \ell(\hat{\theta})}
            {\partial \theta \partial \theta^\prime}.
    """

    _name = "Observed-data ML Homoskedastic (Unadjusted) Covariance"

    @property
    def full_cov(self) -> linearmodels.typing.data.Float64Array:
        return inv(self._hessian)


class ObservedHeteroskedasticCovariance(_ObservedCovarianceBase):
    r"""
    Row-cluster sandwich covariance for the observed-data SUR likelihood.

    Each original row is an independent unit.  Row scores span every
    equation observed in that row, and the meat matrix is

    .. math::

        \hat{J} = \sum_{i=1}^{N} s_i(\hat{\theta}) s_i(\hat{\theta})^\prime

    so that the full sandwich covariance is

    .. math::

        \mathcal{I}(\hat{\theta})^{-1}\hat{J}
            \mathcal{I}(\hat{\theta})^{-1}.
    """

    _name = "Observed-data ML Heteroskedastic (Robust) Covariance"

    @property
    def full_cov(self) -> linearmodels.typing.data.Float64Array:
        info_inv = inv(self._hessian)
        meat = self._row_scores.T @ self._row_scores
        full = info_inv @ meat @ info_inv
        return (full + full.T) / 2.0
