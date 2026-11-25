"""
Estimators for systems of equations

References
----------
Greene, William H. "Econometric analysis 4th edition." International edition,
    New Jersey: Prentice Hall (2000).
StataCorp. 2013. Stata 13 Base Reference Manual. College Station, TX: Stata
    Press.
Henningsen, A., & Hamann, J. (2007). systemfit: A Package for Estimating
    Systems of Simultaneous Equations in R. Journal of Statistical Software,
    23(4), 1 - 40. doi:http://dx.doi.org/10.18637/jss.v023.i04
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import reduce
import textwrap
from typing import Any, Literal, Optional, cast
import warnings

from formulaic.utils.context import capture_context
import numpy as np
from numpy.linalg import inv, lstsq, matrix_rank, solve
from pandas import DataFrame, Index, Series, concat

from linearmodels.iv._utility import IVFormulaParser
from linearmodels.iv.data import IVData
from linearmodels.shared.exceptions import missing_warning
from linearmodels.shared.hypotheses import InvalidTestStatistic, WaldTestStatistic
from linearmodels.shared.linalg import has_constant
from linearmodels.shared.utility import AttrDict
from linearmodels.system._utility import (
    LinearConstraint,
    blocked_column_product,
    blocked_cross_prod,
    blocked_diag_product,
    blocked_inner_prod,
    inv_matrix_sqrt,
)
from linearmodels.system.covariance import (
    ClusteredCovariance,
    GMMHeteroskedasticCovariance,
    GMMHomoskedasticCovariance,
    GMMKernelCovariance,
    HeteroskedasticCovariance,
    HomoskedasticCovariance,
    KernelCovariance,
)
from linearmodels.system.ml import (
    ObservedSURError,
    check_known_sigma,
    fit_observed_sur,
)
from linearmodels.system.observed_covariance import (
    ObservedHeteroskedasticCovariance,
    ObservedHomoskedasticCovariance,
)
from linearmodels.system.gmm import (
    HeteroskedasticWeightMatrix,
    HomoskedasticWeightMatrix,
    KernelWeightMatrix,
)
from linearmodels.system.results import (
    GMMSystemResults,
    ObservedSURResults,
    SystemResults,
)
import linearmodels.typing.data

__all__ = ["IV3SLS", "SUR", "IVSystemGMM", "LinearConstraint"]

UNKNOWN_EQ_TYPE = """\
Contents of each equation must be either a dictionary with keys "dependent"
and "exog" or a 2-element tuple of he form (dependent, exog).
equations[{key}] was {type}
"""

COV_TYPES = {
    "unadjusted": "unadjusted",
    "homoskedastic": "unadjusted",
    "robust": "robust",
    "heteroskedastic": "robust",
    "kernel": "kernel",
    "hac": "kernel",
    "clustered": "clustered",
}

COV_EST = {
    "unadjusted": HomoskedasticCovariance,
    "robust": HeteroskedasticCovariance,
    "kernel": KernelCovariance,
    "clustered": ClusteredCovariance,
}

GMM_W_EST = {
    "unadjusted": HomoskedasticWeightMatrix,
    "robust": HeteroskedasticWeightMatrix,
    "kernel": KernelWeightMatrix,
}

GMM_COV_EST = {
    "unadjusted": GMMHomoskedasticCovariance,
    "robust": GMMHeteroskedasticCovariance,
    "kernel": GMMKernelCovariance,
}

OBSERVED_COV_EST = {
    "unadjusted": ObservedHomoskedasticCovariance,
    "robust": ObservedHeteroskedasticCovariance,
}

OBSERVED_MISSING_VALUES = ("drop", "observed")


def _missing_weights(
    weights: Mapping[str, linearmodels.typing.data.ArrayLike | None],
) -> None:
    """Raise warning if missing weighs found"""
    missing = [key for key in weights if weights[key] is None]
    if missing:
        msg = "Weights not found for equation labels:\n{}".format(", ".join(missing))
        warnings.warn(msg, UserWarning, stacklevel=2)


def _parameters_from_xprod(
    xpx: linearmodels.typing.data.Float64Array,
    xpy: linearmodels.typing.data.Float64Array,
    constraints: LinearConstraint | None = None,
) -> linearmodels.typing.data.Float64Array:
    r"""
    Estimate regression parameters from cross produces

    Parameters
    ----------
    xpx : ndarray
        Cross product measuring variation in x (nvar by nvar)
    xpy : ndarray
        Cross produce measuring covariation between x and y (nvar by 1)
    constraints : LinearConstraint
        Constraints to use in estimation

    Returns
    -------
    params : ndarray
        Estimated parameters (nvar by 1)

    Notes
    -----
    xpx and xpy can be any form similar to the two inputs into the usual
    parameter estimator for a linear regression. In particular, many
    estimators can be written as

    .. math::

        (x^\prime w x)^{-1}(x^\prime w y)

    for some weight matrix :math:`w`.
    """
    if constraints is not None:
        cons = constraints
        xpy = cons.t.T @ xpy - cons.t.T @ xpx @ cons.a.T
        xpx = cons.t.T @ xpx @ cons.t
        params_c = solve(xpx, xpy)
        params = cons.t @ params_c + cons.a.T
    else:
        params = solve(xpx, xpy).astype(float, copy=False)
    return params


class SystemFormulaParser:
    def __init__(  # noqa: PLR0917
        self,
        formula: Mapping[str, str] | str,
        data: DataFrame,
        weights: Mapping[str, linearmodels.typing.data.ArrayLike] | None = None,
        eval_env: int = 2,
        context: Mapping[str, Any] | None = None,
        allow_missing_dependent: bool = False,
    ) -> None:
        if not isinstance(formula, (Mapping, str)):
            raise TypeError("formula must be a string or dictionary-like")
        self._formula: Mapping[str, str] | str = formula
        self._data = data
        self._weights = weights
        self._parsers: dict[str, IVFormulaParser] = {}
        self._weight_dict: dict[str, Series | None] = {}
        self._eval_env = eval_env
        self._clean_formula: dict[str, str] = {}
        self._allow_missing_dependent = allow_missing_dependent
        if context is None:
            self._context = capture_context(eval_env)
        else:
            self._context = context
        self._parse()

    @staticmethod
    def _prevent_autoconst(formula: str) -> str:
        if not (" 0+" in formula or " 0 +" in formula):
            formula = "~ 0 +".join(formula.split("~"))
        return formula

    @staticmethod
    def _convert_to_series(
        value: linearmodels.typing.data.ArrayLike, name: str
    ) -> Series:
        shape = value.shape
        if len(shape) > 2 or (len(shape) == 2 and min(shape) != 1):
            raise ValueError(f"{name} must be squeezable to 1D.")
        if len(shape) == 1:
            value_series = Series(value)
        elif isinstance(value, DataFrame):
            value_series = value.iloc[:, 0]
        else:
            value_series = Series(np.squeeze(value))

        return value_series

    def _parse(self) -> None:
        formula = self._formula
        data = self._data
        weights = self._weights
        parsers = self._parsers
        weight_dict = self._weight_dict
        cln_formula = self._clean_formula

        if isinstance(formula, Mapping):
            for formula_key in formula:
                f = formula[formula_key]
                f = self._prevent_autoconst(f)
                parsers[formula_key] = IVFormulaParser(
                    f,
                    data,
                    context=self._context,
                    allow_missing_dependent=self._allow_missing_dependent,
                )

                if weights is not None:
                    if formula_key in weights:
                        value = weights[formula_key]
                        weight_dict[formula_key] = self._convert_to_series(
                            value, "weights"
                        )
                    else:
                        weight_dict[formula_key] = None
                cln_formula[formula_key] = f
        else:
            formula = formula.replace("\n", " ").strip()
            parts = formula.split("}")
            for _part in parts:
                key = base_key = None
                part = _part.strip()
                if part == "":
                    continue
                part = part.replace("{", "")
                if ":" in part.split("~")[0]:
                    base_key, part = part.split(":")
                    key = base_key = base_key.strip()
                    part = part.strip()
                f = self._prevent_autoconst(part)
                if base_key is None:
                    base_key = key = f.split("~")[0].strip()
                count = 0
                while key in parsers:
                    key = base_key + f".{count}"
                    count += 1
                assert isinstance(key, str)
                parsers[key] = IVFormulaParser(
                    f,
                    data,
                    context=self._context,
                    allow_missing_dependent=self._allow_missing_dependent,
                )
                cln_formula[key] = f
                if weights is not None:
                    if key in weights:
                        value = weights[key]
                        weight_dict[key] = self._convert_to_series(value, "weights")
                    else:
                        weight_dict[key] = None
        _missing_weights(weight_dict)
        self._weight_dict = weight_dict

    def _get_variable(self, variable: str) -> dict[str, DataFrame | None]:
        return {key: getattr(self._parsers[key], variable) for key in self._parsers}

    @property
    def formula(self) -> dict[str, str]:
        """Cleaned version of formula"""
        return self._clean_formula

    @property
    def eval_env(self) -> int:
        """Set or get the eval env depth"""
        return self._eval_env

    @eval_env.setter
    def eval_env(self, value: int) -> None:
        self._eval_env = value
        self._context = capture_context(self._eval_env)
        # Update parsers for new level
        parsers = self._parsers
        new_parsers = {}
        for key in parsers:
            parser = parsers[key]
            new_parsers[key] = IVFormulaParser(
                parser._formula, parser._data, context=self._context
            )
        self._parsers = new_parsers

    @property
    def equation_labels(self) -> list[str]:
        return list(self._parsers.keys())

    @property
    def data(self) -> dict[str, dict[str, linearmodels.typing.data.ArrayLike | None]]:
        out: dict[str, dict[str, linearmodels.typing.data.ArrayLike | None]] = {}
        dep = self.dependent
        for key in dep:
            out[key] = {"dependent": dep[key]}
        exog = self.exog
        for key in exog:
            out[key]["exog"] = exog[key]
        endog = self.endog
        for key in endog:
            out[key]["endog"] = endog[key]
        instr = self.instruments
        for key in instr:
            out[key]["instruments"] = instr[key]
        for key in self._weight_dict:
            if self._weight_dict[key] is not None:
                out[key]["weights"] = self._weight_dict[key]
        return out

    @property
    def dependent(self) -> dict[str, DataFrame | None]:
        return self._get_variable("dependent")

    @property
    def exog(self) -> dict[str, DataFrame | None]:
        return self._get_variable("exog")

    @property
    def endog(self) -> dict[str, DataFrame | None]:
        return self._get_variable("endog")

    @property
    def instruments(self) -> dict[str, DataFrame | None]:
        return self._get_variable("instruments")


class _SystemModelBase:
    r"""
    Base class for system estimators

    Parameters
    ----------
    equations : dict
        Dictionary-like structure containing dependent, exogenous, endogenous
        and instrumental variables.  Each key is an equations label and must
        be a string. Each value must be either a tuple of the form (dependent,
        exog, endog, instrument[, weights]) or a dictionary with keys "dependent",
        and at least one of "exog" or "endog" and "instruments".  When using a
        tuple, values must be provided for all 4 variables, although either
        empty arrays or `None` can be passed if a category of variable is not
        included in a model. The dictionary may contain optional keys for
        "exog", "endog", "instruments", and "weights". "exog" can be omitted
        if all variables in an equation are endogenous. Alternatively, "exog"
        can contain either an empty array or `None` to indicate that an
        equation contains no exogenous regressors. Similarly "endog" and
        "instruments" can either be omitted or may contain an empty array (or
        `None`) if all variables in an equation are exogenous.
    sigma : array_like
        Prespecified residual covariance to use in GLS estimation. If not
        provided, FGLS is implemented based on an estimate of sigma.
    """

    def __init__(
        self,
        equations: Mapping[
            str,
            Mapping[str, linearmodels.typing.data.ArrayLike | None]
            | Sequence[linearmodels.typing.data.ArrayLike | None],
        ],
        *,
        sigma: linearmodels.typing.data.ArrayLike | None = None,
        missing: Literal["drop", "observed"] = "drop",
    ) -> None:
        if not isinstance(equations, Mapping):
            raise TypeError("equations must be a dictionary-like")
        for key in equations:
            if not isinstance(key, str):
                raise ValueError("Equation labels (keys) must be strings")
        if missing not in OBSERVED_MISSING_VALUES:
            raise ValueError(
                f"missing must be either 'drop' or 'observed'; got {missing!r}."
            )
        self._missing = missing

        self._equations = equations
        self._sigma = None
        if sigma is not None:
            self._sigma = np.asarray(sigma)
            k = len(self._equations)
            if self._sigma.shape != (k, k):
                raise ValueError(
                    "sigma must be a square matrix with dimensions "
                    "equal to the number of equations"
                )
        self._param_names: list[str] = []
        self._eq_labels: list[str] = []
        self._dependent: list[IVData] = []
        self._exog: list[IVData] = []
        self._instr: list[IVData] = []
        self._endog: list[IVData] = []

        self._y: list[linearmodels.typing.data.Float64Array] = []
        self._x: list[linearmodels.typing.data.Float64Array] = []
        self._wy: list[linearmodels.typing.data.Float64Array] = []
        self._wx: list[linearmodels.typing.data.Float64Array] = []
        self._w: list[linearmodels.typing.data.Float64Array] = []
        self._z: list[linearmodels.typing.data.Float64Array] = []
        self._wz: list[linearmodels.typing.data.Float64Array] = []

        self._weights: list[IVData] = []
        self._formula: str | dict[str, str] | None = None
        self._constraints: LinearConstraint | None = None
        self._constant_loc: list[int] = []
        self._has_constant: Series = Series(dtype=bool)
        self._common_exog = False
        self._original_index: Index | None = None
        self._observed_mask: linearmodels.typing.data.BoolArray | None = None
        self._empty_rows: linearmodels.typing.data.IntArray = np.empty(
            0, dtype=np.int64
        )
        self._model_name = "Three Stage Least Squares (3SLS)"

        self._validate_data()

    @property
    def formula(self) -> str | dict[str, str] | None:
        """Set or get the formula used to construct the model"""
        return self._formula

    @formula.setter
    def formula(self, value: str | dict[str, str] | None) -> None:
        self._formula = value

    def _validate_data(self) -> None:
        if self._missing == "observed":
            self._reject_observed_weights()
        ids = []
        for i, key in enumerate(self._equations):
            self._eq_labels.append(key)
            eq_data = self._equations[key]
            dep_name = "dependent_" + str(i)
            exog_name = "exog_" + str(i)
            endog_name = "endog_" + str(i)
            instr_name = "instr_" + str(i)
            if isinstance(eq_data, (tuple, list)):
                dep = IVData(eq_data[0], var_name=dep_name)
                self._dependent.append(dep)
                current_id: tuple[int, ...] = (id(eq_data[1]),)
                self._exog.append(
                    IVData(eq_data[1], var_name=exog_name, nobs=dep.shape[0])
                )
                endog = IVData(eq_data[2], var_name=endog_name, nobs=dep.shape[0])
                if endog.shape[1] > 0:
                    current_id += (id(eq_data[2]),)
                ids.append(current_id)
                self._endog.append(endog)

                self._instr.append(
                    IVData(eq_data[3], var_name=instr_name, nobs=dep.shape[0])
                )
                if len(eq_data) == 5:
                    self._weights.append(IVData(eq_data[4]))
                else:
                    dep_shape = self._dependent[-1].shape
                    self._weights.append(IVData(np.ones(dep_shape)))

            elif isinstance(eq_data, (dict, Mapping)):
                dep = IVData(eq_data["dependent"], var_name=dep_name)
                self._dependent.append(dep)

                exog = eq_data.get("exog", None)
                self._exog.append(IVData(exog, var_name=exog_name, nobs=dep.shape[0]))
                current_id = (id(exog),)

                endog_values = eq_data.get("endog", None)
                endog = IVData(endog_values, var_name=endog_name, nobs=dep.shape[0])
                self._endog.append(endog)
                if "endog" in eq_data:
                    current_id += (id(eq_data["endog"]),)
                ids.append(current_id)

                instr_values = eq_data.get("instruments", None)
                instr = IVData(instr_values, var_name=instr_name, nobs=dep.shape[0])
                self._instr.append(instr)

                if "weights" in eq_data:
                    self._weights.append(IVData(eq_data["weights"]))
                else:
                    self._weights.append(IVData(np.ones(dep.shape)))
            else:
                msg = UNKNOWN_EQ_TYPE.format(key=key, type=type(vars))
                raise TypeError(msg)
        self._has_instruments = False
        for instr in self._instr:
            self._has_instruments = self._has_instruments or (instr.shape[1] > 1)

        for i, comps in enumerate(
            zip(
                self._dependent,
                self._exog,
                self._endog,
                self._instr,
                self._weights,
                strict=False,
            )
        ):
            shapes = [a.shape[0] for a in comps]
            if min(shapes) != max(shapes):
                raise ValueError(
                    "Dependent, exogenous, endogenous and "
                    "instruments, and weights, if provided, do "
                    "not have the same number of observations in "
                    "{eq}".format(eq=self._eq_labels[i])
                )

        if self._missing == "observed":
            self._validate_observed_data()
        else:
            self._drop_missing()
        self._common_exog = len(set(ids)) == 1
        if self._common_exog:
            # Common exog requires weights are also equal
            w0 = self._weights[0].ndarray
            for w in self._weights:
                self._common_exog = self._common_exog and bool(np.all(w.ndarray == w0))
        constant = []
        constant_loc = []

        exog_ivd: IVData
        for dep, exog_ivd, endog, instr, w, label in zip(
            self._dependent,
            self._exog,
            self._endog,
            self._instr,
            self._weights,
            self._eq_labels,
            strict=False,
        ):
            y = cast(
                "linearmodels.typing.data.Float64Array",
                dep.ndarray,
            )
            x = np.concatenate([exog_ivd.ndarray, endog.ndarray], 1, dtype=float)
            z = np.concatenate([exog_ivd.ndarray, instr.ndarray], 1, dtype=float)
            w_arr = cast(
                "linearmodels.typing.data.Float64Array",
                w.ndarray,
            )
            w_arr = w_arr / np.nanmean(w_arr)
            w_sqrt = np.sqrt(w_arr)
            self._w.append(w_arr)
            self._y.append(y)
            self._x.append(x)
            self._z.append(z)
            self._wy.append(y * w_sqrt)
            self._wx.append(x * w_sqrt)
            self._wz.append(z * w_sqrt)
            cols = list(exog_ivd.cols) + list(endog.cols)
            self._param_names.extend([label + "_" + col for col in cols])
            if self._missing == "observed":
                # Identification is assessed on each equation's actually
                # observed design; see _validate_observed_data.
                continue
            if y.shape[0] <= x.shape[1]:
                raise ValueError(
                    "Fewer observations than variables in "
                    "equation `{eq}`".format(eq=label)
                )
            if matrix_rank(x) < x.shape[1]:
                raise ValueError(
                    "Equation `{eq}` regressor array is not full rank".format(eq=label)
                )
            if x.shape[1] > z.shape[1]:
                raise ValueError(
                    "Equation `{eq}` has fewer instruments than "
                    "endogenous variables.".format(eq=label)
                )
            if z.shape[1] > z.shape[0]:
                raise ValueError(
                    "Fewer observations than instruments in "
                    "equation `{eq}`".format(eq=label)
                )
            if matrix_rank(z) < z.shape[1]:
                raise ValueError(
                    "Equation {eq} instrument array is full rank".format(eq=label)
                )

        for rhs in self._x:
            const, const_loc = has_constant(rhs)
            constant.append(const)
            constant_loc.append(const_loc)
        self._has_constant = Series(
            constant, index=[d.cols[0] for d in self._dependent]
        )
        self._constant_loc = constant_loc

    def _drop_missing(self) -> None:
        k = len(self._dependent)
        nobs = self._dependent[0].shape[0]
        self._original_index = Index(self._dependent[0].rows)
        missing = np.zeros(nobs, dtype=bool)

        values = [self._dependent, self._exog, self._endog, self._instr, self._weights]
        for i in range(k):
            for value in values:
                nulls = value[i].isnull
                if nulls.any():
                    missing |= np.asarray(nulls)

        missing_warning(missing, stacklevel=4)
        if np.any(missing):
            for i in range(k):
                self._dependent[i].drop(missing)
                self._exog[i].drop(missing)
                self._endog[i].drop(missing)
                self._instr[i].drop(missing)
                self._weights[i].drop(missing)

    def _reject_observed_weights(self) -> None:
        for key, eq_data in self._equations.items():
            has_weights = False
            if isinstance(eq_data, (tuple, list)):
                has_weights = len(eq_data) == 5
            elif isinstance(eq_data, Mapping):
                has_weights = "weights" in eq_data
            if has_weights:
                raise ObservedSURError(
                    "Observation weights are not supported when "
                    f'missing="observed" (equation {key!r}).'
                )

    def _validate_observed_data(self) -> None:
        """Validate and retain all rows for missing='observed' models."""
        k = len(self._dependent)
        nobs = self._dependent[0].shape[0]
        self._original_index = Index(self._dependent[0].rows)

        # Common length and pandas index checks: data must align explicitly.
        pandas_indexes = []
        for i in range(k):
            n_i = self._dependent[i].shape[0]
            if n_i != nobs:
                raise ObservedSURError(
                    "All equations must have the same number of observations "
                    f"when missing='observed'; equation {self._eq_labels[i]!r} "
                    f"has {n_i} rows but the first equation has {nobs}."
                )
            dep_pd = self._dependent[i].pandas
            if not dep_pd.index.is_unique:
                raise ObservedSURError(
                    f"The index of equation {self._eq_labels[i]!r} contains "
                    "duplicate entries; pandas inputs must have unique indexes."
                )
            pandas_indexes.append(dep_pd.index)
            # Every pandas component of the same equation must share the
            # dependent's index exactly; no silent reindexing is allowed.
            for component_name, ivd in (
                ("exogenous", self._exog[i]),
                ("endogenous", self._endog[i]),
                ("instruments", self._instr[i]),
            ):
                if ivd.shape[1] == 0:
                    continue
                comp_index = ivd.pandas.index
                if not comp_index.is_unique:
                    raise ObservedSURError(
                        f"The {component_name} index of equation "
                        f"{self._eq_labels[i]!r} contains duplicate entries; "
                        "pandas inputs must have unique indexes."
                    )
                if not comp_index.equals(dep_pd.index):
                    raise ObservedSURError(
                        f"The {component_name} variables of equation "
                        f"{self._eq_labels[i]!r} do not share the dependent's "
                        "pandas index. Align all inputs explicitly before "
                        "constructing the model; no implicit reindexing is "
                        "performed."
                    )

        first_index = pandas_indexes[0]
        for i in range(1, k):
            if not pandas_indexes[i].equals(first_index):
                raise ObservedSURError(
                    "All equations must share the same, identically ordered "
                    f"pandas index when missing='observed'; equation "
                    f"{self._eq_labels[i]!r} does not match the first "
                    "equation. Align the inputs explicitly before constructing "
                    "the model."
                )

        observed = np.zeros((nobs, k), dtype=bool)
        for i in range(k):
            label = self._eq_labels[i]
            y = cast(
                "linearmodels.typing.data.Float64Array",
                self._dependent[i].ndarray,
            )
            if np.isinf(y).any():
                raise ObservedSURError(
                    f"Dependent variable of equation {label!r} contains an "
                    "infinite value; infinite values are not permitted even "
                    "when the cell is otherwise unobserved."
                )
            # NaN in the dependent alone marks an unobserved cell.
            observed[:, i] = np.isfinite(y.squeeze(1))

            for name, ivd in (
                ("exogenous", self._exog[i]),
                ("endogenous", self._endog[i]),
                ("instruments", self._instr[i]),
            ):
                if ivd.shape[1] == 0:
                    continue
                arr = cast("linearmodels.typing.data.Float64Array", ivd.ndarray)
                if not np.isfinite(arr).all():
                    raise ObservedSURError(
                        f"The {name} variables of equation {label!r} contain "
                        "missing or infinite values. With missing='observed', "
                        "only dependent-variable NaNs may denote unobserved "
                        "cells, and the check applies to every row regardless "
                        "of whether the dependent is observed."
                    )

        if not observed.any():
            raise ObservedSURError(
                "Every dependent cell is missing; the model contains no "
                "observations and cannot be fit."
            )
        self._observed_mask = observed
        # Rows where no equation contributes any likelihood term. They remain
        # in every data array and in all outputs but never enter estimation.
        self._empty_rows = np.flatnonzero(~observed.any(axis=1))

        # Per-equation identification on the equation's observed design.
        for i in range(k):
            rows = np.flatnonzero(observed[:, i])
            n_i = rows.shape[0]
            x = np.concatenate(
                [self._exog[i].ndarray, self._endog[i].ndarray], 1, dtype=float
            )
            if n_i == 0:
                raise ObservedSURError(
                    f"Equation {self._eq_labels[i]!r} has no observed "
                    "dependent values and cannot be estimated."
                )
            if n_i <= x.shape[1]:
                raise ObservedSURError(
                    f"Equation {self._eq_labels[i]!r} has {n_i} observed "
                    f"rows but {x.shape[1]} regressors; it is not identified."
                )
            if matrix_rank(x[rows]) < x.shape[1]:
                raise ObservedSURError(
                    f"Equation {self._eq_labels[i]!r}'s observed design "
                    "matrix is not full rank."
                )

        if self._sigma is not None:
            check_known_sigma(self._sigma, k)

    @property
    def missing(self) -> str:
        """Missing-observation handling mode ('drop' or 'observed')."""
        return self._missing

    @property
    def observed_mask(self) -> DataFrame:
        """Boolean frame marking observed dependent cells (observed mode)."""
        if self._missing != "observed":
            raise AttributeError(
                "observed_mask is only available when missing='observed'."
            )
        return DataFrame(
            self._observed_mask,
            index=self._original_index,
            columns=self._eq_labels,
        )

    def __repr__(self) -> str:
        return self.__str__() + f"\nid: {hex(id(self))}"

    def __str__(self) -> str:
        out = self._model_name + ", "
        out += f"{len(self._y)} Equations:\n"
        eqns = ", ".join(self._equations.keys())
        out += "\n".join(textwrap.wrap(eqns, 70))
        if self._common_exog:
            out += "\nCommon Exogenous Variables"
        return out

    def predict(
        self,
        params: linearmodels.typing.data.ArrayLike,
        *,
        equations: (
            Mapping[str, Mapping[str, linearmodels.typing.data.ArrayLike]] | None
        ) = None,
        data: DataFrame | None = None,
        eval_env: int = 1,
    ) -> DataFrame:
        """
        Predict values for additional data

        Parameters
        ----------
        params : array_like
            Model parameters (nvar by 1)
        equations : dict
            Dictionary-like structure containing exogenous and endogenous
            variables.  Each key is an equations label and must
            match the labels used to fir the model. Each value must be a
            dictionary with keys "exog" and "endog". If predictions are not
            required for one of more of the model equations, these keys can
            be omitted.
        data : DataFrame
            Values to use when making predictions from a model constructed
            from a formula
        eval_env : int
            Depth to  use when evaluating formulas.

        Returns
        -------
        predictions : DataFrame
            Fitted values from supplied data and parameters

        Notes
        -----
        If `data` is not none, then `equations` must be none.
        Predictions from models constructed using formulas can
        be computed using either `equations`, which will treat these are
        arrays of values corresponding to the formula-process data, or using
        `data` which will be processed using the formula used to construct the
        values corresponding to the original model specification.

        When using `exog` and `endog`, the regressor array for a particular
        equation is assembled as
        `[equations[eqn]["exog"], equations[eqn]["endog"]]` where `eqn` is
        an equation label. These must correspond to the columns in the
        estimated model.
        """

        if data is not None:
            assert self.formula is not None
            parser = SystemFormulaParser(
                self.formula, data=data, context=capture_context(eval_env)
            )
            equations_d = parser.data
        else:
            if equations is None:
                raise ValueError("One of equations or data must be provided.")
            assert equations is not None
            equations_d = {k: dict(v) for k, v in equations.items()}
        params = np.atleast_2d(np.asarray(params))
        if params.shape[0] == 1:
            params = params.T
        nx = int(sum(_x.shape[1] for _x in self._x))
        if params.shape[0] != nx:
            raise ValueError(
                f"Parameters must have {nx} elements; found {params.shape[0]}."
            )
        loc = 0
        out = {}
        for i, label in enumerate(self._eq_labels):
            kx = self._x[i].shape[1]
            if label in equations_d:
                b = params[loc : loc + kx]
                eqn = equations_d[label]
                exog = eqn.get("exog", None)
                endog = eqn.get("endog", None)
                if exog is None and endog is None:
                    loc += kx
                    continue

                if exog is not None:
                    exog_endog = IVData(exog).pandas
                    if endog is not None:
                        endog_ivd = IVData(endog)
                        exog_endog = concat([exog_endog, endog_ivd.pandas], axis=1)
                else:
                    exog_endog = IVData(endog).pandas

                fitted = np.asarray(exog_endog) @ b
                fitted_df = DataFrame(fitted, index=exog_endog.index, columns=[label])
                out[label] = fitted_df
            loc += kx
        out_df = reduce(
            lambda left, right: left.merge(
                right, how="outer", left_index=True, right_index=True
            ),
            [out[key] for key in out],
        )
        return out_df

    def _multivariate_ls_fit(
        self,
    ) -> tuple[
        linearmodels.typing.data.Float64Array,
        linearmodels.typing.data.Float64Array,
    ]:
        wy, wx, wxhat = self._wy, self._wx, self._wxhat
        k = len(wxhat)

        xpx = blocked_inner_prod(wxhat, np.eye(len(wxhat)))
        _xpy = [wxhat[i].T @ wy[i] for i in range(k)]
        xpy = np.vstack(_xpy)
        beta = _parameters_from_xprod(xpx, xpy, constraints=self.constraints)

        loc = 0
        eps = []
        for i in range(k):
            nb = wx[i].shape[1]
            b = beta[loc : loc + nb]
            eps.append(wy[i] - wx[i] @ b)
            loc += nb
        eps_arr = np.hstack(eps)

        return beta, eps_arr

    def _construct_xhat(self) -> None:
        k = len(self._x)
        self._xhat = []
        self._wxhat = []
        for i in range(k):
            x, z = self._x[i], self._z[i]
            if z.shape == x.shape and np.all(z == x):
                # OLS, no instruments
                self._xhat.append(x)
                self._wxhat.append(self._wx[i])
            else:
                delta = lstsq(z, x, rcond=None)[0]
                xhat = z @ delta
                self._xhat.append(xhat)
                w = self._w[i]
                self._wxhat.append(xhat * np.sqrt(w))

    def _gls_estimate(
        self,
        eps: linearmodels.typing.data.Float64Array,
        nobs: int,
        total_cols: int,
        ci: Sequence[int],
        full_cov: bool,
        debiased: bool,
    ) -> tuple[
        linearmodels.typing.data.Float64Array,
        linearmodels.typing.data.Float64Array,
        linearmodels.typing.data.Float64Array,
        linearmodels.typing.data.Float64Array,
    ]:
        """Core estimation routine for iterative GLS"""
        wy, wx, wxhat = self._wy, self._wx, self._wxhat

        if self._sigma is None:
            sigma = eps.T @ eps / nobs
            sigma *= self._sigma_scale(debiased)
        else:
            sigma = self._sigma
        est_sigma = sigma

        if not full_cov:
            sigma = np.diag(np.diag(sigma))
        sigma_inv = cast("linearmodels.typing.data.Float64Array", inv(sigma))

        k = len(wy)

        xpx = blocked_inner_prod(wxhat, sigma_inv)
        xpy = np.zeros((total_cols, 1))
        for i in range(k):
            sy = np.zeros((nobs, 1))
            for j in range(k):
                sy += sigma_inv[i, j] * wy[j]
            xpy[ci[i] : ci[i + 1]] = wxhat[i].T @ sy

        beta = _parameters_from_xprod(xpx, xpy, constraints=self.constraints)

        loc = 0
        for j in range(k):
            _wx = wx[j]
            _wy = wy[j]
            kx = _wx.shape[1]
            eps[:, [j]] = _wy - _wx @ beta[loc : loc + kx]
            loc += kx

        return beta, eps, sigma, est_sigma

    def _multivariate_ls_finalize(
        self,
        beta: linearmodels.typing.data.Float64Array,
        eps: linearmodels.typing.data.Float64Array,
        sigma: linearmodels.typing.data.Float64Array,
        cov_type: str,
        **cov_config: bool,
    ) -> SystemResults:
        k = len(self._wx)

        # Covariance estimation
        cov_estimator = COV_EST[cov_type]
        cov_est = cov_estimator(
            self._wxhat,
            eps,
            sigma,
            sigma,
            gls=False,
            constraints=self._constraints,
            **cov_config,
        )
        cov = cov_est.cov

        individual = AttrDict()
        debiased = cov_config.get("debiased", False)
        for i in range(k):
            wy = wye = self._wy[i]
            w = self._w[i]
            cons = bool(self.has_constant.iloc[i])
            if cons:
                wc = np.ones_like(wy) * np.sqrt(w)
                wye = wy - wc @ lstsq(wc, wy, rcond=None)[0]
            total_ss = float(np.squeeze(wye.T @ wye))
            stats = self._common_indiv_results(
                i,
                beta,
                cov,
                eps,
                eps,
                "OLS",
                cov_type,
                cov_est,
                0,
                debiased,
                cons,
                total_ss,
            )
            key = self._eq_labels[i]
            individual[key] = stats

        nobs = eps.size
        results = self._common_results(
            beta, cov, "OLS", 0, nobs, cov_type, sigma, individual, debiased
        )
        results["wresid"] = results.resid
        results["cov_estimator"] = cov_est
        results["cov_config"] = cov_est.cov_config
        individual = results["individual"]
        r2s = [individual[eq].r2 for eq in individual]
        results["system_r2"] = self._system_r2(eps, sigma, "ols", False, debiased, r2s)

        return SystemResults(results)

    @property
    def has_constant(self) -> Series:
        """Vector indicating which equations contain constants"""
        return self._has_constant

    def _f_stat(
        self, stats: AttrDict, debiased: bool
    ) -> WaldTestStatistic | InvalidTestStatistic:
        cov = stats.cov
        k = cov.shape[0]
        sel = list(range(k))
        if stats.has_constant:
            sel.pop(stats.constant_loc)
        cov = cov[sel][:, sel]
        params = stats.params[sel]
        df = params.shape[0]
        nobs = stats.nobs
        null = "All parameters ex. constant are zero"
        name = "Equation F-statistic"
        try:
            stat = float(np.squeeze(params.T @ inv(cov) @ params))

        except np.linalg.LinAlgError:
            return InvalidTestStatistic(
                "Covariance is singular, possibly due to constraints.", name=name
            )

        if debiased:
            total_reg = np.sum([s.shape[1] for s in self._wx])
            df_denom = len(self._wx) * nobs - total_reg
            wald = WaldTestStatistic(stat / df, null, df, df_denom=df_denom, name=name)
        else:
            return WaldTestStatistic(stat, null=null, df=df, name=name)

        return wald

    def _common_indiv_results(
        self,
        index: int,
        beta: linearmodels.typing.data.Float64Array,
        cov: linearmodels.typing.data.Float64Array,
        wresid: linearmodels.typing.data.Float64Array,
        resid: linearmodels.typing.data.Float64Array,
        method: str,
        cov_type: str,
        cov_est: (
            HomoskedasticCovariance
            | HeteroskedasticCovariance
            | KernelCovariance
            | ClusteredCovariance
            | GMMHeteroskedasticCovariance
            | GMMHomoskedasticCovariance
        ),
        iter_count: int,
        debiased: bool,
        constant: bool,
        total_ss: float,
        *,
        weight_est: None | (
            HomoskedasticWeightMatrix | HeteroskedasticWeightMatrix | KernelWeightMatrix
        ) = None,
    ) -> AttrDict:
        loc = 0
        for i in range(index):
            loc += self._wx[i].shape[1]
        i = index
        stats = AttrDict()
        # Static properties
        stats["eq_label"] = self._eq_labels[i]
        stats["dependent"] = self._dependent[i].cols[0]
        stats["instruments"] = (
            self._instr[i].cols if self._instr[i].shape[1] > 0 else None
        )
        stats["endog"] = self._endog[i].cols if self._endog[i].shape[1] > 0 else None
        stats["method"] = method
        stats["cov_type"] = cov_type
        stats["cov_estimator"] = cov_est
        stats["cov_config"] = cov_est.cov_config
        stats["weight_estimator"] = weight_est
        stats["index"] = self._dependent[i].rows
        stats["original_index"] = self._original_index
        stats["iter"] = iter_count
        stats["debiased"] = debiased
        stats["has_constant"] = constant
        assert self._constant_loc is not None
        stats["constant_loc"] = self._constant_loc[i]

        # Parameters, errors and measures of fit
        wxi = self._wx[i]
        nobs, df = wxi.shape
        b = beta[loc : loc + df]
        e = wresid[:, [i]]
        nobs = e.shape[0]
        df_c = nobs - int(constant)
        df_r = nobs - df

        stats["params"] = b
        stats["cov"] = cov[loc : loc + df, loc : loc + df]
        stats["wresid"] = e
        stats["nobs"] = nobs
        stats["df_model"] = df
        stats["resid"] = resid[:, [i]]
        stats["fitted"] = self._x[i] @ b
        stats["resid_ss"] = float(np.squeeze(resid[:, [i]].T @ resid[:, [i]]))
        stats["total_ss"] = total_ss
        stats["r2"] = 1.0 - stats.resid_ss / stats.total_ss
        stats["r2a"] = 1.0 - (stats.resid_ss / df_r) / (stats.total_ss / df_c)

        names = self._param_names[loc : loc + df]
        offset = len(stats.eq_label) + 1
        stats["param_names"] = [n[offset:] for n in names]

        # F-statistic
        stats["f_stat"] = self._f_stat(stats, debiased)

        return stats

    def _common_results(
        self,
        beta: linearmodels.typing.data.Float64Array,
        cov: linearmodels.typing.data.Float64Array,
        method: str,
        iter_count: int,
        nobs: int,
        cov_type: str,
        sigma: linearmodels.typing.data.Float64Array,
        individual: AttrDict,
        debiased: bool,
    ) -> AttrDict:
        results = AttrDict()
        results["method"] = method
        results["iter"] = iter_count
        results["nobs"] = nobs
        results["cov_type"] = cov_type
        results["index"] = self._dependent[0].rows
        results["original_index"] = self._original_index
        names = list(individual.keys())
        results["sigma"] = DataFrame(sigma, columns=names, index=names)
        results["individual"] = individual
        results["params"] = beta
        results["df_model"] = beta.shape[0]
        results["param_names"] = self._param_names
        results["cov"] = cov
        results["debiased"] = debiased

        total_ss = resid_ss = 0.0
        residuals = []
        for key in individual:
            total_ss += individual[key].total_ss
            resid_ss += individual[key].resid_ss
            residuals.append(individual[key].resid)
        resid = np.hstack(residuals)

        results["resid_ss"] = resid_ss
        results["total_ss"] = total_ss
        results["r2"] = 1.0 - results.resid_ss / results.total_ss
        results["resid"] = resid
        results["constraints"] = self._constraints
        results["model"] = self

        x = self._x
        k = len(x)
        loc = 0
        fitted_vals = []
        for i in range(k):
            nb = x[i].shape[1]
            b = beta[loc : loc + nb]
            fitted_vals.append(x[i] @ b)
            loc += nb
        fitted = np.hstack(fitted_vals)

        results["fitted"] = fitted

        return results

    def _system_r2(
        self,
        eps: linearmodels.typing.data.Float64Array,
        sigma: linearmodels.typing.data.Float64Array,
        method: str,
        full_cov: bool,
        debiased: bool,
        r2s: Sequence[float],
    ) -> Series:
        sigma_resid = sigma

        # System regression on a constant using weights if provided
        wy, w = self._wy, self._w
        wi = [
            cast("linearmodels.typing.data.Float64Array", np.sqrt(weights))
            for weights in w
        ]
        if method == "ols":
            est_sigma = np.eye(len(wy))
        else:  # gls
            est_sigma = sigma
            if not full_cov:
                est_sigma = np.diag(np.diag(est_sigma))
        est_sigma_inv = cast("linearmodels.typing.data.Float64Array", inv(est_sigma))
        nobs = wy[0].shape[0]
        k = len(wy)
        xpx = blocked_inner_prod(wi, est_sigma_inv)
        xpy = np.zeros((k, 1))
        for i in range(k):
            sy = np.zeros((nobs, 1))
            for j in range(k):
                sy += est_sigma_inv[i, j] * wy[j]
            xpy[i : (i + 1)] = wi[i].T @ sy

        mu = _parameters_from_xprod(xpx, xpy)
        eps_const = np.hstack([self._y[j] - mu[j] for j in range(k)])
        # Judge
        judge = 1 - (eps**2).sum() / (eps_const**2).sum()
        # Dhrymes
        tot_eps_const_sq = (eps_const**2).sum(0)
        r2s_arr = np.asarray(r2s)
        dhrymes = (r2s_arr * tot_eps_const_sq).sum() / tot_eps_const_sq.sum()

        # Berndt
        sigma_y = (eps_const.T @ eps_const / nobs) * self._sigma_scale(debiased)
        berndt = np.nan
        # Avoid division by 0
        if np.linalg.det(sigma_y) > 0:
            berndt = 1 - np.linalg.det(sigma_resid) / np.linalg.det(sigma_y)

        mcelroy = np.nan
        # Check that the matrix is invertible
        if np.linalg.matrix_rank(sigma) == sigma.shape[0]:
            # McElroy
            sigma_m12 = inv_matrix_sqrt(sigma)
            std_eps = eps @ sigma_m12
            numerator = (std_eps**2).sum()
            std_eps_const = eps_const @ sigma_m12
            denom = (std_eps_const**2).sum()
            mcelroy = 1.0 - numerator / denom
        r2 = {"mcelroy": mcelroy, "berndt": berndt, "judge": judge, "dhrymes": dhrymes}
        return Series(r2)

    def _gls_finalize(
        self,
        beta: linearmodels.typing.data.Float64Array,
        sigma: linearmodels.typing.data.Float64Array,
        full_sigma: linearmodels.typing.data.Float64Array,
        est_sigma: linearmodels.typing.data.Float64Array,
        gls_eps: linearmodels.typing.data.Float64Array,
        eps: linearmodels.typing.data.Float64Array,
        full_cov: bool,
        cov_type: str,
        iter_count: int,
        **cov_config: bool,
    ) -> SystemResults:
        """Collect results to return after GLS estimation"""
        k = len(self._wy)

        # Covariance estimation
        cov_estimator = COV_EST[cov_type]
        gls_eps = np.reshape(gls_eps, (k, gls_eps.shape[0] // k)).T
        eps = np.reshape(eps, (k, eps.shape[0] // k)).T
        cov_est = cov_estimator(
            self._wxhat,
            eps,
            sigma,
            full_sigma,
            gls=True,
            constraints=self._constraints,
            **cov_config,
        )
        cov = cov_est.cov

        # Repackage results for individual equations
        individual = AttrDict()
        debiased = cov_config.get("debiased", False)
        method = "Iterative GLS" if iter_count > 1 else "GLS"
        for i in range(k):
            cons = bool(self.has_constant.iloc[i])

            if cons:
                c = np.sqrt(self._w[i])
                ye = self._wy[i] - c @ lstsq(c, self._wy[i], rcond=None)[0]
            else:
                ye = self._wy[i]
            total_ss = float(np.squeeze(ye.T @ ye))
            stats = self._common_indiv_results(
                i,
                beta,
                cov,
                gls_eps,
                eps,
                method,
                cov_type,
                cov_est,
                iter_count,
                debiased,
                cons,
                total_ss,
            )
            key = self._eq_labels[i]
            individual[key] = stats

        # Populate results dictionary
        nobs = eps.size
        results = self._common_results(
            beta,
            cov,
            method,
            iter_count,
            nobs,
            cov_type,
            est_sigma,
            individual,
            debiased,
        )

        # wresid is different between GLS and OLS
        wresiduals = [
            individual[individual_key].wresid for individual_key in individual
        ]
        wresid = np.hstack(wresiduals)
        results["wresid"] = wresid
        results["cov_estimator"] = cov_est
        results["cov_config"] = cov_est.cov_config
        individual = results["individual"]
        r2s = [individual[eq].r2 for eq in individual]
        results["system_r2"] = self._system_r2(
            eps, sigma, "gls", full_cov, debiased, r2s
        )

        return SystemResults(results)

    def _sigma_scale(
        self, debiased: bool
    ) -> float | linearmodels.typing.data.Float64Array:
        if not debiased:
            return 1.0
        nobs = float(self._wx[0].shape[0])
        scales = np.array([nobs - x.shape[1] for x in self._wx], dtype=np.float64)
        scales = cast("linearmodels.typing.data.Float64Array", np.sqrt(nobs / scales))
        return scales[:, None] @ scales[None, :]

    @property
    def constraints(self) -> LinearConstraint | None:
        """
        Model constraints

        Returns
        -------
        cons : {LinearConstraint, None}
            Constraint object
        """
        return self._constraints

    def add_constraints(self, r: DataFrame, q: Series | None = None) -> None:
        r"""
        Add parameter constraints to a model.

        Parameters
        ----------
        r : DataFrame
            Constraint matrix. nconstraints by nparameters
        q : Series
            Constraint values (nconstraints).  If not set, set to 0

        Notes
        -----
        Constraints are of the form

        .. math ::

            r \beta = q

        The property `param_names` can be used to determine the order of
        parameters.
        """
        if self._missing == "observed":
            raise ObservedSURError(
                "Linear parameter constraints are not supported with "
                'missing="observed".'
            )
        self._constraints = LinearConstraint(
            r, q=q, num_params=len(self._param_names), require_pandas=True
        )

    def reset_constraints(self) -> None:
        """Remove all model constraints"""
        self._constraints = None

    @property
    def param_names(self) -> list[str]:
        """
        Model parameter names

        Returns
        -------
        names : list[str]
            Normalized, unique model parameter names
        """
        return self._param_names


class _LSSystemModelBase(_SystemModelBase):
    """Base class for least-squares-based system estimators"""

    def fit(
        self,
        *,
        method: Optional[Literal["ols", "gls"]] = None,
        full_cov: bool = True,
        iterate: bool = False,
        iter_limit: int = 100,
        tol: float = 1e-6,
        cov_type: str = "robust",
        **cov_config: bool,
    ) -> SystemResults:
        """
        Estimate model parameters

        Parameters
        ----------
        method : {None, "gls", "ols"}
            Estimation method.  Default auto selects based on regressors,
            using OLS only if all regressors are identical. The other two
            arguments force the use of GLS or OLS.
        full_cov : bool
            Flag indicating whether to utilize information in correlations
            when estimating the model with GLS
        iterate : bool
            Flag indicating to iterate GLS until convergence of iter limit
            iterations have been completed
        iter_limit : int
            Maximum number of iterations for iterative GLS
        tol : float
            Tolerance to use when checking for convergence in iterative GLS
        cov_type : str
            Name of covariance estimator. Valid options are

            * "unadjusted", "homoskedastic" - Classic covariance estimator
            * "robust", "heteroskedastic" - Heteroskedasticity robust
              covariance estimator
            * "kernel" - Allows for heteroskedasticity and autocorrelation
            * "clustered" - Allows for 1 and 2-way clustering of errors
              (Rogers).

        cov_config
            Additional parameters to pass to covariance estimator. All
            estimators support debiased which employs a small-sample adjustment

        Returns
        -------
        results : SystemResults
            Estimation results

        See Also
        --------
        linearmodels.system.covariance.HomoskedasticCovariance
        linearmodels.system.covariance.HeteroskedasticCovariance
        linearmodels.system.covariance.KernelCovariance
        linearmodels.system.covariance.ClusteredCovariance
        """
        if method is None:
            method = (
                "ols" if (self._common_exog and self._constraints is None) else "gls"
            )
        else:
            if method.lower() not in ("ols", "gls"):
                raise ValueError(
                    f"method must be 'ols' or 'gls' when not None. Got {method}."
                )
            method = cast("Literal['ols', 'gls']", method.lower())

        cov_type = cov_type.lower()
        if cov_type not in COV_TYPES:
            raise ValueError(f"Unknown cov_type: {cov_type}")
        cov_type = COV_TYPES[cov_type]
        k = len(self._dependent)
        col_sizes = [0] + [v.shape[1] for v in self._x]
        col_idx = [int(i) for i in np.cumsum(col_sizes)]
        total_cols = col_idx[-1]
        self._construct_xhat()
        beta, eps = self._multivariate_ls_fit()
        nobs = eps.shape[0]
        debiased = cov_config.get("debiased", False)
        full_sigma = sigma = (eps.T @ eps / nobs) * self._sigma_scale(debiased)

        if method == "ols":
            return self._multivariate_ls_finalize(
                beta, eps, sigma, cov_type, **cov_config
            )

        beta_hist = [beta]
        nobs = eps.shape[0]
        iter_count = 0
        delta = np.inf
        while (
            (iter_count < iter_limit and iterate) or iter_count == 0
        ) and delta >= tol:
            beta, eps, sigma, est_sigma = self._gls_estimate(
                eps, nobs, total_cols, col_idx, full_cov, debiased
            )
            beta_hist.append(beta)
            diff = beta_hist[-1] - beta_hist[-2]
            delta = float(np.sqrt(np.mean(diff**2)))
            iter_count += 1

        sigma_m12 = inv_matrix_sqrt(sigma)
        wy = blocked_column_product(self._wy, sigma_m12)
        wx = blocked_diag_product(self._wx, sigma_m12)
        gls_eps = wy - wx @ beta

        y = blocked_column_product(self._y, np.eye(k))
        x = blocked_diag_product(self._x, np.eye(k))
        eps = y - x @ beta

        return self._gls_finalize(
            beta,
            sigma,
            full_sigma,
            est_sigma,
            gls_eps,
            eps,
            full_cov,
            cov_type,
            iter_count,
            **cov_config,
        )


class IV3SLS(_LSSystemModelBase):
    r"""
    Three-stage Least Squares (3SLS) Estimator

    Parameters
    ----------
    equations : dict
        Dictionary-like structure containing dependent, exogenous, endogenous
        and instrumental variables.  Each key is an equations label and must
        be a string. Each value must be either a tuple of the form (dependent,
        exog, endog, instrument[, weights]) or a dictionary with keys "dependent",
        and at least one of "exog" or "endog" and "instruments".  When using a
        tuple, values must be provided for all 4 variables, although either
        empty arrays or `None` can be passed if a category of variable is not
        included in a model. The dictionary may contain optional keys for
        "exog", "endog", "instruments", and "weights". "exog" can be omitted
        if all variables in an equation are endogenous. Alternatively, "exog"
        can contain either an empty array or `None` to indicate that an
        equation contains no exogenous regressors. Similarly "endog" and
        "instruments" can either be omitted or may contain an empty array (or
        `None`) if all variables in an equation are exogenous.
    sigma : array_like
        Prespecified residual covariance to use in GLS estimation. If not
        provided, FGLS is implemented based on an estimate of sigma.

    Notes
    -----
    Estimates a set of regressions which are seemingly unrelated in the sense
    that separate estimation would lead to consistent parameter estimates.
    Each equation is of the form

    .. math::

        y_{i,k} = x_{i,k}\beta_i + \epsilon_{i,k}

    where k denotes the equation and i denoted the observation index. By
    stacking vertically arrays of dependent and placing the exogenous
    variables into a block diagonal array, the entire system can be compactly
    expressed as

    .. math::

        Y = X\beta + \epsilon

    where

    .. math::

        Y = \left[\begin{array}{x}Y_1 \\ Y_2 \\ \vdots \\ Y_K\end{array}\right]

    and

    .. math::

        X = \left[\begin{array}{cccc}
                 X_1 & 0 & \ldots & 0 \\
                 0 & X_2 & \dots & 0 \\
                 \vdots & \vdots & \ddots & \vdots \\
                 0 & 0 & \dots & X_K
            \end{array}\right]

    The system instrumental variable (IV) estimator is

    .. math::

        \hat{\beta}_{IV} & = (X'Z(Z'Z)^{-1}Z'X)^{-1}X'Z(Z'Z)^{-1}Z'Y \\
                         & = (\hat{X}'\hat{X})^{-1}\hat{X}'Y

    where :math:`\hat{X} = Z(Z'Z)^{-1}Z'X` and.  When certain conditions are
    satisfied, a GLS estimator of the form

    .. math::

        \hat{\beta}_{3SLS} = (\hat{X}'\Omega^{-1}\hat{X})^{-1}\hat{X}'\Omega^{-1}Y

    can improve accuracy of coefficient estimates where

    .. math::

        \Omega = \Sigma \otimes I_N

    where :math:`\Sigma` is the covariance matrix of the residuals.
    """

    def __init__(
        self,
        equations: Mapping[
            str,
            Mapping[str, linearmodels.typing.data.ArrayLike | None]
            | Sequence[linearmodels.typing.data.ArrayLike | None],
        ],
        *,
        sigma: linearmodels.typing.data.ArrayLike | None = None,
    ) -> None:
        super().__init__(equations, sigma=sigma)

    @classmethod
    def multivariate_iv(
        cls,
        dependent: linearmodels.typing.data.ArrayLike,
        exog: linearmodels.typing.data.ArrayLike | None = None,
        endog: linearmodels.typing.data.ArrayLike | None = None,
        instruments: linearmodels.typing.data.ArrayLike | None = None,
    ) -> IV3SLS:
        """
        Interface for specification of multivariate IV models

        Parameters
        ----------
        dependent : array_like
            nobs by ndep array of dependent variables
        exog : array_like
            nobs by nexog array of exogenous regressors common to all models
        endog : array_like
            nobs by nendog array of endogenous regressors common to all models
        instruments : array_like
            nobs by ninstr array of instruments to use in all equations

        Returns
        -------
        model : IV3SLS
            Model instance

        Notes
        -----
        At least one of exog or endog must be provided.

        Utility function to simplify the construction of multivariate IV
        models which all use the same regressors and instruments. Constructs
        the dictionary of equations from the variables using the common
        exogenous, endogenous and instrumental variables.
        """
        equations = {}
        dependent_ivd = IVData(dependent, var_name="dependent")
        if exog is None and endog is None:
            raise ValueError("At least one of exog or endog must be provided")
        exog_ivd = IVData(exog, var_name="exog")
        endog_ivd = IVData(endog, var_name="endog", nobs=dependent.shape[0])
        instr_ivd = IVData(instruments, var_name="instruments", nobs=dependent.shape[0])
        for col in dependent_ivd.pandas:
            equations[str(col)] = {
                # TODO: Bug in pandas-stubs
                #  https://github.com/pandas-dev/pandas-stubs/issues/97
                "dependent": dependent_ivd.pandas[[col]],
                "exog": exog_ivd.pandas,
                "endog": endog_ivd.pandas,
                "instruments": instr_ivd.pandas,
            }

        return cls(equations)

    @classmethod
    def from_formula(
        cls,
        formula: str | dict[str, str],
        data: DataFrame,
        *,
        sigma: linearmodels.typing.data.ArrayLike | None = None,
        weights: Mapping[str, linearmodels.typing.data.ArrayLike] | None = None,
    ) -> IV3SLS:
        """
        Specify a 3SLS using the formula interface

        Parameters
        ----------
        formula : {str, dict-like}
            Either a string or a dictionary of strings where each value in
            the dictionary represents a single equation. See Notes for a
            description of the accepted syntax
        data : DataFrame
            Frame containing named variables
        sigma : array_like
            Prespecified residual covariance to use in GLS estimation. If
            not provided, FGLS is implemented based on an estimate of sigma.
        weights : dict-like
            Dictionary like object (e.g. a DataFrame) containing variable
            weights.  Each entry must have the same number of observations as
            data.  If an equation label is not a key weights, the weights will
            be set to unity

        Returns
        -------
        model : IV3SLS
            Model instance

        Notes
        -----
        Models can be specified in one of two ways. The first uses curly
        braces to encapsulate equations.  The second uses a dictionary
        where each key is an equation name.

        Examples
        --------
        The simplest format uses standard formulas for each equation
        in a dictionary.  Best practice is to use an Ordered Dictionary

        >>> import pandas as pd
        >>> import numpy as np
        >>> cols = ["y1", "x1_1", "x1_2", "z1", "y2", "x2_1", "x2_2", "z2"]
        >>> data = pd.DataFrame(np.random.randn(500, 8), columns=cols)
        >>> from linearmodels.system import IV3SLS
        >>> formula = {"eq1": "y1 ~ 1 + x1_1 + [x1_2 ~ z1]",
        ...            "eq2": "y2 ~ 1 + x2_1 + [x2_2 ~ z2]"}
        >>> mod = IV3SLS.from_formula(formula, data)

        The second format uses curly braces {} to surround distinct equations

        >>> formula = "{y1 ~ 1 + x1_1 + [x1_2 ~ z1]} {y2 ~ 1 + x2_1 + [x2_2 ~ z2]}"
        >>> mod = IV3SLS.from_formula(formula, data)

        It is also possible to include equation labels when using curly braces

        >>> formula = "{eq1: y1 ~ x1_1 + [x1_2 ~ z1]} {eq2: y2 ~ 1 + [x2_2 ~ z2]}"
        >>> mod = IV3SLS.from_formula(formula, data)
        """
        context = capture_context(1)
        parser = SystemFormulaParser(formula, data, weights, context=context)
        eqns = parser.data
        mod = cls(eqns, sigma=sigma)
        mod.formula = formula
        return mod


class SUR(_LSSystemModelBase):
    r"""
    Seemingly unrelated regression estimation (SUR/SURE)

    Parameters
    ----------
    equations : dict
        Dictionary-like structure containing dependent and exogenous variable
        values.  Each key is an equations label and must be a string. Each
        value must be either a tuple of the form (dependent,
        exog, [weights]) or a dictionary with keys "dependent" and "exog" and
        the optional key "weights".
    sigma : array_like
        Prespecified residual covariance to use in GLS estimation. If not
        provided, FGLS is implemented based on an estimate of sigma.
    missing : {"drop", "observed"}
        How to handle NaN values in the dependent variables. The default
        ``"drop"`` removes every row containing a missing value in any
        equation before estimation. ``"observed"`` keeps the original rows
        and fits the joint Gaussian likelihood of the dependent cells that
        are actually observed; see the notes and :meth:`fit` for details.

    Notes
    -----
    Estimates a set of regressions which are seemingly unrelated in the sense
    that separate estimation would lead to consistent parameter estimates.
    Each equation is of the form

    .. math::

        y_{i,k} = x_{i,k}\beta_i + \epsilon_{i,k}

    where k denotes the equation and i denoted the observation index. By
    stacking vertically arrays of dependent and placing the exogenous
    variables into a block diagonal array, the entire system can be compactly
    expressed as

    .. math::

        Y = X\beta + \epsilon

    where

    .. math::

        Y = \left[\begin{array}{x}Y_1 \\ Y_2 \\ \vdots \\ Y_K\end{array}\right]

    and

    .. math::

        X = \left[\begin{array}{cccc}
                 X_1 & 0 & \ldots & 0 \\
                 0 & X_2 & \dots & 0 \\
                 \vdots & \vdots & \ddots & \vdots \\
                 0 & 0 & \dots & X_K
            \end{array}\right]

    The system OLS estimator is

    .. math::

        \hat{\beta}_{OLS} = (X'X)^{-1}X'Y

    When certain conditions are satisfied, a GLS estimator of the form

    .. math::

        \hat{\beta}_{GLS} = (X'\Omega^{-1}X)^{-1}X'\Omega^{-1}Y

    can improve accuracy of coefficient estimates where

    .. math::

        \Omega = \Sigma \otimes I_N

    where :math:`\Sigma` is the covariance matrix of the residuals.

    SUR is a special case of 3SLS where there are no endogenous regressors and
    no instruments.

    With ``missing="observed"`` the missing dependent cells are never
    imputed. The fit maximizes the observed-data normal likelihood: rows are
    independent, equations observed within a row share the constant
    cross-equation covariance :math:`\Sigma`, and a missing dependent simply
    contributes no density term. Rows where every dependent is missing remain
    in the data but never enter the likelihood. If ``sigma`` is supplied it is
    treated as known (and must be finite, symmetric and positive definite);
    otherwise the coefficients and a positive-definite :math:`\Sigma` are
    estimated jointly. This mode requires ``fit(method="ml")`` and does not
    support observation weights or linear parameter constraints.
    """

    def __init__(
        self,
        equations: Mapping[
            str,
            Mapping[str, linearmodels.typing.data.ArrayLike | None]
            | Sequence[linearmodels.typing.data.ArrayLike | None],
        ],
        *,
        sigma: linearmodels.typing.data.ArrayLike | None = None,
        missing: Literal["drop", "observed"] = "drop",
    ) -> None:
        if not isinstance(equations, Mapping):
            raise TypeError("equations must be a dictionary-like")
        for key in equations:
            if not isinstance(key, str):
                raise ValueError("Equation labels (keys) must be strings")
        reformatted = {}
        for key in equations:
            eqn = equations[key]
            if isinstance(eqn, tuple):
                if len(eqn) == 3:
                    w = eqn[-1]
                    eqn = eqn[:2]
                    eqn = (*eqn, None, None, w)
                else:
                    eqn = (*eqn, None, None)
            reformatted[key] = eqn
        super().__init__(reformatted, sigma=sigma, missing=missing)
        self._model_name = "Seemingly Unrelated Regression (SUR)"

    def fit(  # type: ignore[override]
        self,
        *,
        method: Optional[Literal["ols", "gls", "ml"]] = None,
        full_cov: bool = True,
        iterate: bool = False,
        iter_limit: int = 100,
        tol: float = 1e-6,
        cov_type: str = "robust",
        **cov_config: bool,
    ) -> SystemResults:
        """
        Estimate model parameters

        Parameters
        ----------
        method : {None, "ols", "gls", "ml"}
            Estimation method.  The default ``None`` selects GLS.  Models
            built with ``missing="observed"`` require ``"ml"`` and maximize
            the joint normal likelihood of the observed dependent cells;
            ``"ols"`` and ``"gls"`` raise an error in that mode.
        full_cov : bool
            Flag indicating to use the full residual covariance when
            estimating GLS parameters.  Ignored by the observed-data ML path.
        iterate : bool
            Flag indicating to iterate GLS.  Ignored by the observed-data ML
            path.
        iter_limit : int
            Maximum number of likelihood optimizer iterations.
        tol : float
            Convergence tolerance applied to the analytic likelihood score.
        cov_type : str
            Covariance estimator.  The observed-data ML path supports
            ``"unadjusted"`` (inverse observed information) and
            ``"robust"`` (row-score sandwich); other estimators are rejected.
        **cov_config
            Additional covariance options.  Debiased estimation and kernel
            or clustered covariances are not supported with
            ``missing="observed"``.
        """
        if self._missing == "observed":
            if method != "ml":
                raise ObservedSURError(
                    'Models constructed with missing="observed" only support '
                    'fit(method="ml"). The OLS and GLS estimators require '
                    "complete rectangular data."
                )
            return self._fit_observed_ml(
                iter_limit=iter_limit,
                tol=tol,
                cov_type=cov_type,
                cov_config=cov_config,
            )
        drop_method: Optional[Literal["ols", "gls"]] = (
            None if method is None or method == "ml" else method
        )
        return super().fit(
            method=drop_method,
            full_cov=full_cov,
            iterate=iterate,
            iter_limit=iter_limit,
            tol=tol,
            cov_type=cov_type,
            **cov_config,
        )

    def _fit_observed_ml(
        self,
        *,
        iter_limit: int,
        tol: float,
        cov_type: str,
        cov_config: dict[str, bool],
    ) -> SystemResults:
        if iter_limit <= 0:
            raise ValueError("iter_limit must be a positive integer.")
        if tol <= 0.0:
            raise ValueError("tol must be strictly positive.")
        if self._constraints is not None:
            raise ObservedSURError(
                "Linear parameter constraints are not supported with "
                'missing="observed".'
            )
        normalized = cov_type.lower()
        if normalized not in COV_TYPES or COV_TYPES[normalized] not in (
            "unadjusted",
            "robust",
        ):
            raise ValueError(
                'missing="observed" supports cov_type "unadjusted" and '
                f'"robust" only; got {cov_type!r}.'
            )
        cov_type = COV_TYPES[normalized]
        debiased = bool(cov_config.pop("debiased", False))
        if debiased:
            raise ValueError(
                "Debiased covariance estimation is not supported with "
                'missing="observed".'
            )
        if cov_config:
            key = next(iter(cov_config))
            raise ValueError(
                f"Unsupported covariance configuration {key!r}; observed-data "
                "ML accepts no additional covariance options."
            )

        assert self._observed_mask is not None
        ml_result = fit_observed_sur(
            self._x,
            self._y,
            self._observed_mask,
            sigma=self._sigma,
            iter_limit=iter_limit,
            tol=tol,
        )
        return self._finalize_observed_ml(ml_result, cov_type)

    def _finalize_observed_ml(self, ml_result: Any, cov_type: str) -> SystemResults:
        k = len(self._x)
        beta = ml_result.params
        sigma = ml_result.sigma
        n = self._x[0].shape[0]
        known_sigma = ml_result.known_sigma

        cov_cls = OBSERVED_COV_EST[cov_type]
        cov_est = cov_cls(
            n_params=beta.shape[0],
            hessian=ml_result.hessian,
            row_scores=ml_result.row_scores,
            known_sigma=known_sigma,
        )
        cov = cov_est.cov

        # Residuals and fitted values retain every original row; unobserved
        # cells stay NaN and are never conditionally filled from other eqs.
        resid_full = ml_result.residuals.copy()
        fitted_full = np.full((n, k), np.nan)
        loc = 0
        for i in range(k):
            nb = self._x[i].shape[1]
            b = beta[loc : loc + nb]
            fitted_full[:, i] = (self._x[i] @ b).squeeze(1)
            loc += nb

        assert self._observed_mask is not None
        individual = AttrDict()
        for i in range(k):
            cons = bool(self.has_constant.iloc[i])
            observed_i = self._observed_mask[:, i]
            yi = self._y[i][observed_i]
            if cons:
                ye = yi - yi.mean()
            else:
                ye = yi
            total_ss = float((ye.T @ ye).squeeze())
            stats = self._observed_indiv_results(
                i,
                beta,
                cov,
                resid_full,
                fitted_full,
                cov_type,
                cov_est,
                total_ss,
                cons,
                int(ml_result.nobs_per_eq[i]),
            )
            individual[self._eq_labels[i]] = stats

        # System nobs is the sum of the per-equation observed counts.
        results = AttrDict()
        results["method"] = "ML (observed data)"
        results["iter"] = ml_result.iter_count
        results["nobs"] = int(ml_result.nobs_total)
        results["cov_type"] = cov_type
        results["index"] = self._dependent[0].rows
        results["original_index"] = self._original_index
        names = list(individual.keys())
        results["sigma"] = DataFrame(sigma, columns=names, index=names)
        results["individual"] = individual
        results["params"] = beta
        results["df_model"] = beta.shape[0]
        results["param_names"] = self._param_names
        results["cov"] = cov
        results["debiased"] = False
        results["resid_ss"] = float(np.nansum(resid_full * resid_full))
        results["total_ss"] = float(sum(individual[key].total_ss for key in individual))
        # R2 over observed cells only; no zeros replace missing residuals.
        results["r2"] = 1.0 - results["resid_ss"] / results["total_ss"]
        results["resid"] = resid_full
        results["constraints"] = None
        results["model"] = self
        results["fitted"] = fitted_full
        results["wresid"] = resid_full
        results["cov_estimator"] = cov_est
        results["cov_config"] = cov_est.cov_config
        results["loglik"] = ml_result.loglik
        results["converged"] = ml_result.converged
        results["termination"] = ml_result.termination
        results["score_norm"] = ml_result.score_norm
        results["iter_limit_tol"] = ml_result.tol
        results["known_sigma"] = known_sigma
        results["observed_mask"] = self._observed_mask
        results["empty_rows"] = self._empty_rows
        results["nobs_per_eq"] = ml_result.nobs_per_eq
        results["full_information"] = ml_result.hessian
        results["row_scores"] = ml_result.row_scores
        results["score_rows"] = ml_result.score_rows
        # Likelihood engine retained for diagnostics and advanced checks.
        self._fit_engine_likelihood = ml_result.extra["likelihood"]
        # Rectangular-residual system fit measures are not defined with holes.
        results["system_r2"] = Series(
            np.full(4, np.nan),
            index=["mcelroy", "berndt", "judge", "dhrymes"],
        )
        return ObservedSURResults(results)

    def _observed_indiv_results(  # noqa: PLR0917
        self,
        index: int,
        beta: linearmodels.typing.data.Float64Array,
        cov: linearmodels.typing.data.Float64Array,
        resid_full: linearmodels.typing.data.Float64Array,
        fitted_full: linearmodels.typing.data.Float64Array,
        cov_type: str,
        cov_est: Any,
        total_ss: float,
        constant: bool,
        nobs: int,
    ) -> AttrDict:
        loc = 0
        for i in range(index):
            loc += self._x[i].shape[1]
        i = index
        df = self._x[i].shape[1]
        b = beta[loc : loc + df]
        stats = AttrDict()
        stats["eq_label"] = self._eq_labels[i]
        stats["dependent"] = self._dependent[i].cols[0]
        stats["instruments"] = None
        stats["endog"] = None
        stats["method"] = "ML (observed data)"
        stats["cov_type"] = cov_type
        stats["cov_estimator"] = cov_est
        stats["cov_config"] = cov_est.cov_config
        stats["weight_estimator"] = None
        stats["index"] = self._dependent[i].rows
        stats["original_index"] = self._original_index
        stats["iter"] = 0
        stats["debiased"] = False
        stats["has_constant"] = constant
        assert self._constant_loc is not None
        stats["constant_loc"] = self._constant_loc[i]
        stats["params"] = b
        stats["cov"] = cov[loc : loc + df, loc : loc + df]
        stats["wresid"] = resid_full[:, [i]]
        stats["nobs"] = nobs
        stats["df_model"] = df
        stats["resid"] = resid_full[:, [i]]
        stats["fitted"] = fitted_full[:, [i]]
        assert self._observed_mask is not None
        observed = self._observed_mask[:, i]
        ei = resid_full[observed, i]
        stats["resid_ss"] = float(ei @ ei)
        stats["total_ss"] = total_ss
        stats["r2"] = 1.0 - stats["resid_ss"] / stats["total_ss"]
        df_c = nobs - int(constant)
        df_r = nobs - df
        stats["r2a"] = 1.0 - (stats["resid_ss"] / df_r) / (stats["total_ss"] / df_c)
        names = self._param_names[loc : loc + df]
        offset = len(stats.eq_label) + 1
        stats["param_names"] = [nm[offset:] for nm in names]
        stats["f_stat"] = self._f_stat(stats, False)
        return stats

    @classmethod
    def multivariate_ls(
        cls,
        dependent: linearmodels.typing.data.ArrayLike,
        exog: linearmodels.typing.data.ArrayLike,
    ) -> SUR:
        """
        Interface for specification of multivariate regression models

        Parameters
        ----------
        dependent : array_like
            nobs by ndep array of dependent variables
        exog : array_like
            nobs by nvar array of exogenous regressors common to all models

        Returns
        -------
        model : SUR
            Model instance

        Notes
        -----
        Utility function to simplify the construction of multivariate
        regression models which all use the same regressors. Constructs
        the dictionary of equations from the variables using the common
        exogenous variable.

        Examples
        --------
        A simple CAP-M can be estimated as a multivariate regression

        >>> from linearmodels.datasets import french
        >>> from linearmodels.system import SUR
        >>> data = french.load()
        >>> portfolios = data[["S1V1","S1V5","S5V1","S5V5"]]
        >>> factors = data[["MktRF"]].copy()
        >>> factors["alpha"] = 1
        >>> mod = SUR.multivariate_ls(portfolios, factors)
        """
        equations = {}
        dependent_ivd = IVData(dependent, var_name="dependent")
        exog_ivd = IVData(exog, var_name="exog")
        for col in dependent_ivd.pandas:
            # TODO: Bug in pandas-stubs
            #  https://github.com/pandas-dev/pandas-stubs/issues/97
            equations[str(col)] = (dependent_ivd.pandas[[col]], exog_ivd.pandas)
        return cls(equations)

    @classmethod
    def from_formula(
        cls,
        formula: str | dict[str, str],
        data: DataFrame,
        *,
        sigma: linearmodels.typing.data.ArrayLike | None = None,
        weights: Mapping[str, linearmodels.typing.data.ArrayLike] | None = None,
        missing: Literal["drop", "observed"] = "drop",
    ) -> SUR:
        """
        Specify a SUR using the formula interface

        Parameters
        ----------
        formula : {str, dict[str, str]}
            Either a string or a dictionary of strings where each value in
            the dictionary represents a single equation. See Notes for a
            description of the accepted syntax
        data : DataFrame
            Frame containing named variables
        sigma : array_like
            Prespecified residual covariance to use in GLS estimation. If
            not provided, FGLS is implemented based on an estimate of sigma.
        weights : dict[str, array_like]
            Dictionary like object (e.g. a DataFrame) containing variable
            weights.  Each entry must have the same number of observations as
            data.  If an equation label is not a key weights, the weights will
            be set to unity
        missing : {"drop", "observed"}
            Missing-value handling.  ``"observed"`` retains NaN dependent
            cells as unobserved and requires ``fit(method="ml")``; see the
            class documentation for the supported data requirements.

        Returns
        -------
        model : SUR
            Model instance

        Notes
        -----
        Models can be specified in one of two ways. The first uses curly
        braces to encapsulate equations.  The second uses a dictionary
        where each key is an equation name.

        Examples
        --------
        The simplest format uses standard formulas for each equation
        in a dictionary.  Best practice is to use an Ordered Dictionary

        >>> import pandas as pd
        >>> import numpy as np
        >>> data = pd.DataFrame(np.random.randn(500, 4),
        ...                     columns=["y1", "x1_1", "y2", "x2_1"])
        >>> from linearmodels.system import SUR
        >>> formula = {"eq1": "y1 ~ 1 + x1_1", "eq2": "y2 ~ 1 + x2_1"}
        >>> mod = SUR.from_formula(formula, data)

        The second format uses curly braces {} to surround distinct equations

        >>> formula = "{y1 ~ 1 + x1_1} {y2 ~ 1 + x2_1}"
        >>> mod = SUR.from_formula(formula, data)

        It is also possible to include equation labels when using curly braces

        >>> formula = "{eq1: y1 ~ 1 + x1_1} {eq2: y2 ~ 1 + x2_1}"
        >>> mod = SUR.from_formula(formula, data)
        """
        context = capture_context(1)
        parser = SystemFormulaParser(
            formula,
            data,
            weights,
            context=context,
            allow_missing_dependent=missing == "observed",
        )
        eqns = parser.data
        mod = cls(eqns, sigma=sigma, missing=missing)
        mod.formula = formula
        return mod


class IVSystemGMM(_SystemModelBase):
    r"""
    System Generalized Method of Moments (GMM) estimation of linear IV models

    Parameters
    ----------
    equations : dict
        Dictionary-like structure containing dependent, exogenous, endogenous
        and instrumental variables.  Each key is an equations label and must
        be a string. Each value must be either a tuple of the form (dependent,
        exog, endog, instrument[, weights]) or a dictionary with keys "dependent",
        "exog".  The dictionary may contain optional keys for "endog",
        "instruments", and "weights". Endogenous and/or Instrument can be empty
        if all variables in an equation are exogenous.
    sigma : array_like
        Prespecified residual covariance to use in GLS estimation. If not
        provided, FGLS is implemented based on an estimate of sigma. Only used
        if weight_type is "unadjusted"
    weight_type : str
        Name of moment condition weight function to use in the GMM estimation
    weight_config
        Additional keyword arguments to pass to the moment condition weight
        function

    Notes
    -----
    Estimates a linear model using GMM. Each equation is of the form

    .. math::

        y_{i,k} = x_{i,k}\beta_i + \epsilon_{i,k}

    where k denotes the equation and i denoted the observation index. By
    stacking vertically arrays of dependent and placing the exogenous
    variables into a block diagonal array, the entire system can be compactly
    expressed as

    .. math::

        Y = X\beta + \epsilon

    where

    .. math::

        Y = \left[\begin{array}{x}Y_1 \\ Y_2 \\ \vdots \\ Y_K\end{array}\right]

    and

    .. math::

        X = \left[\begin{array}{cccc}
                 X_1 & 0 & \ldots & 0 \\
                 0 & X_2 & \dots & 0 \\
                 \vdots & \vdots & \ddots & \vdots \\
                 0 & 0 & \dots & X_K
            \end{array}\right]

    The system GMM estimator uses the moment condition

    .. math::

        z_{ij}(y_{ij} - x_{ij}\beta_j) = 0

    where j indexes the equation. The estimator for the coefficients is given
    by

    .. math::

        \hat{\beta}_{GMM} & = (X'ZW^{-1}Z'X)^{-1}X'ZW^{-1}Z'Y \\

    where :math:`W` is a positive definite weighting matrix.
    """

    def __init__(
        self,
        equations: Mapping[
            str,
            Mapping[str, linearmodels.typing.data.ArrayLike | None]
            | Sequence[linearmodels.typing.data.ArrayLike | None],
        ],
        *,
        sigma: linearmodels.typing.data.ArrayLike | None = None,
        weight_type: str = "robust",
        **weight_config: bool | str | float,
    ) -> None:
        super().__init__(equations, sigma=sigma)
        self._weight_type = weight_type
        self._weight_config = weight_config

        if weight_type not in COV_TYPES:
            raise ValueError("Unknown estimator for weight_type")

        if weight_type not in ("unadjusted", "homoskedastic") and sigma is not None:
            warnings.warn(
                "sigma has been provided but the estimated weight "
                "matrix not unadjusted (homoskedastic).  sigma will "
                "be ignored.",
                UserWarning,
                stacklevel=1,
            )
        weight_type = COV_TYPES[weight_type]
        self._weight_est = GMM_W_EST[weight_type](**weight_config)

    def fit(
        self,
        *,
        iter_limit: int = 2,
        tol: float = 1e-6,
        initial_weight: linearmodels.typing.data.Float64Array | None = None,
        cov_type: str = "robust",
        **cov_config: bool | float,
    ) -> GMMSystemResults:
        """
        Estimate model parameters

        Parameters
        ----------
        iter_limit : int
            Maximum number of iterations for iterative GLS
        tol : float
            Tolerance to use when checking for convergence in iterative GLS
        initial_weight : ndarray
            Initial weighting matrix to use in the first step. If not
            specified, uses the average outer-product of the set containing
            the exogenous variables and instruments.
        cov_type : str
            Name of covariance estimator. Valid options are

            * "unadjusted", "homoskedastic" - Classic covariance estimator
            * "robust", "heteroskedastic" - Heteroskedasticity robust
              covariance estimator

        cov_config
            Additional parameters to pass to covariance estimator. All
            estimators support debiased which employs a small-sample adjustment

        Returns
        -------
        GMMSystemResults
            Estimation results
        """
        if cov_type not in COV_TYPES:
            raise ValueError(f"Unknown cov_type: {cov_type}")
        # Parameter estimation
        wx, wy, wz = self._wx, self._wy, self._wz
        k = len(wx)
        nobs = wx[0].shape[0]
        k_total = sum(a.shape[1] for a in wz)
        if initial_weight is None:
            w = blocked_inner_prod(wz, np.eye(k_total)) / nobs
        else:
            w = initial_weight
        assert w is not None
        beta_last = beta = self._blocked_gmm(
            wx,
            wy,
            wz,
            w=cast("linearmodels.typing.data.Float64Array", w),
            constraints=self.constraints,
        )
        _eps = []
        loc = 0
        for i in range(k):
            nb = wx[i].shape[1]
            b = beta[loc : loc + nb]
            _eps.append(wy[i] - wx[i] @ b)
            loc += nb
        eps = np.hstack(_eps)
        sigma = self._weight_est.sigma(eps, wx) if self._sigma is None else self._sigma
        vinv = None
        iters = 1
        norm = 10 * tol + 1
        while iters < iter_limit and norm > tol:
            sigma = (
                self._weight_est.sigma(eps, wx) if self._sigma is None else self._sigma
            )
            w = self._weight_est.weight_matrix(wx, wz, eps, sigma=sigma)
            beta = self._blocked_gmm(
                wx,
                wy,
                wz,
                w=cast("linearmodels.typing.data.Float64Array", w),
                constraints=self.constraints,
            )
            delta = beta_last - beta
            if vinv is None:
                winv = np.linalg.inv(w)
                xpz = blocked_cross_prod(wx, wz, np.eye(k))
                xpz = cast("linearmodels.typing.data.Float64Array", xpz / nobs)
                v = (xpz @ winv @ xpz.T) / nobs
                vinv = inv(v)
            norm = float(np.squeeze(delta.T @ vinv @ delta))
            beta_last = beta

            _eps = []
            loc = 0
            for i in range(k):
                nb = wx[i].shape[1]
                b = beta[loc : loc + nb]
                _eps.append(wy[i] - wx[i] @ b)
                loc += nb
            eps = np.hstack(_eps)
            iters += 1

        cov_type = COV_TYPES[cov_type]
        cov_est = GMM_COV_EST[cov_type]
        cov = cov_est(
            wx, wz, eps, w, sigma=sigma, constraints=self._constraints, **cov_config
        )

        weps = eps
        _eps = []
        loc = 0
        x, y = self._x, self._y
        for i in range(k):
            nb = x[i].shape[1]
            b = beta[loc : loc + nb]
            _eps.append(y[i] - x[i] @ b)
            loc += nb
        eps = np.hstack(_eps)
        iters += 1
        return self._finalize_results(
            beta,
            cov.cov,
            weps,
            eps,
            cast("np.ndarray", w),
            sigma,
            iters - 1,
            cov_type,
            cov_config,
            cov,
        )

    @staticmethod
    def _blocked_gmm(
        x: linearmodels.typing.ArraySequence,
        y: linearmodels.typing.ArraySequence,
        z: linearmodels.typing.ArraySequence,
        *,
        w: linearmodels.typing.data.Float64Array,
        constraints: LinearConstraint | None = None,
    ) -> linearmodels.typing.data.Float64Array:
        k = len(x)
        xpz = blocked_cross_prod(x, z, np.eye(k))
        wi = np.linalg.inv(w)
        xpz_wi_zpx = xpz @ wi @ xpz.T
        zpy_arrs = []
        zpy_arrs = [(z[i].T @ y[i]) for i in range(k)]
        zpy = np.vstack(zpy_arrs)
        xpz_wi_zpy = xpz @ wi @ zpy
        params = _parameters_from_xprod(xpz_wi_zpx, xpz_wi_zpy, constraints=constraints)

        return params

    def _finalize_results(
        self,
        beta: linearmodels.typing.data.Float64Array,
        cov: linearmodels.typing.data.Float64Array,
        weps: linearmodels.typing.data.Float64Array,
        eps: linearmodels.typing.data.Float64Array,
        wmat: linearmodels.typing.data.Float64Array,
        sigma: linearmodels.typing.data.Float64Array,
        iter_count: int,
        cov_type: str,
        cov_config: dict[str, bool | float],
        cov_est: GMMHeteroskedasticCovariance | GMMHomoskedasticCovariance,
    ) -> GMMSystemResults:
        """Collect results to return after GLS estimation"""
        k = len(self._wy)
        # Repackage results for individual equations
        individual = AttrDict()
        debiased = bool(cov_config.get("debiased", False))
        method = f"{iter_count}-Step System GMM"
        if iter_count > 2:
            method = "Iterative System GMM"
        for i in range(k):
            cons = bool(self.has_constant.iloc[i])

            if cons:
                c = np.sqrt(self._w[i])
                ye = self._wy[i] - c @ lstsq(c, self._wy[i], rcond=None)[0]
            else:
                ye = self._wy[i]
            total_ss = float(np.squeeze(ye.T @ ye))
            stats = self._common_indiv_results(
                i,
                beta,
                cov,
                weps,
                eps,
                method,
                cov_type,
                cov_est,
                iter_count,
                debiased,
                cons,
                total_ss,
                weight_est=self._weight_est,
            )

            key = self._eq_labels[i]
            individual[key] = stats

        # Populate results dictionary
        nobs = eps.size
        results = self._common_results(
            beta, cov, method, iter_count, nobs, cov_type, sigma, individual, debiased
        )

        # wresid is different between GLS and OLS
        wresiduals = [
            individual[individual_key].wresid for individual_key in individual
        ]
        wresid = np.hstack(wresiduals)
        results["wresid"] = wresid
        results["wmat"] = wmat
        results["weight_type"] = self._weight_type
        results["weight_config"] = self._weight_est.config
        results["cov_estimator"] = cov_est
        results["cov_config"] = cov_est.cov_config
        results["weight_estimator"] = self._weight_est
        results["j_stat"] = self._j_statistic(beta, wmat)
        r2s = [individual[eq].r2 for eq in individual]
        results["system_r2"] = self._system_r2(eps, sigma, "gls", False, debiased, r2s)

        return GMMSystemResults(results)

    @classmethod
    def from_formula(
        cls,
        formula: str | dict[str, str],
        data: DataFrame,
        *,
        weights: dict[str, linearmodels.typing.data.ArrayLike] | None = None,
        weight_type: str = "robust",
        **weight_config: bool | str | float,
    ) -> IVSystemGMM:
        """
        Specify a 3SLS using the formula interface

        Parameters
        ----------
        formula : {str, dict-like}
            Either a string or a dictionary of strings where each value in
            the dictionary represents a single equation. See Notes for a
            description of the accepted syntax
        data : DataFrame
            Frame containing named variables
        weights : dict-like
            Dictionary like object (e.g. a DataFrame) containing variable
            weights.  Each entry must have the same number of observations as
            data.  If an equation label is not a key weights, the weights will
            be set to unity
        weight_type : str
            Name of moment condition weight function to use in the GMM
            estimation. Valid options are:

            * "unadjusted", "homoskedastic" - Assume moments are homoskedastic
            * "robust", "heteroskedastic" - Allow for heteroskedasticity

        weight_config
            Additional keyword arguments to pass to the moment condition weight
            function

        Returns
        -------
        model : IVSystemGMM
            Model instance

        Notes
        -----
        Models can be specified in one of two ways. The first uses curly
        braces to encapsulate equations.  The second uses a dictionary
        where each key is an equation name.

        Examples
        --------
        The simplest format uses standard formulas for each equation
        in a dictionary.  Best practice is to use an Ordered Dictionary

        >>> import pandas as pd
        >>> import numpy as np
        >>> cols = ["y1", "x1_1", "x1_2", "z1", "y2", "x2_1", "x2_2", "z2"]
        >>> data = pd.DataFrame(np.random.randn(500, 8), columns=cols)
        >>> from linearmodels.system import IVSystemGMM
        >>> formula = {"eq1": "y1 ~ 1 + x1_1 + [x1_2 ~ z1]",
        ...            "eq2": "y2 ~ 1 + x2_1 + [x2_2 ~ z2]"}
        >>> mod = IVSystemGMM.from_formula(formula, data)

        The second format uses curly braces {} to surround distinct equations

        >>> formula = "{y1 ~ 1 + x1_1 + [x1_2 ~ z1]} {y2 ~ 1 + x2_1 + [x2_2 ~ z2]}"
        >>> mod = IVSystemGMM.from_formula(formula, data)

        It is also possible to include equation labels when using curly braces

        >>> formula = "{eq1: y1 ~ x1_1 + [x1_2 ~ z1]} {eq2: y2 ~ 1 + [x2_2 ~ z2]}"
        >>> mod = IVSystemGMM.from_formula(formula, data)
        """
        context = capture_context(1)
        parser = SystemFormulaParser(formula, data, weights, context=context)
        eqns = parser.data
        mod = cls(eqns, sigma=None, weight_type=weight_type, **weight_config)
        mod.formula = formula
        return mod

    def _j_statistic(
        self,
        params: linearmodels.typing.data.Float64Array,
        weight_mat: linearmodels.typing.data.Float64Array,
    ) -> WaldTestStatistic:
        """
        J stat and test

        Parameters
        ----------
        params : ndarray
            Estimated model parameters
        weight_mat : ndarray
            Weighting matrix used in estimation of the parameters

        Returns
        -------
        stat : WaldTestStatistic
            Test statistic

        Notes
        -----
        Assumes that the efficient weighting matrix has been used.  Using
        other weighting matrices will not produce the correct test.
        """
        y, x, z = self._wy, self._wx, self._wz
        k = len(x)
        ze_lst = []
        idx = 0
        for i in range(k):
            kx = x[i].shape[1]
            beta = params[idx : idx + kx]
            eps = y[i] - x[i] @ beta
            ze_lst.append(z[i] * eps)
            idx += kx
        ze = np.concatenate(ze_lst, 1)
        g_bar = ze.mean(0)
        nobs = x[0].shape[0]
        stat = float(nobs * g_bar.T @ np.linalg.inv(weight_mat) @ g_bar.T)
        null = "Expected moment conditions are equal to 0"
        ninstr = sum(a.shape[1] for a in z)
        nvar = sum(a.shape[1] for a in x)
        ncons = 0 if self.constraints is None else self.constraints.r.shape[0]

        return WaldTestStatistic(stat, null, ninstr - (nvar - ncons))
