import pathlib
from typing import Any, Dict, TypeAlias, Union

import numpy as np

from imod.mf6.package import Package
from imod.mf6.write_context import WriteContext
from imod.schemata import AllValueSchema, DimsSchema, DTypeSchema
from imod.typing import GridDataset

_PeriodDataType: TypeAlias = dict[np.int64, list]
_PeriodDataVarNames: TypeAlias = tuple[str, str, str, str, str]


def _assign_data_to_perioddata(
    perioddata: _PeriodDataType,
    varnames: _PeriodDataVarNames,
    start: np.int64,
    data: GridDataset,
) -> None:
    """Assign data from a dataset to a perioddata. Modifies perioddata in place."""
    if start in perioddata:
        raise ValueError(f"Duplicate start time {start} in perioddata.")
    perioddata[start] = [data[var].values[()] for var in varnames]


class AdaptiveTimeStepping(Package):
    """
    Adaptive Time Stepping (ATS) Package for MODFLOW 6.

    This package allows for adaptive time stepping in the simulation, adjusting
    the time step size based on model convergence and stability criteria.

    Parameters
    ----------
    dt_init: xr.DataArray of floats
        Initial time step length, ``dt0`` in MODFLOW 6. If zero, then the final time
        step from the previous stress period will be used as the initial time
        step.
    dt_min: xr.DataArray of floats
        Minimum allowed time length size. This value must be greater than zero
        and less than dtmax. dtmin must be a small value in order to ensure that
        simulation times end at the end of stress periods and the end of the
        simulation. A small value, such as 1.e-5, is recommended.
    dt_max: xr.DataArray of floats
        Maximum allowed time step length. This value must be greater than dtmin.
    dt_multiplier: xr.DataArray of floats, default 0.0
        Multiplier for the time step length, ``dtadj`` in MODFLOW6. If the
        number of outer solver iterations are less than the product of the
        maximum number of outer iterations (outer_maximum) and
        ``ats_outer_maximum_fraction`` (an optional variable in
        :class:`imod.mf6.Solution`), then the time step length is multiplied by
        ``dt_multiplier``. If the number of outer solver iterations are greater
        than the product of the maximum number of outer iterations and 1.0 minus
        ``ats_outer_maximum_fraction``, then the time step length is divided by
        ``dt_multiplier``. ``dt_multiplier`` must be zero, one, or greater than
        one. If ``dt_multiplier`` is zero or one, then it has no effect on the
        simulation. A value between 2.0 and 5.0 can be used as an initial
        estimate.
    dt_fail_multiplier: xr.DataArray of floats, default 0.0
        Divisor of the time step length when a time step fails to converge. If
        there is solver failure, then the time step will be tried again with a
        shorter time step length calculated as the previous time step length
        divided by ``dt_fail_multiplier``. ``dt_fail_multiplier`` must be zero,
        one, or greater than one. If ``dt_fail_multiplier`` is zero or one, then
        time steps will not be retried with shorter lengths. In this case, the
        program will terminate with an error, or it will continue of the
        CONTINUE option is set in the simulation name file. Initial tests with
        this variable should be set to 5.0 or larger to determine if convergence
        can be achieved.
    validate: {True, False}
        Flag to indicate whether the package should be validated upon
        initialization. Defaults to True.
    """

    _pkg_id = "ats"
    _keyword_map = {}
    _template = Package._initialize_template(_pkg_id)

    _period_data: _PeriodDataVarNames = (
        "dt_init",
        "dt_min",
        "dt_max",
        "dt_multiplier",
        "dt_fail_multiplier",
    )
    _init_schemata = {
        "dt_init": [DimsSchema("time") | DimsSchema(), DTypeSchema(np.floating)],
        "dt_min": [DimsSchema("time") | DimsSchema(), DTypeSchema(np.floating)],
        "dt_max": [DimsSchema("time"), DTypeSchema(np.floating)],
        "dt_multiplier": [
            DimsSchema("time") | DimsSchema(),
            DTypeSchema(np.floating),
        ],
        "dt_fail_multiplier": [
            DimsSchema("time") | DimsSchema(),
            DTypeSchema(np.floating),
        ],
    }

    _write_schemata = {
        "dt_init": [AllValueSchema(">=", 0.0)],
        "dt_min": [AllValueSchema("<", "dt_max"), AllValueSchema(">", 0.0)],
        "dt_multiplier": [AllValueSchema("==", 0.0) | AllValueSchema(">=", 1.0)],
    }

    def __init__(
        self,
        dt_init,
        dt_min,
        dt_max,
        dt_multiplier=0.0,
        dt_fail_multiplier=0.0,
        validate=True,
    ):
        dict_dataset = {
            "dt_init": dt_init,
            "dt_min": dt_min,
            "dt_max": dt_max,
            "dt_multiplier": dt_multiplier,
            "dt_fail_multiplier": dt_fail_multiplier,
        }
        super().__init__(dict_dataset)
        self._validate_init_schemata(validate)

    def _get_render_dictionary(
        self,
        directory: pathlib.Path,
        pkgname: str,
        globaltimes: Union[list[np.datetime64], np.ndarray],
        binary: bool,
    ) -> Dict[str, Any]:
        perioddata: _PeriodDataType = {}
        # Force to np.int64 for mypy and numpy >= 2.2.4
        one = np.int64(1)
        if "time" in self.dataset:  # one of bin_ds has time
            package_times = self.dataset.coords["time"].values
            starts = np.searchsorted(globaltimes, package_times) + one
            for i, start in enumerate(starts):
                data = self.dataset.sel(time=package_times[i])
                _assign_data_to_perioddata(perioddata, self._period_data, start, data)
        else:
            _assign_data_to_perioddata(perioddata, self._period_data, one, self.dataset)

        d: dict[str, int | _PeriodDataType] = {}
        d["maxats"] = len(package_times)
        d["perioddata"] = perioddata
        return d

    def _validate(self, schemata: dict, **kwargs):
        # Insert additional kwargs
        kwargs["dt_max"] = self["dt_max"]
        errors = super()._validate(schemata, **kwargs)

        return errors

    def _write(
        self,
        pkgname: str,
        globaltimes: Union[list[np.datetime64], np.ndarray],
        write_context: WriteContext,
    ) -> None:
        ats_content = self._render(
            write_context.write_directory,
            pkgname,
            globaltimes,
            write_context.use_binary,
        )
        timedis_path = write_context.write_directory / f"{pkgname}.ats"
        with open(timedis_path, "w") as f:
            f.write(ats_content)
