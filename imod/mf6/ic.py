import warnings
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np

from imod.logging import init_log_decorator
from imod.mf6.interfaces.iregridpackage import IRegridPackage
from imod.mf6.package import Package
from imod.mf6.utilities.regrid import (
    RegridderType,
    RegridderWeightsCache,
    _regrid_package_data,
)
from imod.mf6.validation import PKG_DIMS_SCHEMA
from imod.schemata import DTypeSchema, IdentityNoDataSchema, IndexesSchema
from imod.typing import GridDataArray


class InitialConditions(Package, IRegridPackage):
    """
    Initial Conditions (IC) Package information is read from the file that is
    specified by "IC6" as the file type. Only one IC Package can be specified
    for a GWF model.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=46

    Parameters
    ----------
    head: array of floats (xr.DataArray)
        for backwards compatibility this argument is maintained, but please use
        the start-argument instead.
    start: array of floats (xr.DataArray)
        is the initial (starting) head or concentration—that is, the simulation's
        initial state.
        STRT must be specified for all simulations, including steady-state simulations.
        One value is read for every model cell. For
        simulations in which the first stress period is steady state, the values
        used for STRT generally do not affect the simulation (exceptions may
        occur if cells go dry and (or) rewet). The execution time, however, will
        be less if STRT includes hydraulic heads that are close to the
        steadystate solution. A head value lower than the cell bottom can be
        provided if a cell should start as dry. (strt)
    validate: {True, False}
        Flag to indicate whether the package should be validated upon
        initialization. This raises a ValidationError if package input is
        provided in the wrong manner. Defaults to True.
    """

    _pkg_id = "ic"
    _grid_data = {"head": np.float64}
    _keyword_map = {"head": "strt"}
    _template = Package._initialize_template(_pkg_id)

    _init_schemata = {
        "start": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
        ],
    }
    _write_schemata = {
        "start": [
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ],
    }

    _grid_data = {"start": np.float64}
    _keyword_map = {"start": "strt"}
    _template = Package._initialize_template(_pkg_id)

    _regrid_method = {
        "start": (
            RegridderType.OVERLAP,
            "mean",
        ),  # TODO set to barycentric once supported
    }

    @init_log_decorator()
    def __init__(self, start=None, head=None, validate: bool = True):
        if start is None:
            start = head
            warnings.warn(
                'The keyword argument "head" is deprecated. Please use the start argument.',
                DeprecationWarning,
            )
            if head is None:
                raise ValueError("start and head arguments cannot both be None")
        else:
            if head is not None:
                raise ValueError("start and head arguments cannot both be defined")

        dict_dataset = {"start": start}
        super().__init__(dict_dataset)
        self._validate_init_schemata(validate)

    def render(self, directory, pkgname, globaltimes, binary):
        d = {}

        icdirectory = directory / pkgname
        d["layered"], d["strt"] = self._compose_values(
            self["start"], icdirectory, "strt", binary=binary
        )
        return self._template.render(d)

    def get_regrid_methods(self) -> Optional[dict[str, Tuple[RegridderType, str]]]:
        return self._regrid_method

    @classmethod
    def from_imod5_data(
        cls,
        imod5_data: dict[str, dict[str, GridDataArray]],
        target_grid: GridDataArray,
        regridder_types: Optional[dict[str, tuple[RegridderType, str]]] = None,
    ) -> "InitialConditions":
        """
        Construct a SpecificStorage-package from iMOD5 data, loaded with the
        :func:`imod.formats.prj.open_projectfile_data` function.

        .. note::

            The method expects the iMOD5 model to be fully 3D, not quasi-3D.

        Parameters
        ----------
        imod5_data: dict
            Dictionary with iMOD5 data. This can be constructed from the
            :func:`imod.formats.prj.open_projectfile_data` method.
        target_grid: GridDataArray
            The grid that should be used for the new package. Does not
            need to be identical to one of the input grids.
        regridder_types: dict, optional
            Optional dictionary with regridder types for a specific variable.
            Use this to override default regridding methods.

        Returns
        -------
        Modflow 6 SpecificStorage package. Its specific yield is 0 and it's transient if any specific_storage
             is larger than 0. All cells are set to inconvertible (they stay confined throughout the simulation)
        """

        data = {
            "start": imod5_data["shd"]["head"],
        }

        regridder_settings = deepcopy(cls._regrid_method)
        if regridder_types is not None:
            regridder_settings.update(regridder_types)

        regrid_context = RegridderWeightsCache()

        new_package_data = _regrid_package_data(
            data, target_grid, regridder_settings, regrid_context, {}
        )
        return InitialConditions(**new_package_data)
