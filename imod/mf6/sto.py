import abc
from copy import deepcopy
from typing import Optional

import numpy as np

from imod.logging import init_log_decorator
from imod.mf6.interfaces.iregridpackage import IRegridPackage
from imod.mf6.package import Package
from imod.mf6.regrid.regrid_schemes import (
    SpecificStorageRegridMethod,
    StorageCoefficientRegridMethod,
)
from imod.mf6.utilities.regrid import RegridderWeightsCache, _regrid_package_data
from imod.mf6.utilities.regridding_types import RegridderType
from imod.mf6.validation import PKG_DIMS_SCHEMA
from imod.schemata import (
    AllValueSchema,
    DimsSchema,
    DTypeSchema,
    IdentityNoDataSchema,
    IndexesSchema,
)
from imod.typing import GridDataArray
from imod.typing.grid import zeros_like


class Storage(Package):
    _pkg_id = "sto_deprecated"

    def __init__(*args, **kwargs):
        raise NotImplementedError(
            r"Storage package has been removed. Use SpecificStorage or StorageCoefficient instead."
        )


class StorageBase(Package, IRegridPackage, abc.ABC):
    def get_options(self, d):
        # Skip both variables in grid_data and "transient".
        not_options = list(self._grid_data.keys())
        not_options += "transient"

        for varname in self.dataset.data_vars.keys():  # pylint:disable=no-member
            if varname in not_options:
                continue
            v = self.dataset[varname].values[()]
            if self._valid(v):  # skip None and False
                d[varname] = v
        return d

    def _render_dict(self, directory, pkgname, globaltimes, binary):
        d = {}
        stodirectory = directory / pkgname
        for varname in self._grid_data:
            key = self._keyword_map.get(varname, varname)
            layered, value = self._compose_values(
                self[varname], stodirectory, key, binary=binary
            )
            if self._valid(value):  # skip False or None
                d[f"{key}_layered"], d[key] = layered, value

        periods = {}
        if "time" in self.dataset["transient"].coords:
            package_times = self.dataset["transient"].coords["time"].values
            starts = np.searchsorted(globaltimes, package_times) + 1
            for i, s in enumerate(starts):
                periods[s] = self.dataset["transient"].isel(time=i).values[()]
        else:
            periods[1] = self.dataset["transient"].values[()]

        d["periods"] = periods

        d = self.get_options(d)

        return d


class SpecificStorage(StorageBase):
    """
    Storage Package with specific storage.

    From wikipedia (https://en.wikipedia.org/wiki/Specific_storage):

    "The specific storage is the amount of water that a portion of an aquifer
    releases from storage, per unit mass or volume of aquifer, per unit change
    in hydraulic head, while remaining fully saturated."

    If the STO Package is not included for a model, then storage changes will
    not be calculated, and thus, the model will be steady state. Only one STO
    Package can be specified for a GWF model.

    Parameters
    ----------
    specific_storage: array of floats (xr.DataArray)
        Is specific storage. Specific storage values must be greater than
        or equal to 0. (ss)
    specific_yield: array of floats (xr.DataArray)
        Is specific yield. Specific yield values must be greater than or
        equal to 0. Specific yield does not have to be specified if there are no
        convertible cells (convertible=0 in every cell). (sy)
    transient: ({True, False}), or a DataArray with a time coordinate and dtype Bool
        Boolean to indicate if the model is transient or steady-state.
    convertible: array of int (xr.DataArray)
        Is a flag for each cell that specifies whether or not a cell is
        convertible for the storage calculation. 0 indicates confined storage is
        used. >0 indicates confined storage is used when head is above cell top
        and a mixed formulation of unconfined and confined storage is used when
        head is below cell top. (iconvert)
    save_flows: ({True, False}, optional)
        Indicates that storage flow terms will be written to the file specified
        with "BUDGET FILEOUT" in Output Control. Default is False.
    validate: {True, False}
        Flag to indicate whether the package should be validated upon
        initialization. This raises a ValidationError if package input is
        provided in the wrong manner. Defaults to True.
    """

    _pkg_id = "sto"
    _grid_data = {
        "convertible": np.int32,
        "specific_storage": np.float64,
        "specific_yield": np.float64,
    }
    _keyword_map = {
        "specific_storage": "ss",
        "specific_yield": "sy",
        "convertible": "iconvert",
    }

    _init_schemata = {
        "specific_storage": (
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
        ),
        "specific_yield": (
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
        ),
        "transient": (
            DTypeSchema(np.bool_),
            IndexesSchema(),
            DimsSchema("time") | DimsSchema(),
        ),
        "convertible": (
            DTypeSchema(np.integer),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
        ),
        "save_flows": (DTypeSchema(np.bool_), DimsSchema()),
    }

    _write_schemata = {
        "specific_storage": (
            AllValueSchema(">=", 0.0),
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
            # No need to check coords: dataset ensures they align with idomain.
        ),
        "specific_yield": (
            AllValueSchema(">=", 0.0),
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ),
        "convertible": (
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ),
    }

    _template = Package._initialize_template(_pkg_id)
    _regrid_method = SpecificStorageRegridMethod()

    @init_log_decorator()
    def __init__(
        self,
        specific_storage,
        specific_yield,
        transient,
        convertible,
        save_flows: bool = False,
        validate: bool = True,
    ):
        dict_dataset = {
            "specific_storage": specific_storage,
            "specific_yield": specific_yield,
            "convertible": convertible,
            "transient": transient,
            "save_flows": save_flows,
        }
        super().__init__(dict_dataset)
        self._validate_init_schemata(validate)

    def render(self, directory, pkgname, globaltimes, binary):
        d = self._render_dict(directory, pkgname, globaltimes, binary)
        return self._template.render(d)


class StorageCoefficient(StorageBase):
    """
    Storage Package with a storage coefficient.  Be careful,
    this is not the same as the specific storage.

    From wikipedia (https://en.wikipedia.org/wiki/Specific_storage):

    "Storativity or the storage coefficient is the volume of water released
    from storage per unit decline in hydraulic head in the aquifer, per
    unit area of the aquifer.  Storativity is a dimensionless quantity, and
    is always greater than 0.

    Under confined conditions:

    S = Ss * b, where S is the storage coefficient,
    Ss the specific storage, and b the aquifer thickness.

    Under unconfined conditions:

    S = Sy, where Sy is the specific yield"

    If the STO Package is not included for a model, then storage changes will
    not be calculated, and thus, the model will be steady state. Only one STO
    Package can be specified for a GWF model.

    Parameters
    ----------
    storage_coefficient: array of floats (xr.DataArray)
        Is storage coefficient. Storage coefficient values must be greater than
        or equal to 0. (ss)
    specific_yield: array of floats (xr.DataArray)
        Is specific yield. Specific yield values must be greater than or
        equal to 0. Specific yield does not have to be specified if there are no
        convertible cells (convertible=0 in every cell). (sy)
    transient: ({True, False})
        Boolean to indicate if the model is transient or steady-state.
    convertible: array of int (xr.DataArray)
        Is a flag for each cell that specifies whether or not a cell is
        convertible for the storage calculation. 0 indicates confined storage is
        used. >0 indicates confined storage is used when head is above cell top
        and a mixed formulation of unconfined and confined storage is used when
        head is below cell top. (iconvert)
    save_flows: ({True, False}, optional)
        Indicates that storage flow terms will be written to the file specified
        with "BUDGET FILEOUT" in Output Control. Default is False.
    validate: {True, False}
        Flag to indicate whether the package should be validated upon
        initialization. This raises a ValidationError if package input is
        provided in the wrong manner. Defaults to True.
    """

    _pkg_id = "sto"
    _grid_data = {
        "convertible": np.int32,
        "storage_coefficient": np.float64,
        "specific_yield": np.float64,
    }
    _keyword_map = {
        "storage_coefficient": "ss",
        "specific_yield": "sy",
        "convertible": "iconvert",
    }

    _init_schemata = {
        "storage_coefficient": (
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
        ),
        "specific_yield": (
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
        ),
        "transient": (
            DTypeSchema(np.bool_),
            IndexesSchema(),
            DimsSchema("time") | DimsSchema(),
        ),
        "convertible": (
            DTypeSchema(np.integer),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
        ),
        "save_flows": (DTypeSchema(np.bool_), DimsSchema()),
    }

    _write_schemata = {
        "storage_coefficient": (
            AllValueSchema(">=", 0.0),
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ),
        "specific_yield": (
            AllValueSchema(">=", 0.0),
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ),
        "convertible": (
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
            # No need to check coords: dataset ensures they align with idomain.
        ),
    }

    _template = Package._initialize_template(_pkg_id)
    _regrid_method = StorageCoefficientRegridMethod()

    @init_log_decorator()
    def __init__(
        self,
        storage_coefficient,
        specific_yield,
        transient,
        convertible,
        save_flows: bool = False,
        validate: bool = True,
    ):
        dict_dataset = {
            "storage_coefficient": storage_coefficient,
            "specific_yield": specific_yield,
            "convertible": convertible,
            "transient": transient,
            "save_flows": save_flows,
        }
        super().__init__(dict_dataset)
        self._validate_init_schemata(validate)

    def render(self, directory, pkgname, globaltimes, binary):
        d = self._render_dict(directory, pkgname, globaltimes, binary)
        d["storagecoefficient"] = True
        return self._template.render(d)

    @classmethod
    def from_imod5_data(
        cls,
        imod5_data: dict[str, dict[str, GridDataArray]],
        target_grid: GridDataArray,
        regridder_types: Optional[dict[str, tuple[RegridderType, str]]] = None,
    ) -> "StorageCoefficient":
        """
        Construct a StorageCoefficient-package from iMOD5 data, loaded with the
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
        Modflow 6 StorageCoefficient package. Its specific yield is 0 and it's transient if any storage_coefficient
             is larger than 0. All cells are set to inconvertible (they stay confined throughout the simulation)
        """

        data = {
            "storage_coefficient": imod5_data["sto"]["storage_coefficient"],
        }

        regridder_settings = deepcopy(cls._regrid_method)
        if regridder_types is not None:
            regridder_settings.update(regridder_types)

        regrid_context = RegridderWeightsCache()

        new_package_data = _regrid_package_data(
            data, target_grid, regridder_settings, regrid_context, {}
        )

        new_package_data["convertible"] = zeros_like(
            new_package_data["storage_coefficient"], dtype=int
        )
        new_package_data["transient"] = np.any(
            new_package_data["storage_coefficient"].values > 0
        )
        new_package_data["specific_yield"] = None

        return cls(**new_package_data)
