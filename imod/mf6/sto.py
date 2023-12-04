import abc

import numpy as np

from imod.mf6.package import Package
from imod.mf6.regridding_utils import RegridderType
from imod.mf6.validation import PKG_DIMS_SCHEMA
from imod.schemata import (
    AllValueSchema,
    DimsSchema,
    DTypeSchema,
    IdentityNoDataSchema,
    IndexesSchema,
)


class Storage(Package):
    _pkg_id = "sto_deprecated"

    def __init__(*args, **kwargs):
        raise NotImplementedError(
            r"Storage package has been removed. Use SpecificStorage or StorageCoefficient instead."
        )


class StorageBase(Package, abc.ABC):
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

    _regrid_method = {
        "convertible": (RegridderType.OVERLAP, "mode"),
        "specific_storage": (RegridderType.OVERLAP, "mean"),
        "specific_yield": (RegridderType.OVERLAP, "mean"),
    }

    _template = Package._initialize_template(_pkg_id)

    def __init__(
        self,
        specific_storage,
        specific_yield,
        transient,
        convertible,
        save_flows: bool = False,
        validate: bool = True,
    ):
        super().__init__(locals())
        self.dataset["specific_storage"] = specific_storage
        self.dataset["specific_yield"] = specific_yield
        self.dataset["convertible"] = convertible
        self.dataset["transient"] = transient
        self.dataset["save_flows"] = save_flows
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

    _regrid_method = {
        "convertible": (RegridderType.OVERLAP, "mode"),
        "storage_coefficient": (RegridderType.OVERLAP, "mean"),
        "specific_yield": (RegridderType.OVERLAP, "mean"),
    }

    _template = Package._initialize_template(_pkg_id)

    def __init__(
        self,
        storage_coefficient,
        specific_yield,
        transient,
        convertible,
        save_flows: bool = False,
        validate: bool = True,
    ):
        super().__init__(locals())
        self.dataset["storage_coefficient"] = storage_coefficient
        self.dataset["specific_yield"] = specific_yield
        self.dataset["convertible"] = convertible
        self.dataset["transient"] = transient
        self.dataset["save_flows"] = save_flows
        self._validate_init_schemata(validate)

    def render(self, directory, pkgname, globaltimes, binary):
        d = self._render_dict(directory, pkgname, globaltimes, binary)
        d["sy_present"] = self.dataset["specific_yield"].values[()] is not None
        d["storagecoefficient"] = True
        return self._template.render(d)
