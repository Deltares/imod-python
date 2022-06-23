from pathlib import Path
from typing import Tuple

import numpy as np
import xarray as xr

from imod.mf6.pkgbase import Package, VariableMetaData

from .read_input import read_sto_blockfile, shape_to_max_rows


class Storage(Package):
    def __init__(*args, **kwargs):
        raise NotImplementedError(
            r"Storage package has been removed. Use SpecificStorage or StorageCoefficient instead."
        )


class StorageBase(Package):
    def _render(self, directory, pkgname, globaltimes, binary):
        d = {}
        stodirectory = directory / pkgname
        for varname in self._grid_data.keys():
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
        return d

    @staticmethod
    def open(
        path: Path,
        simroot: Path,
        shape: Tuple[int],
        coords: Tuple[str],
        dims: Tuple[str],
        globaltimes: np.ndarray,
    ):
        sections = {
            "iconvert": (np.int32, shape_to_max_rows),
            "ss": (np.float64, shape_to_max_rows),
            "sy": (np.float64, shape_to_max_rows),
        }
        content = read_sto_blockfile(
            path=simroot / path,
            simroot=simroot,
            sections=sections,
            shape=shape,
        )

        griddata = content.pop("griddata")
        for field, data in griddata.items():
            content[field] = xr.DataArray(data, coords, dims)
        periods = content.pop("periods")
        content["transient"] = xr.DataArray(
            list(periods.values()),
            coords={"time": globaltimes[list(periods.keys())]},
            dims=("time",),
        )

        storagecoefficient = content.pop("storagecoefficient", False)
        if storagecoefficient:
            cls = StorageCoefficient
        else:
            cls = SpecificStorage

        filtered_content = cls.filter_and_rename(content)
        return cls(**filtered_content)


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
    convertible: array of int (xr.DataArray)
        Is a flag for each cell that specifies whether or not a cell is
        convertible for the storage calculation. 0 indicates confined storage is
        used. >0 indicates confined storage is used when head is above cell top
        and a mixed formulation of unconfined and confined storage is used when
        head is below cell top. (iconvert)
    transient: ({True, False})
        Boolean to indicate if the model is transient or steady-state.
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
    _metadata_dict = {
        "specific_storage": VariableMetaData(np.floating),
        "specific_yield": VariableMetaData(np.floating),
        "convertible": VariableMetaData(np.integer),
    }
    _template = Package._initialize_template(_pkg_id)

    def __init__(self, specific_storage, specific_yield, transient, convertible):
        super().__init__(locals())
        self.dataset["specific_storage"] = specific_storage
        self.dataset["specific_yield"] = specific_yield
        self.dataset["convertible"] = convertible
        self.dataset["transient"] = transient

        self._pkgcheck()

    def render(self, directory, pkgname, globaltimes, binary):
        d = self._render(directory, pkgname, globaltimes, binary)
        return self._template.render(d)


class StorageCoefficient(StorageBase):
    """
    Storage Package with a storage coefficient. Be careful, this is not the
    same as the specific storage.

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
        Is specific storage. Storage coefficient values must be greater than
        or equal to 0. (ss)
    specific_yield: array of floats (xr.DataArray)
        Is specific yield. Specific yield values must be greater than or
        equal to 0. Specific yield does not have to be specified if there are no
        convertible cells (convertible=0 in every cell). (sy)
    convertible: array of int (xr.DataArray)
        Is a flag for each cell that specifies whether or not a cell is
        convertible for the storage calculation. 0 indicates confined storage is
        used. >0 indicates confined storage is used when head is above cell top
        and a mixed formulation of unconfined and confined storage is used when
        head is below cell top. (iconvert)
    transient: ({True, False})
        Boolean to indicate if the model is transient or steady-state.
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
    _metadata_dict = {
        "storage_coefficient": VariableMetaData(np.floating),
        "specific_yield": VariableMetaData(np.floating),
        "convertible": VariableMetaData(
            np.integer,
        ),
    }
    _template = Package._initialize_template(_pkg_id)

    def __init__(self, storage_coefficient, specific_yield, transient, convertible):
        super().__init__(locals())
        self.dataset["storage_coefficient"] = storage_coefficient
        self.dataset["specific_yield"] = specific_yield
        self.dataset["convertible"] = convertible
        self.dataset["transient"] = transient

        self._pkgcheck()

    def render(self, directory, pkgname, globaltimes, binary):
        d = self._render(directory, pkgname, globaltimes, binary)
        d["storagecoefficient"] = True
        return self._template.render(d)
