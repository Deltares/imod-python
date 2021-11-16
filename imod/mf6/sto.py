import numpy as np

from imod.mf6.pkgbase import Package


class Storage(Package):
    __slots__ = ()  # to quell FutureWarning

    def __init__(*args, **kwargs):
        raise DeprecationWarning(
            r"Storage package has been deprecated. Use SpecificStorage or StorageCoefficient instead."
        )


class SpecificStorage(Package):
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

    __slots__ = ("specific_storage", "specific_yield", "convertible", "transient")
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
    _template = Package._initialize_template(_pkg_id)

    def __init__(self, specific_storage, specific_yield, transient, convertible):
        super().__init__()
        self["specific_storage"] = specific_storage
        self["specific_yield"] = specific_yield
        self["convertible"] = convertible
        self["transient"] = transient

    def render(self, directory, pkgname, globaltimes, binary):
        d = {}
        stodirectory = directory / "sto"
        for varname in ["specific_storage", "specific_yield", "convertible"]:
            key = self._keyword_map.get(varname, varname)
            layered, value = self._compose_values(
                self[varname], stodirectory, key, binary=binary
            )
            if self._valid(value):  # skip False or None
                d[f"{key}_layered"], d[key] = layered, value

        periods = {}
        if "time" in self["transient"].coords:
            package_times = self["transient"].coords["time"].values
            starts = np.searchsorted(globaltimes, package_times) + 1
            for i, s in enumerate(starts):
                periods[s] = self["transient"].isel(time=i).values[()]
        else:
            periods[1] = self["transient"].values[()]

        d["periods"] = periods

        return self._template.render(d)


class StorageCoefficient(Package):
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

    __slots__ = ("storage_coefficient", "specific_yield", "convertible", "transient")
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
    _template = Package._initialize_template(_pkg_id)

    def __init__(self, storage_coefficient, specific_yield, transient, convertible):
        super().__init__()
        self["storage_coefficient"] = storage_coefficient
        self["specific_yield"] = specific_yield
        self["convertible"] = convertible
        self["transient"] = transient

    def render(self, directory, pkgname, globaltimes, binary):
        d = {}
        stodirectory = directory / "sto"
        for varname in ["storage_coefficient", "specific_yield", "convertible"]:
            key = self._keyword_map.get(varname, varname)
            layered, value = self._compose_values(
                self[varname], stodirectory, key, binary=binary
            )
            if self._valid(value):  # skip False or None
                d[f"{key}_layered"], d[key] = layered, value

        periods = {}
        if "time" in self["transient"].coords:
            package_times = self["transient"].coords["time"].values
            starts = np.searchsorted(globaltimes, package_times) + 1
            for i, s in enumerate(starts):
                periods[s] = self["transient"].isel(time=i).values[()]
        else:
            periods[1] = self["transient"].values[()]

        d["periods"] = periods
        d["storagecoefficient"] = True

        return self._template.render(d)
