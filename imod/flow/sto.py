from imod.flow.pkgbase import Package


class StorageCoefficient(Package):
    """
    Storage coefficient [-].  Be careful, this is not the same as the specific
    storage.

    From wikipedia (https://en.wikipedia.org/wiki/Specific_storage):

    Storativity or the storage coefficient is the volume of water released
    from storage per unit decline in hydraulic head in the aquifer, per
    unit area of the aquifer.  Storativity is a dimensionless quantity, and
    is always greater than 0.

    Under confined conditions:

    S = Ss * b, where S is the storage coefficient,
    Ss the specific storage, and b the aquifer thickness.

    Under unconfined conditions:

    S = Sy, where Sy is the specific yield

    Parameters
    ----------
    storage_coefficient : float or xr.DataArray
        Storage coefficient, dims = ("layer", "y", "x").

    """

    _pkg_id = "sto"
    _variable_order = ["storage_coefficient"]

    def __init__(self, storage_coefficient):
        super().__init__()
        self.dataset["storage_coefficient"] = storage_coefficient


class SpecificStorage(Package):
    """
    Specific storage [L-1].  Be careful, this is not the same as the storage
    coefficient.

    From wikipedia (https://en.wikipedia.org/wiki/Specific_storage):

    The specific storage is the amount of water that a portion of an aquifer
    releases from storage, per unit mass or volume of aquifer, per unit change
    in hydraulic head, while remaining fully saturated.

    Parameters
    ----------
    specific_storage : float or xr.DataArray
        Specific storage, dims ``("layer", "y", "x")``.
    """

    _pkg_id = "ssc"
    _variable_order = ["specific_storage"]

    def __init__(self, specific_storage):
        super().__init__()
        self.dataset["specific_storage"] = specific_storage
