from imod.fixed_format import VariableMetaData
from imod.msw.pkgbase import Package


class LanduseOptions(Package):
    """
    Land use options. This object is responsible for luse_svat.inp

    Parameters
    ----------
    landuse_name: array of strings (xr.DataArray)
        Names of land use
    vegetation_index: array of integers (xr.DataArray)
        Vegetation indices
    jarvis_o2_stress: array of floats (xr.DataArray)
        Jarvis parameter for oxygen stress
    jarvis_drought_stress: array of floats (xr.DataArray)
        Jarvis parameter for drought stress
    feddes_p1: array of floats (xr.DataArray)
        p1 (m) in Feddes function for transpiration reduction
    feddes_p2: array of floats (xr.DataArray)
        p2 (m) in Feddes function for transpiration reduction
    feddes_p3h: array of floats (xr.DataArray)
        p3h (m) in Feddes function for transpiration reduction
    feddes_p3l: array of floats (xr.DataArray)
        p3l (m) in Feddes function for transpiration reduction
    feddes_p4: array of floats (xr.DataArray)
        p4 (m) in Feddes function for transpiration reduction
    feddes_t3h: array of floats (xr.DataArray)
        t3h (mm/d) in Feddes function for transpiration reduction
    feddes_t3l: array of floats (xr.DataArray)
        t3l (mm/d) in Feddes function for transpiration reduction
    threshold_sprinkling: array of floats (xr.DataArray)
        If <0, pressure head (m) at which sprinkling begins. If >0 drought
        stress at which sprinkling begins.
    fraction_evaporated_sprinkling: array of floats (xr.DataArray)
        Fraction evaporated sprinkling water
    gift: array of floats (xr.DataArray)
        Gift (mm) during rotational period
    gift_duration: array of floats (xr.DataArray)
        Gift duration (d)
    rotational_period: array of floats (xr.DataArray)
        Rotational period (d)
    start_sprinkling_season: array of floats (xr.DataArray)
        Day of year at which sprinkling season starts (d)
    end_sprinkling_season: array of floats (xr.DataArray)
        Day of year at which sprinkling season ends (d)
    interception_option: array of integers (xr.DataAray)
        Choose interception model. 0=Rutter, 1=Von Hoyningen. NOTE: option
        2=GASH, but this is not supported by MetaSWAP v8.1.0.3 and lower
    interception_capacity_per_LAI: array of floats (xr.DataArray)
        interception capacity (mm/LAI) # Rutter and Von Hoyningen the same???
    interception_intercept: array of floats (xr.DataArray)
        intercept of the interception evaporation curve. Pun unintended.

    Notes
    -----
    No Penman-Monteith is supported in iMOD Python, so albedo, rsc, rsw, rsoil,
    kdif, and kdir cannot be specified. (We might create a seperate object for
    this if there is a demand for it.)

    The GASH model (interception_option = 2) and salt stress parameters Maas &
    Hoffman are not supported by MetaSWAP at the time of writing this class. So
    these are not supported.
    """

    _metadata_dict = {
        "landuse_index": VariableMetaData(6, 1, 999, int),
        "landuse_name": VariableMetaData(20, None, None, str),
        "vegetation_index": VariableMetaData(6, 0.0, 1e6, int),
        # Jarvis stress
        # Columns 33-35 and 36-38, but both F6?
        "jarvis_o2_stress": VariableMetaData(3, 0.0, 1e6, float),
        "jarvis_drought_stress": VariableMetaData(3, 0.0, 1e6, float),
        # Feddes transpiration function
        "feddes_p1": VariableMetaData(8, -160.0, 100.0, float),
        "feddes_p2": VariableMetaData(8, -160.0, 100.0, float),
        "feddes_p3h": VariableMetaData(8, -160.0, 0.0, float),
        "feddes_p3l": VariableMetaData(8, -160.0, 0.0, float),
        "feddes_p4": VariableMetaData(8, -160.0, 0.0, float),
        "feddes_t3h": VariableMetaData(8, 0.1, 10.0, float),
        "feddes_t3l": VariableMetaData(8, 0.1, 10.0, float),
        # Sprinkling
        "threshold_sprinkling": VariableMetaData(8, -160.0, 1.0, float),
        "fraction_evaporated_sprinkling": VariableMetaData(8, 0.0, 1.0, float),
        "gift": VariableMetaData(8, 1.0, 1000.0, float),
        "gift_duration": VariableMetaData(8, 0.01, 1000.0, float),
        "rotational_period": VariableMetaData(6, 1.0, 366.0, float),
        "start_sprinkling_season": VariableMetaData(6, 0.0, 366.0, float),
        "end_sprinkling_season": VariableMetaData(6, 0.0, 366.0, float),
        # Penman-Monteith: not supported
        "albedo": VariableMetaData(8, None, None, str),
        "rsc": VariableMetaData(8, None, None, str),
        "rsw": VariableMetaData(8, None, None, str),
        "rsoil": VariableMetaData(8, None, None, str),
        "kdif": VariableMetaData(8, None, None, str),
        "kdir": VariableMetaData(8, None, None, str),
        # Interception
        "interception_option": VariableMetaData(6, 0, 2, int),
        "interception_capacity_per_LAI_Rutter": VariableMetaData(8, 0.0, 10.0, float),
        "interception_intercept": VariableMetaData(8, 0.0, 1.0, float),
        "interception_capacity_per_LAI_VonHoyningen": VariableMetaData(
            8, 0.0, 10.0, float
        ),
        # Gash interception: not supported
        "pfree": VariableMetaData(8, None, None, str),
        "pstem": VariableMetaData(8, None, None, str),
        "scanopy": VariableMetaData(8, None, None, str),
        "avprec": VariableMetaData(8, None, None, str),
        "avevap": VariableMetaData(8, None, None, str),
        # Maas-Hoffman: not supported
        "saltmax": VariableMetaData(8, None, None, str),
        "saltslope": VariableMetaData(8, None, None, str),
    }

    _file_name = "luse_svat.inp"

    def __init__(
        self,
        landuse_name,
        vegetation_index,
        jarvis_o2_stress,
        jarvis_drought_stress,
        feddes_p1,
        feddes_p2,
        feddes_p3h,
        feddes_p3l,
        feddes_p4,
        feddes_t3h,
        feddes_t3l,
        threshold_sprinkling,
        fraction_evaporated_sprinkling,
        gift,
        gift_duration,
        rotational_period,
        start_sprinkling_season,
        end_sprinkling_season,
        interception_option,
        interception_capacity_per_LAI,
        interception_intercept,
    ):
        super().__init__()
        self.dataset["landuse_name"] = landuse_name
        self.dataset["vegetation_index"] = vegetation_index
        self.dataset["jarvis_o2_stress"] = jarvis_o2_stress
        self.dataset["jarvis_drought_stress"] = jarvis_drought_stress
        self.dataset["feddes_p1"] = feddes_p1
        self.dataset["feddes_p2"] = feddes_p2
        self.dataset["feddes_p3h"] = feddes_p3h
        self.dataset["feddes_p3l"] = feddes_p3l
        self.dataset["feddes_p4"] = feddes_p4
        self.dataset["feddes_t3h"] = feddes_t3h
        self.dataset["feddes_t3l"] = feddes_t3l
        self.dataset["threshold_sprinkling"] = threshold_sprinkling
        self.dataset["fraction_evaporated_sprinkling"] = fraction_evaporated_sprinkling
        self.dataset["gift"] = gift
        self.dataset["gift_duration"] = gift_duration
        self.dataset["rotational_period"] = rotational_period
        self.dataset["start_sprinkling_season"] = start_sprinkling_season
        self.dataset["end_sprinkling_season"] = end_sprinkling_season
        self.dataset["interception_option"] = interception_option
        self.dataset[
            "interception_capacity_per_LAI_Rutter"
        ] = interception_capacity_per_LAI
        self.dataset[
            "interception_capacity_per_LAI_VonHoyningen"
        ] = interception_capacity_per_LAI
        self.dataset["interception_intercept"] = interception_intercept

        self._pkgcheck()

    def _render(self, file, *args):
        dataframe = self.dataset.to_dataframe(
            dim_order=("landuse_index",)
        ).reset_index()

        self._check_range(dataframe)

        # Find missing columns
        missing_keys = set(self._metadata_dict.keys()) ^ set(dataframe.columns)

        # Add missing columns
        for key in missing_keys:
            dataframe[key] = ""

        # Reorder columns to _metadata_dict order
        dataframe = dataframe[list(self._metadata_dict.keys())]

        return self.write_dataframe_fixed_width(file, dataframe)

    def _pkgcheck(self):
        dims = self.dataset.dims
        dims_expected = ("landuse_index",)
        if len(set(dims) - set(dims_expected)) > 0:
            raise ValueError(
                f"Please provide DataArrays with dimensions {dims_expected}"
            )
