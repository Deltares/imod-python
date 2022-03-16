import pathlib
from tkinter import Variable

import numpy as np
from sklearn.metrics import average_precision_score
import xarray as xr

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
    start_sprinkling_seasion: array of floats (xr.DataArray)
        Day of year at which sprinkling season starts (d)
    end_sprinkling_seasion: array of floats (xr.DataArray)
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
        "landuse_name": VariableMetaData(19, None, None, str),
        "vegetation_index": VariableMetaData(6, 0.0, 1e6, int),
        # Jarvis stress
        "jarvis_o2_stress": VariableMetaData(6, 0.0, 1e6, float),
        "jarvis_drought_stress": VariableMetaData(6, 0.0, 1e6, float),
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
        "start_sprinkling_seasion": VariableMetaData(6, 0.0, 366.0, float),
        "end_sprinkling_seasion": VariableMetaData(6, 0.0, 366.0, float),
        # Penman-Monteith: not supported
        "albedo": VariableMetaData(8, None, None, str),
        "rsc": VariableMetaData(8, None, None, str),
        "rsw": VariableMetaData(8, None, None, str),
        "rsoil": VariableMetaData(8, None, None, str),
        "kdif": VariableMetaData(8, None, None, str),
        "kdir": VariableMetaData(8, None, None, str),
        # Interception
        "interception_option": VariableMetaData(6, 0, 2, int),
        "interception_capacity_per_LAI": VariableMetaData(8, 0.0, 10.0, float),
        "interception_intercept": VariableMetaData(8, 0.0, 1.0, float),
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

    def __init__(self):
        pass
