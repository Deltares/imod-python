from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from imod.msw import (
    AnnualCropFactors,
    CouplerMapping,
    EvapotranspirationMapping,
    FileCopier,
    GridData,
    IdfMapping,
    Infiltration,
    InitialConditionsEquilibrium,
    InitialConditionsPercolation,
    InitialConditionsRootzonePressureHead,
    InitialConditionsSavedState,
    LanduseOptions,
    MeteoGrid,
    MeteoGridCopy,
    Ponding,
    PrecipitationMapping,
    ScalingFactors,
    SprinklingGrid,
    SprinklingPoints,
    TimeOutputControl,
    VariableOutputControl,
)


def get_grid_da(dtype, value=1, subunit=True):
    """
    This function creates a dataarray with scalar values for a grid of 2 subunits and 9 rows and columns.
    """
    shape = nsub, nrow, ncol = 2, 9, 9
    dims = ("subunit", "y", "x")

    dx = 10.0
    dy = -10.0
    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = abs(dy) * nrow

    subunits = np.arange(0, nsub)
    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"subunit": subunits, "y": y, "x": x}

    values = np.full(shape, fill_value=value, dtype=dtype)

    da = xr.DataArray(values, coords=coords, dims=dims)
    if subunit is False:
        da = da.sel(subunit=0, drop=True)
    return da


def get_time_grid(dtype, value=1):
    """
    This function creates a dataarray with scalar values for a time grid of ntimes steps.
    """
    ntimes = 10
    shape = (ntimes,)
    dims = ("time",)
    coords = {"time": pd.date_range("2000-01-01", periods=ntimes)}

    da_time = xr.DataArray(
        np.ones(shape, dtype=dtype) * value, coords=coords, dims=dims
    )

    da_grid = get_grid_da(dtype, value, subunit=False)

    return da_time * da_grid


def get_landuse_da(dtype, value=1):
    landuse_index = np.arange(1, 4)
    coords = {"landuse_index": landuse_index}

    values = np.full((3,), fill_value=value, dtype=dtype)

    lu_da = xr.DataArray(data=values, coords=coords, dims=("landuse_index",))
    return lu_da


def get_vegetation_da(dtype, value=1):
    vegetation_index = np.arange(1, 4)
    day_of_year = np.arange(1, 367)
    coords = {"vegetation_index": vegetation_index, "day_of_year": day_of_year}

    values = np.full((3, 366), fill_value=value, dtype=dtype)

    veg_da = xr.DataArray(
        data=values, coords=coords, dims=("vegetation_index", "day_of_year")
    )
    return veg_da


def _paths():
    return [Path("path"), "path"]


def get_package_instances():
    return [
        FileCopier(_paths()),
        CouplerMapping(),
        GridData(
            area=get_grid_da(float),
            landuse=get_grid_da(int),
            rootzone_depth=get_grid_da(float),
            surface_elevation=get_grid_da(float, subunit=False),
            soil_physical_unit=get_grid_da(int, subunit=False),
            active=get_grid_da(bool, subunit=False),
        ),
        IdfMapping(
            area=get_grid_da(float),
            nodata=-9999.0,
        ),
        Infiltration(
            infiltration_capacity=get_grid_da(float),
            downward_resistance=get_grid_da(float),
            upward_resistance=get_grid_da(float),
            bottom_resistance=get_grid_da(float, subunit=False),
            extra_storage_coefficient=get_grid_da(float, subunit=False),
        ),
        InitialConditionsEquilibrium(),
        InitialConditionsPercolation(),
        InitialConditionsRootzonePressureHead(),
        InitialConditionsSavedState(Path("path")),
        LanduseOptions(
            landuse_name=get_landuse_da(str),
            vegetation_index=get_landuse_da(int),
            jarvis_o2_stress=get_landuse_da(float),
            jarvis_drought_stress=get_landuse_da(float),
            feddes_p1=get_landuse_da(float),
            feddes_p2=get_landuse_da(float),
            feddes_p3h=get_landuse_da(float),
            feddes_p3l=get_landuse_da(float),
            feddes_p4=get_landuse_da(float),
            feddes_t3h=get_landuse_da(float),
            feddes_t3l=get_landuse_da(float),
            threshold_sprinkling=get_landuse_da(float),
            fraction_evaporated_sprinkling=get_landuse_da(float),
            gift=get_landuse_da(float),
            gift_duration=get_landuse_da(float),
            rotational_period=get_landuse_da(float),
            start_sprinkling_season=get_landuse_da(float),
            end_sprinkling_season=get_landuse_da(float),
            interception_option=get_landuse_da(int),
        ),
        MeteoGrid(
            get_time_grid(float),
            get_time_grid(float),
        ),
        MeteoGridCopy(Path("path")),
        EvapotranspirationMapping(get_time_grid(float)),
        PrecipitationMapping(get_time_grid(float)),
        TimeOutputControl(get_time_grid(float)),
        VariableOutputControl(),
        Ponding(
            get_grid_da(float),
            get_grid_da(float),
            get_grid_da(float),
        ),
        ScalingFactors(
            get_grid_da(float),
            get_grid_da(float),
            get_grid_da(float),
            get_grid_da(float, subunit=False),
        ),
        SprinklingGrid(
            get_grid_da(float),
            get_grid_da(float),
        ),
        SprinklingPoints(
            get_grid_da(int),
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1, 2, 3],
            [1, 1, 1],
            [1.0, 2.0, 3.0],
        ),
        AnnualCropFactors(
            get_vegetation_da(float),
            get_vegetation_da(float),
            get_vegetation_da(float),
            get_vegetation_da(float),
            get_vegetation_da(float),
            get_vegetation_da(float),
            get_vegetation_da(float),
        ),
    ]


MSW_PACKAGE_INSTANCES = get_package_instances()
