import inspect
from inspect import signature

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xugrid as xu

import imod
from imod.mf6.pkgbase import AdvancedBoundaryCondition, BoundaryCondition, Package
import pathlib
def get_darray(dtype):
    """
    helper function for creating an xarray dataset of a given type
    """
    shape = nlay, nrow, ncol = 3, 9, 9
    dx = 10.0
    dy = -10.0
    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = abs(dy) * nrow
    dims = ("layer", "y", "x")

    layer = np.arange(1, nlay + 1)
    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"layer": layer, "y": y, "x": x}

    da = xr.DataArray(np.ones(shape, dtype=dtype), coords=coords, dims=dims)
    return da

def get_vertices_discretization():
    grid = imod.data.circle()

    nface = grid.n_face

    nlayer = 2

    idomain = xu.UgridDataArray(
        xr.DataArray(
            np.ones((nlayer, nface), dtype=np.int32),
            coords={"layer": [1, 2]},
            dims=["layer", grid.face_dimension],
        ),
        grid=grid,
    )    
    k = xu.full_like(idomain, 1.0, dtype=np.float64)    
    bottom = k * xr.DataArray([5.0, 0.0], dims=["layer"])
    return imod.mf6.VerticesDiscretization(
        top=10.0, bottom=bottom, idomain=idomain
    )

def get_idomain():
    nlay = 3
    nrow = 15
    ncol = 15
    ntimes = 3
    globaltimes = np.array(
        [
            "2000-01-01",
            "2000-01-02",
            "2000-01-03",
        ],
        dtype="datetime64[ns]",
    )
    shape = (ntimes, nlay, nrow, ncol)

    dx = 5000.0
    dy = -5000.0
    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = abs(dy) * nrow
    dims = ("time", "layer", "y", "x")

    layer = np.array([1, 2, 3])
    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"time": globaltimes, "layer": layer, "y": y, "x": x}

    # Discretization data
    return xr.DataArray(np.ones(shape), coords=coords, dims=dims)

def state():
    idomain = get_idomain()

    # Constant cocnentration
    concentration = xr.full_like(idomain, np.nan)
    concentration[...] = np.nan
    concentration[..., 0] = 0.0
    return concentration

def constant_concentration():
    concentration = state()

    return imod.mf6.ConstantConcentration(
        concentration, print_input=True, print_flows=True, save_flows=True
    )

def constant_head():
    head = state()

    return imod.mf6.ConstantHead(
        head, print_input=True, print_flows=True, save_flows=True
    )



def drainage():
    globaltimes = np.array(
        [
            "2000-01-01",
            "2000-01-02",
            "2000-01-03",
            "2000-01-04",
            "2000-01-05",
        ],
        dtype="datetime64[ns]",
    )
    repeat_stress = xr.DataArray(
        [
            [globaltimes[3], globaltimes[0]],
            [globaltimes[4], globaltimes[1]],
        ],
        dims=("repeat", "repeat_items"),
    )   
    
    return imod.mf6.Drainage(
        elevation=elevation_fc(),
        conductance=conductance_fc(),
        repeat_stress=repeat_stress,
    )


def elevation_fc():
    idomain = get_idomain()

    elevation = xr.full_like(idomain, np.nan)
    return elevation
def conductance_fc():
    idomain = get_idomain()

    # Constant head
    conductance = xr.full_like(idomain, np.nan)
    return conductance

PACKAGE_INSTANCES = [
    imod.mf6.adv.Advection("upstream"),
    imod.mf6.Buoyancy(
        reference_density=1000.0,
        reference_concentration=[4.0, 25.0],
        density_concentration_slope=[0.7, -0.375],
        modelname=["gwt-1", "gwt-2"],
        species=["salinity", "temperature"],
    ),
    imod.mf6.StructuredDiscretization(
        2.0, get_darray(np.float32), get_darray(np.int32)
    ),
    get_vertices_discretization(),
    imod.mf6.Dispersion(1e-4, 10.0, 10.0, 5.0, 2.0, 4.0, False, True),
    imod.mf6.InitialConditions(start=get_darray(np.float32)),
    imod.mf6.SolutionPresetSimple(modelnames=["gwf-1"]),
    imod.mf6.MobileStorageTransfer(0.35, 0.01, 0.02, 1300.0, 0.1),
    imod.mf6.NodePropertyFlow(get_darray(np.int32), 3.0, True, 32.0, 34.0, 7),
    imod.mf6.OutputControl(),
    imod.mf6.SpecificStorage(0.001, 0.1, True, get_darray(np.int32)),
    imod.mf6.StorageCoefficient(0.001, 0.1, True, get_darray(np.int32)),
    imod.mf6.TimeDiscretization(xr.DataArray(
        data=[0.001, 7.0, 365.0],
        coords={"time": pd.date_range("2000-01-01", "2000-01-03")},
        dims=["time"], ), 23, 1.02),
   
]

BOUNDARY_INSTANCES =[
    constant_concentration(),
    constant_head(),
    drainage()
]
