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


def get_structured_grid_array(dtype, value =1):
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

    da = xr.DataArray(np.ones(shape, dtype=dtype)* value, coords=coords, dims=dims)
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

def boundary_array():
    idomain = get_structured_grid_array(np.float64)

    # Constant cocnentration
    concentration = xr.full_like(idomain, np.nan)
    concentration[...] = np.nan
    concentration[..., 0] = 0.0
    return concentration
def concentration_array():
    idomain = get_structured_grid_array(np.float64)

    # Constant cocnentration
    concentration = xr.full_like(idomain, np.nan)
    concentration[..., 0] = 0.0
    concentration =concentration.expand_dims(species =["Na"])  
    return concentration

def repeat_stress():
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
    return repeat_stress
    
'''
def evapotranspiration():
    
    directory = pathlib.Path("mymodel")
    globaltimes = np.array(
        [
            "2000-01-01",
            "2000-01-02",
            "2000-01-03",
        ],
        dtype="datetime64[ns]",
    )

    evt = imod.mf6.Evapotranspiration(
        surface=elevation_fc(),
        rate= rate_fc(),
        depth=elevation_fc(),
        proportion_rate=proportion_rate_fc(),
        proportion_depth=proportion_depth_fc(),
        concentration= constant_concentration(),
        concentration_boundary_type="AUX",
    )
    return evt
'''

GRIDLESS_PACKAGES = [
    imod.mf6.adv.Advection("upstream"),
    imod.mf6.Buoyancy(
        reference_density=1000.0,
        reference_concentration=[4.0, 25.0],
        density_concentration_slope=[0.7, -0.375],
        modelname=["gwt-1", "gwt-2"],
        species=["salinity", "temperature"],
    ),
    imod.mf6.OutputControl(),
    imod.mf6.SolutionPresetSimple(modelnames=["gwf-1"]),
    imod.mf6.TimeDiscretization(xr.DataArray(
        data=[0.001, 7.0, 365.0],
        coords={"time": pd.date_range("2000-01-01", "2000-01-03")},
        dims=["time"], ), 23, 1.02),
]

STRUCTURED_GRID_PACKAGES =[

    imod.mf6.StructuredDiscretization(
        2.0, get_structured_grid_array(np.float32), get_structured_grid_array(np.int32)
    ),
    get_vertices_discretization(),
    imod.mf6.Dispersion(1e-4, 10.0, 10.0, 5.0, 2.0, 4.0, False, True),
    imod.mf6.InitialConditions(start=get_structured_grid_array(np.float32)),

    imod.mf6.MobileStorageTransfer(0.35, 0.01, 0.02, 1300.0, 0.1),
    imod.mf6.NodePropertyFlow(get_structured_grid_array(np.int32), 3.0, True, 32.0, 34.0, 7),

    imod.mf6.SpecificStorage(0.001, 0.1, True, get_structured_grid_array(np.int32)),
    imod.mf6.StorageCoefficient(0.001, 0.1, True, get_structured_grid_array(np.int32)),
    
   ]



STRUCTURED_GRID_BOUNDARY_INSTANCES =[
    imod.mf6.ConstantConcentration(
        boundary_array(), print_input=True, print_flows=True, save_flows=True
    ),
    imod.mf6.ConstantHead(
        boundary_array(), print_input=True, print_flows=True, save_flows=True
    ),
    imod.mf6.Drainage(
        elevation=get_structured_grid_array(np.float64, 4),
        conductance=get_structured_grid_array(np.float64,1e-3),
        repeat_stress=repeat_stress(),
    ),
    imod.mf6.Evapotranspiration(surface=get_structured_grid_array(np.float64,3),
                                rate=  get_structured_grid_array(np.float64,2),
                                depth=   get_structured_grid_array(np.float64,1),
                                proportion_rate=  get_structured_grid_array(np.float64,0.2),
                                proportion_depth= get_structured_grid_array(np.float64,0.2),
                                concentration =  concentration_array(),
                                concentration_boundary_type = "auxmixed",
                                fixed_cell= True       
    ),
    imod.mf6.GeneralHeadBoundary(head=get_structured_grid_array(np.float64,3),
                                 conductance=get_structured_grid_array(np.float64,0.33)),

    imod.mf6.HorizontalFlowBarrierHydraulicCharacteristic(
        hydraulic_characteristic= get_structured_grid_array(np.float64,0.33),
        idomain=get_structured_grid_array(np.int,1),
        print_input=True),
    imod.mf6.HorizontalFlowBarrierMultiplier(        
        multiplier= get_structured_grid_array(np.float64,0.33),
        idomain=get_structured_grid_array(np.int,1),
        print_input=True),
    imod.mf6.HorizontalFlowBarrierResistance(
        resistance= get_structured_grid_array(np.float64,0.33),
        idomain=get_structured_grid_array(np.int,1),
        print_input=True),
    imod.mf6.MassSourceLoading(
        rate= get_structured_grid_array(np.float64,0.33),
    ),
    imod.mf6.Recharge(
        rate= get_structured_grid_array(np.float64,0.33),
    ),
    imod.mf6.River(stage=get_structured_grid_array(np.float64,0.33),
                   conductance=get_structured_grid_array(np.float64,0.33),
                   bottom_elevation=get_structured_grid_array(np.float64,0.33)),
    imod.mf6.SourceSinkMixing(package_names=["a", "b"], concentration_boundary_type=["a", "b"],auxiliary_variable_name=["a", "b"]),
    imod.mf6.WellDisStructured(
        layer= [3, 2, 1],
        row=[1, 2, 3],
        column= [2, 2, 2],
        rate=[-5.0] * 3,
    )
    ]
ALL_PACKAGE_INSTANCES=GRIDLESS_PACKAGES+STRUCTURED_GRID_PACKAGES+STRUCTURED_GRID_BOUNDARY_INSTANCES