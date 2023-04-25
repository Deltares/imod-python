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


def get_grid_array(is_unstructured, dtype, value =1, include_time = False):
    """
    helper function for creating an xarray dataset of a given type
    Depending on the is_unstructured input parameter, it will create an array for a
    structured grid or for an unstructured grid.
    """

    if not is_unstructured:
        if include_time:
            shape =  ntime,nlay, nrow, ncol =1, 3, 9, 9
            dims = ("time", "layer", "y", "x")
        else:
            shape = nlay,  nrow, ncol = 3, 9, 9
            dims = ( "layer", "y", "x")

        dx = 10.0
        dy = -10.0
        xmin = 0.0
        xmax = dx * ncol
        ymin = 0.0
        ymax = abs(dy) * nrow

        layer = np.arange(1, nlay + 1)
        y = np.arange(ymax, ymin, dy) + 0.5 * dy
        x = np.arange(xmin, xmax, dx) + 0.5 * dx
        time =["2000-01-01"]
        if include_time:        
            coords = {"time": time,"layer": layer, "y": y, "x": x}
        else:
            coords = {"layer": layer, "y": y, "x": x}

        da = xr.DataArray(np.ones(shape, dtype=dtype)* value, coords=coords, dims=dims)
        return da
    else:

        grid = imod.data.circle() 
        nface = grid.n_face
        nlayer = 2

        if include_time:
            dims = ("time", "layer", grid.face_dimension)
            shape =(1, nlayer, nface)
        else:
            dims = ( "layer", grid.face_dimension)
            shape = (nlayer, nface)


        idomain = xu.UgridDataArray(
            xr.DataArray(
                np.ones(shape, dtype=dtype)* value,
                coords={"layer": [1, 2]},
                dims=dims,
            ),
            grid=grid,
        )    
        return idomain
    

def get_vertices_discretization():
   
    idomain = get_grid_array(True, int, value =1)
    bottom = idomain * xr.DataArray([5.0, 0.0], dims=["layer"])
    return imod.mf6.VerticesDiscretization(
        top=10.0, bottom=bottom, idomain=idomain
    )

def boundary_array( is_unstructured):
    idomain = get_grid_array(is_unstructured, np.float64)

    # Constant cocnentration
    if is_unstructured:
        boundary_array = xu.full_like(idomain, np.nan)
    else:
        boundary_array = xr.full_like(idomain, np.nan)        
        
    boundary_array[...] = np.nan
    boundary_array[..., 0] = 0.0
    return boundary_array

def concentration_array( is_unstructured):
    concentration = get_grid_array( is_unstructured, np.float64,value=np.nan)
    concentration[..., 0] = 0.0
    concentration =concentration.expand_dims(species =["Na"])  
    return concentration

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


def create_instance_packages(is_unstructured):
    return [
        
        imod.mf6.Dispersion(1e-4, 10.0, 10.0, 5.0, 2.0, 4.0, False, True),
        imod.mf6.InitialConditions(start=get_grid_array(is_unstructured,np.float32)),

        imod.mf6.MobileStorageTransfer(0.35, 0.01, 0.02, 1300.0, 0.1),
        imod.mf6.NodePropertyFlow(get_grid_array(is_unstructured,np.int32), 3.0, True, 32.0, 34.0, 7),

        imod.mf6.SpecificStorage(0.001, 0.1, True, get_grid_array(is_unstructured,np.int32)),
        imod.mf6.StorageCoefficient(0.001, 0.1, True, get_grid_array(is_unstructured,np.int32)),       
   ]

def create_instance_boundary_condition_packages(is_unstructured):
    return  [
    imod.mf6.ConstantConcentration(
        boundary_array(is_unstructured), print_input=True, print_flows=True, save_flows=True
    ),
    imod.mf6.ConstantHead(
        boundary_array(is_unstructured), print_input=True, print_flows=True, save_flows=True
    ),
    imod.mf6.Drainage(
        elevation=get_grid_array(is_unstructured, np.float64, 4),
        conductance=get_grid_array(is_unstructured, np.float64,1e-3)
    ),
    imod.mf6.Evapotranspiration(surface=get_grid_array(is_unstructured, np.float64,3),
                                rate=  get_grid_array(is_unstructured, np.float64,2),
                                depth=   get_grid_array(is_unstructured, np.float64,1),
                                proportion_rate=  get_grid_array(is_unstructured, np.float64,0.2),
                                proportion_depth= get_grid_array(is_unstructured, np.float64,0.2),
                                fixed_cell= True       
    ),
    imod.mf6.GeneralHeadBoundary(head=get_grid_array(is_unstructured, np.float64,3),
                                 conductance=get_grid_array(is_unstructured, np.float64,0.33)),

    imod.mf6.HorizontalFlowBarrierHydraulicCharacteristic(
        hydraulic_characteristic= get_grid_array(is_unstructured, np.float64,0.33),
        idomain=get_grid_array(is_unstructured, int,1),
        print_input=True),
    imod.mf6.HorizontalFlowBarrierMultiplier(        
        multiplier= get_grid_array(is_unstructured, np.float64,0.33),
        idomain=get_grid_array(is_unstructured, int,1),
        print_input=True),
    imod.mf6.HorizontalFlowBarrierResistance(
        resistance= get_grid_array(is_unstructured, np.float64,0.33),
        idomain=get_grid_array(is_unstructured, int,1),
        print_input=True),
    imod.mf6.Recharge(
        rate= get_grid_array(is_unstructured, np.float64,0.33),
    ),
    imod.mf6.River(stage=get_grid_array(is_unstructured, np.float64,0.33),
                   conductance=get_grid_array(is_unstructured, np.float64,0.33),
                   bottom_elevation=get_grid_array(is_unstructured, np.float64,0.33)),
    imod.mf6.SourceSinkMixing(package_names=["a", "b"], concentration_boundary_type=["a", "b"],auxiliary_variable_name=["a", "b"]),
    imod.mf6.WellDisStructured(
        layer= [3, 2, 1],
        row=[1, 2, 3],
        column= [2, 2, 2],
        rate=[-5.0] * 3,
    ),
    ]


STRUCTURED_GRID_PACKAGES =[
    imod.mf6.StructuredDiscretization(
        2.0, get_grid_array(False, np.float32), get_grid_array(False, np.int32)
    ), 
    imod.mf6.WellDisStructured(
        layer= [3, 2, 1],
        row=[1, 2, 3],
        column= [2, 2, 2],
        rate=[-5.0] * 3,
    )] +[* create_instance_packages(is_unstructured=False),*create_instance_boundary_condition_packages(False)]


UNSTRUCTURED_GRID_PACKAGES =  [get_vertices_discretization()] + [*create_instance_packages(is_unstructured=True)
,  *create_instance_boundary_condition_packages(True)]


ALL_PACKAGE_INSTANCES=GRIDLESS_PACKAGES+STRUCTURED_GRID_PACKAGES+UNSTRUCTURED_GRID_PACKAGES


kk = ''''   imod.mf6.MassSourceLoading(
        rate=get_grid_array(is_unstructured, np.float64,0.33, False))'''
