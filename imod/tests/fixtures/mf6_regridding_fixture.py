import numpy as np
import xarray as xr
import xugrid as xu
import imod
import pytest
from typing import Union

def grid_data_structured(dtype, value, cellsize) -> xr.DataArray:
    """
    This function creates a dataarray with scalar values for a grid of configurable cell size.
    """
    horizontal_range = 10
    y = np.arange(horizontal_range, -cellsize, -cellsize)
    x = np.arange(0, horizontal_range + cellsize, cellsize)

    nlayer = 3

    shape = nlayer, len(x), len(y)
    dims = ("layer", "y", "x")
    layer = np.arange(1, nlayer + 1)

    coords = {"layer": layer, "y": y, "x": x, "dx": cellsize, "dy": cellsize}

    da = xr.DataArray(np.ones(shape, dtype=dtype) * value, coords=coords, dims=dims)

    return da

def grid_data_structured_layered(dtype, value, cellsize) -> xr.DataArray:
    """
    This function creates a dataarray with scalar values for a grid of configurable cell size. The values are
    multiplied with the layer index.
    """
    horizontal_range = 10
    y = np.arange(horizontal_range, -cellsize, -cellsize)
    x = np.arange(0, horizontal_range + cellsize, cellsize)

    nlayer = 3

    shape = nlayer, len(x), len(y)
    dims = ("layer", "y", "x")
    layer = np.arange(1, nlayer + 1)

    coords = {"layer": layer, "y": y, "x": x, "dx": cellsize, "dy": cellsize}

    da = xr.DataArray(np.ones(shape, dtype=dtype), coords=coords, dims=dims)
    for ilayer in range(1, nlayer + 1):
        layer_value = ilayer * value
        da.loc[dict(layer=ilayer)] = layer_value
    return da


def grid_data_unstructured(dtype, value, cellsize) -> xu.UgridDataArray:
    """
    This function creates a dataarray with scalar values for a grid of configurable cell size.
    First a regular grid is constructed and then this is converted to an ugrid dataarray.
    """
    return xu.UgridDataArray.from_structured(
        grid_data_structured(dtype, value, cellsize)
    )


def grid_data_unstructured_layered(dtype, value, cellsize) -> xu.UgridDataArray:
    """
    This function creates a dataarray with scalar values for a grid of configurable cell size. The values are
    multiplied with the layer index. First a regular grid is constructed and then this is converted to an ugrid dataarray.
    """
    return xu.UgridDataArray.from_structured(
        grid_data_structured_layered(dtype, value, cellsize)
    )

def make_model_from_idomain(grid_data_function, cellsize: float):
    gwf_model = imod.mf6.GroundwaterFlowModel()

    grid_data_function(np.float64, 1, cellsize)
    constant_head = grid_data_function(np.float64, 1, cellsize)
    gwf_model["chd"] = imod.mf6.ConstantHead(
        constant_head, print_input=True, print_flows=True, save_flows=True
    )
    gwf_model["ic"] = imod.mf6.InitialConditions(start=0.0)

    icelltype = grid_data_function(np.int32, 1, cellsize)
    k = grid_data_function(np.float64, 1.23, cellsize)
    k33 = k.copy()

    gwf_model["npf"] = imod.mf6.NodePropertyFlow(
        icelltype=icelltype,
        k=k,
        k33=k33,
        save_flows=True,
    )
    gwf_model["sto"] = imod.mf6.SpecificStorage(
        specific_storage=grid_data_function(np.float64, 0.002, cellsize),
        specific_yield=0.15,
        transient=False,
        convertible=0,
    )
    gwf_model["oc"] = imod.mf6.OutputControl(save_head="all", save_budget="all")
    rch_rate_all =grid_data_function(np.float64, 0.002, cellsize)
    rch_rate = rch_rate_all.sel(layer=[1])

  
    gwf_model["rch"] = imod.mf6.Recharge(rch_rate)

    return gwf_model


@pytest.fixture(scope="session")
def structured_flow_model() -> imod.mf6.GroundwaterFlowModel:
    cellsize = 2.0

    idomain = grid_data_structured(np.int32, 1, cellsize)
    gwf_model = make_model_from_idomain(grid_data_structured, cellsize)

    bottom =grid_data_structured_layered(np.float64, -1.0,cellsize)    

    gwf_model["disv"] = imod.mf6.StructuredDiscretization(
        top=10.0, bottom=bottom, idomain=idomain
    )
    return gwf_model



@pytest.fixture(scope="session")
def unstructured_flow_model() -> imod.mf6.GroundwaterFlowModel:
    cellsize = 2.0

    idomain = grid_data_unstructured(np.int32, 1, cellsize)
    gwf_model = make_model_from_idomain(grid_data_unstructured, cellsize)

    bottom =grid_data_unstructured_layered(np.float64, -1.0,cellsize)    
    
    gwf_model["disv"] = imod.mf6.VerticesDiscretization(
        top=10.0, bottom=bottom, idomain=idomain
    )
    return gwf_model
