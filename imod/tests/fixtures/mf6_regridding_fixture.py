from typing import Callable, Union

import numpy as np
import pytest
import xarray as xr
import xugrid as xu

import imod


def grid_data_structured(
    dtype: type, value: Union[int, float], cellsize: float
) -> xr.DataArray:
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

    structured_grid_data = xr.DataArray(
        np.ones(shape, dtype=dtype) * value, coords=coords, dims=dims
    )

    return structured_grid_data


def grid_data_structured_layered(
    dtype: type, value: Union[int, float], cellsize: float
) -> xr.DataArray:
    """
    This function creates a dataarray with scalar values for a grid of configurable cell size. The values are
    multiplied with the layer index.
    """
    unstructured_grid_data = grid_data_structured(dtype, value, cellsize)
    nlayer = unstructured_grid_data.coords["layer"].max()
    for ilayer in range(1, nlayer + 1):
        layer_value = ilayer * value
        unstructured_grid_data.loc[dict(layer=ilayer)] = layer_value
    return unstructured_grid_data


def grid_data_unstructured(
    dtype: type, value: Union[int, float], cellsize: float
) -> xu.UgridDataArray:
    """
    This function creates a dataarray with scalar values for a grid of configurable cell size.
    First a regular grid is constructed and then this is converted to an ugrid dataarray.
    """
    return xu.UgridDataArray.from_structured(
        grid_data_structured(dtype, value, cellsize)
    )


def grid_data_unstructured_layered(
    dtype: type, value: Union[int, float], cellsize: float
) -> xu.UgridDataArray:
    """
    This function creates a dataarray with scalar values for a grid of configurable cell size. The values are
    multiplied with the layer index. First a regular grid is constructed and then this is converted to an ugrid dataarray.
    """
    return xu.UgridDataArray.from_structured(
        grid_data_structured_layered(dtype, value, cellsize)
    )


def make_model(
    grid_data_function: Callable[[type, Union[int, float], float], xr.DataArray]
    | Callable[[type, Union[int, float], float], xu.UgridDataArray],
    cellsize: float,
) -> imod.mf6.GroundwaterFlowModel:
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
    rch_rate_all = grid_data_function(np.float64, 0.002, cellsize)
    rch_rate = rch_rate_all.sel(layer=[1])

    gwf_model["rch"] = imod.mf6.Recharge(rch_rate)

    return gwf_model


@pytest.fixture(scope="function")
def structured_flow_model() -> imod.mf6.GroundwaterFlowModel:
    cellsize = 2.0

    gwf_model = make_model(grid_data_structured, cellsize)

    bottom = grid_data_structured_layered(np.float64, -1.0, cellsize)
    idomain = grid_data_structured(np.int32, 1, cellsize)
    gwf_model["disv"] = imod.mf6.StructuredDiscretization(
        top=10.0, bottom=bottom, idomain=idomain
    )
    return gwf_model


@pytest.fixture(scope="function")
def unstructured_flow_model() -> imod.mf6.GroundwaterFlowModel:
    cellsize = 2.0

    gwf_model = make_model(grid_data_unstructured, cellsize)

    bottom = grid_data_unstructured_layered(np.float64, -1.0, cellsize)
    idomain = grid_data_unstructured(np.int32, 1, cellsize)
    gwf_model["disv"] = imod.mf6.VerticesDiscretization(
        top=10.0, bottom=bottom, idomain=idomain
    )
    return gwf_model
