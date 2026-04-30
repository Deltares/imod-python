import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imod.mf6 import StructuredDiscretization
from imod.typing import Imod5DataDict


def zeros_grid(n):
    x = np.arange(n, dtype=float) + 1.0
    y = np.arange(n, dtype=float)[::-1] + 1.0
    dx = 1.0
    dy = -1.0

    coords = {"x": x, "y": y, "dx": dx, "dy": dy}
    shape = (len(y), len(x))
    data = np.zeros(shape)

    return xr.DataArray(data, coords=coords, dims=("y", "x"))


def zeros_dask_grid(n):
    da = zeros_grid(n)
    return da.chunk({"x": n, "y": n})


@pytest.fixture(scope="function")
def cap_coupled_dis_grid() -> StructuredDiscretization:
    n = 3
    layer_coord = np.arange(n) + 1
    layer_template = xr.DataArray(
        layer_coord, dims="layer", coords={"layer": layer_coord}
    )
    template_grid = layer_template * zeros_grid(n)

    idomain = template_grid.astype(int) + 1
    top = 10.0
    bottom = xr.DataArray(
        [0.0, -10.0, -20.0], dims="layer", coords={"layer": layer_coord}
    )

    return StructuredDiscretization(idomain=idomain, top=top, bottom=bottom)


@pytest.fixture(scope="function")
def cap_coupled_dis_grid__big() -> StructuredDiscretization:
    n = 1000
    layer_coord = np.arange(3) + 1
    layer_template = xr.DataArray(
        layer_coord, dims="layer", coords={"layer": layer_coord}
    )
    template_grid = layer_template * zeros_dask_grid(n)

    idomain = template_grid.astype(int) + 1
    top = 10.0
    bottom = xr.DataArray(
        [0.0, -10.0, -20.0], dims="layer", coords={"layer": layer_coord}
    )

    return StructuredDiscretization(idomain=idomain, top=top, bottom=bottom)


@pytest.fixture(scope="function")
def cap_data_sprinkling_grid() -> Imod5DataDict:
    n = 3
    boundary = zeros_grid(n) + 1
    wetted_area = zeros_grid(n) + 0.5
    urban_area = zeros_grid(n) + 0.25

    artificial_rch_type = zeros_grid(n)
    artificial_rch_type[:, 1] = 1
    artificial_rch_type[:, 2] = 2
    layer = xr.ones_like(artificial_rch_type)
    layer[:, 1] = 2

    cap_data = {
        "boundary": boundary,
        "wetted_area": wetted_area,
        "urban_area": urban_area,
        "artificial_recharge": artificial_rch_type,
        "artificial_recharge_layer": layer,
        "artificial_recharge_capacity": xr.DataArray(25.0),
    }

    return {"cap": cap_data, "extra": {"paths": ["path1", "path2"]}}


@pytest.fixture(scope="function")
def cap_data_sprinkling_grid__big() -> Imod5DataDict:
    n = 1000
    boundary = zeros_dask_grid(n) + 1
    wetted_area = zeros_dask_grid(n) + 0.5
    urban_area = zeros_dask_grid(n) + 0.25

    artificial_rch_type = zeros_dask_grid(n) + 2.0
    layer = xr.ones_like(artificial_rch_type)
    layer[:, 1] = 2

    cap_data = {
        "boundary": boundary,
        "wetted_area": wetted_area,
        "urban_area": urban_area,
        "artificial_recharge": artificial_rch_type,
        "artificial_recharge_layer": layer,
        "artificial_recharge_capacity": xr.DataArray(25.0),
    }

    return {"cap": cap_data, "extra": {"paths": ["path1", "path2"]}}


@pytest.fixture(scope="function")
def cap_data_sprinkling_grid_with_layer() -> Imod5DataDict:
    n = 3
    boundary = zeros_grid(n) + 1
    wetted_area = zeros_grid(n) + 0.5
    urban_area = zeros_grid(n) + 0.25

    artificial_rch_type = zeros_grid(n)
    artificial_rch_type[:, 1] = 1
    artificial_rch_type[:, 2] = 2
    layer = xr.ones_like(artificial_rch_type)
    layer[:, 1] = 2

    cap_data = {
        "boundary": boundary.expand_dims("layer"),
        "wetted_area": wetted_area.expand_dims("layer"),
        "urban_area": urban_area.expand_dims("layer"),
        "artificial_recharge": artificial_rch_type.expand_dims("layer"),
        "artificial_recharge_layer": layer.expand_dims("layer"),
        "artificial_recharge_capacity": xr.DataArray(25.0),
    }

    return {"cap": cap_data, "extra": {"paths": ["path1", "path2"]}}


@pytest.fixture(scope="function")
def cap_data_sprinkling_points() -> Imod5DataDict:
    n = 3
    boundary = zeros_grid(n) + 1
    wetted_area = zeros_grid(n) + 0.5
    urban_area = zeros_grid(n) + 0.25

    artificial_rch_type = zeros_grid(n)
    artificial_rch_type[:, 1] = 3000
    artificial_rch_type[:, 2] = 4000

    data = {
        "id": [3000, 4000],
        "layer": [2, 3],
        "capacity": [15.0, 30.0],
        "y": [1.0, 2.0],
        "x": [1.0, 2.0],
    }

    layer = pd.DataFrame(data=data)
    cap_data = {
        "boundary": boundary,
        "wetted_area": wetted_area,
        "urban_area": urban_area,
        "artificial_recharge": artificial_rch_type,
        "artificial_recharge_layer": layer,
        "artificial_recharge_capacity": xr.DataArray(25.0),
    }

    return {"cap": cap_data, "extra": {"paths": ["path1", "path2"]}}
