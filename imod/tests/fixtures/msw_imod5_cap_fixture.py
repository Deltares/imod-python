import numpy as np
import pytest
import xarray as xr
from numpy import nan

from imod.typing import GridDataDict


@pytest.fixture(scope="function")
def imod5_cap_data() -> GridDataDict:
    x = [1.0, 2.0, 3.0]
    y = [3.0, 2.0, 1.0]
    layer = [1]
    dx = 1.0
    dy = -1.0

    da_kwargs = {}
    da_kwargs["dims"] = ("layer", "y", "x")
    da_kwargs["coords"] = {"layer": layer, "y": y, "x": x, "dx": dx, "dy": dy}

    imod5_data = {}
    d_cap = {}

    # fmt: off
    d_cap["boundary"] = xr.DataArray(
        np.array([
            [
                [1, 1, 1],
                [0, 0, 0],
                [1, 1, 0],
            ]],
        dtype=int),
        **da_kwargs
    )
    d_cap["landuse"] = xr.DataArray(
        np.array([
            [
                [1, 2, 3],
                [0, 0, 0],
                [3, 2, 1],
            ]],
        dtype=int),
        **da_kwargs
    )
    d_cap["rootzone_thickness"] = xr.DataArray(
        np.array([
            [
                [0.1, 0.1, 0.1],
                [nan, nan, nan],
                [0.2, 0.2, 0.2],
            ]]
        ),
        **da_kwargs
    )
    d_cap["soil_physical_unit"] = xr.DataArray(
        np.array([
            [
                [1, 1, 1],
                [0, 0, 0],
                [2, 1, 1],
            ]],
        dtype=int),
        **da_kwargs
    )
    d_cap["surface_elevation"] = xr.DataArray(
        np.array([
            [
                [1.1, 1.2, 1.3],
                [nan, nan, nan],
                [1.4, 1.5, 1.6],
            ]]
        ),
        **da_kwargs
    )
    d_cap["artificial_recharge"] = xr.DataArray(
        np.array([
            [
                [0.2, 0.2, 0.2],
                [nan, nan, nan],
                [0.3, 0.3, 0.3],
            ]]
        ),
        **da_kwargs
    )
    d_cap["artificial_recharge_layer"] = xr.DataArray(
        np.array([
            [
                [1, 2, 3],
                [0, 0, 0],
                [3, 2, 1],
            ]],
        dtype=int),
        **da_kwargs
    )
    d_cap["artificial_recharge_capacity"] = xr.DataArray(
        np.array([
            [
                [0.4, 0.4, 0.4],
                [nan, nan, nan],
                [0.6, 0.6, 0.6],
            ]]
        ),
        **da_kwargs
    )
    d_cap["wetted_area"] = xr.DataArray(
        np.array([
            [
                [0.1, 0.1, 0.1],
                [nan, nan, nan],
                [0.2, 0.2, 0.2],
            ]]
        ),
        **da_kwargs
    )
    d_cap["urban_area"] = xr.DataArray(
        np.array([
            [
                [0.2, 0.2, 0.2],
                [nan, nan, nan],
                [0.3, 0.3, 0.3],
            ]]
        ),
        **da_kwargs
    )
    d_cap["urban_ponding_depth"] = xr.DataArray(
        np.array([
            [
                [2.2, 2.2, 2.2],
                [nan, nan, nan],
                [2.3, 2.3, 2.3],
            ]]
        ),
        **da_kwargs
    )
    d_cap["rural_ponding_depth"] = xr.DataArray(
        np.array([
            [
                [1.2, 1.2, 1.2],
                [nan, nan, nan],
                [1.3, 1.3, 1.3],
            ]]
        ),
        **da_kwargs
    )
    d_cap["urban_runoff_resistance"] = xr.DataArray(
        np.array([
            [
                [3.2, 3.2, 3.2],
                [nan, nan, nan],
                [3.3, 3.3, 3.3],
            ]]
        ),
        **da_kwargs
    )
    d_cap["rural_runoff_resistance"] = xr.DataArray(
        np.array([
            [
                [3.6, 3.6, 3.6],
                [nan, nan, nan],
                [3.7, 3.7, 3.7],
            ]]
        ),
        **da_kwargs
    )
    d_cap["urban_runon_resistance"] = xr.DataArray(
        np.array([
            [
                [5.2, 5.2, 5.2],
                [nan, nan, nan],
                [5.3, 5.3, 5.3],
            ]]
        ),
        **da_kwargs
    )
    d_cap["rural_runon_resistance"] = xr.DataArray(
        np.array([
            [
                [5.6, 5.6, 5.6],
                [nan, nan, nan],
                [5.7, 5.7, 5.7],
            ]]
        ),
        **da_kwargs
    )
    d_cap["urban_infiltration_capacity"] = xr.DataArray(
        np.array([
            [
                [10.2, 10.2, 10.2],
                [nan, nan, nan],
                [10.3, 10.3, 10.3],
            ]]
        ),
        **da_kwargs
    )
    d_cap["rural_infiltration_capacity"] = xr.DataArray(
        np.array([
            [
                [20.2, 20.2, 20.2],
                [nan, nan, nan],
                [20.3, 20.3, 20.3],
            ]]
        ),
        **da_kwargs
    )
    d_cap["perched_water_table_level"]= xr.DataArray(
        np.array([
            [
                [2.0, 2.0, 2.0],
                [nan, nan, nan],
                [2.0, 2.0, 2.0],
            ]]
        ),
        **da_kwargs
    )
    d_cap["soil_moisture_fraction"]= xr.DataArray(
        np.array([
            [
                [1.5, 1.5, 1.5],
                [nan, nan, nan],
                [1.5, 1.5, 1.5],
            ]]
        ),
        **da_kwargs
    )
    d_cap["conductivity_factor"]= xr.DataArray(
        np.array([
            [
                [2.5, 2.5, 2.5],
                [nan, nan, nan],
                [2.5, 2.5, 2.5],
            ]]
        ),
        **da_kwargs
    )
    d_bnd = {}
    d_bnd["ibound"] = xr.DataArray(
        np.array([
            [
                [1, 1, 1],
                [0, 0, 0],
                [1, 1, 0],
            ]],
        dtype=int),
        **da_kwargs
    )
    # fmt: on
    imod5_data["cap"] = d_cap
    imod5_data["bnd"] = d_bnd

    return imod5_data
