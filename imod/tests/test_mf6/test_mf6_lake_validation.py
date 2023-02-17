import textwrap

import numpy as np
import pytest
import xarray as xr

from imod.mf6.lak import CONNECTION_DIM, LAKE_DIM, OUTLET_DIM, Lake
from imod.schemata import ValidationError


def test_lake_init_validation_dim_mismatch():
    # this setup contains input errors in the dimension name of the different arrays
    lake_numbers = xr.DataArray([1], dims="dim_0")
    lake_starting_stage = xr.DataArray([1.0], dims="dim_0")
    lake_boundname = xr.DataArray(["lake1"], dims="dim_0")
    connection_lake_number = xr.DataArray([1], dims="dim_0")
    connection_cell_id = xr.DataArray(
        data=[
            [-1, -3],
        ],
        coords={"celldim": ["layer", "cell2d"]},
        dims=(CONNECTION_DIM, "celldim"),
    )
    connection_type = xr.DataArray(["vertical"], dims="dim_0")
    connection_bed_leak = xr.DataArray([1.0], dims="dim_0")
    connection_bottom_elevation = xr.DataArray([1.0], dims="dim_0")
    connection_top_elevation = xr.DataArray([1.0], dims="dim_0")
    connection_width = xr.DataArray([1.0], dims="dim_0")
    connection_length = xr.DataArray([1.0], dims="dim_0")

    with pytest.raises(ValidationError) as error:
        _ = Lake(
            lake_numbers,
            lake_starting_stage,
            lake_boundname,
            connection_lake_number,
            connection_cell_id,
            connection_type,
            connection_bed_leak,
            connection_bottom_elevation,
            connection_top_elevation,
            connection_width,
            connection_length,
        )

    expected = textwrap.dedent(
        """
        * lake_number
        \t- dim mismatch: expected ('lake_dim',), got ('dim_0',)
        * lake_starting_stage
        \t- dim mismatch: expected ('lake_dim',), got ('dim_0',)
        * lake_boundname
        \t- dim mismatch: expected ('lake_dim',), got ('dim_0',)
        * connection_lake_number
        \t- dim mismatch: expected ('connection_dim',), got ('dim_0',)
        * connection_type
        \t- dim mismatch: expected ('connection_dim',), got ('dim_0',)
        * connection_bed_leak
        \t- dim mismatch: expected ('connection_dim',), got ('dim_0',)
        * connection_bottom_elevation
        \t- dim mismatch: expected ('connection_dim',), got ('dim_0',)
        * connection_top_elevation
        \t- dim mismatch: expected ('connection_dim',), got ('dim_0',)
        * connection_width
        \t- dim mismatch: expected ('connection_dim',), got ('dim_0',)
        * connection_length
        \t- dim mismatch: expected ('connection_dim',), got ('dim_0',)"""
    )
    assert str(error.value) == expected


def test_lake_init_validation_type_mismatch():
    # this setup contains input errors in the data type of the different arrays
    lake_numbers = xr.DataArray([1.0], dims=LAKE_DIM)
    lake_starting_stage = xr.DataArray(
        [1],
        dims=LAKE_DIM,
    )
    lake_boundname = xr.DataArray([3], dims=LAKE_DIM)
    connection_lake_number = xr.DataArray([4.2], dims=CONNECTION_DIM)
    connection_cell_id = xr.DataArray(
        data=[
            [1.0, 3.0],
        ],
        coords={"celldim": ["layer", "cell2d"]},
        dims=("boundary", "celldim"),
    )
    connection_type = xr.DataArray([3], dims=CONNECTION_DIM)
    connection_bed_leak = xr.DataArray([1], dims=CONNECTION_DIM)
    connection_bottom_elevation = xr.DataArray([1], dims=CONNECTION_DIM)
    connection_top_elevation = xr.DataArray([1], dims=CONNECTION_DIM)
    connection_width = xr.DataArray([1], dims=CONNECTION_DIM)
    connection_length = xr.DataArray([1], dims=CONNECTION_DIM)

    with pytest.raises(ValidationError) as error:
        _ = Lake(
            lake_numbers,
            lake_starting_stage,
            lake_boundname,
            connection_lake_number,
            connection_cell_id,
            connection_type,
            connection_bed_leak,
            connection_bottom_elevation,
            connection_top_elevation,
            connection_width,
            connection_length,
        )

    expected = textwrap.dedent(
        """
        * lake_number
        \t- dtype float64 != <class 'numpy.integer'>
        * lake_starting_stage
        \t- dtype int32 != <class 'numpy.floating'>
        * lake_boundname
        \t- dtype int32 != <U0
        * connection_lake_number
        \t- dtype float64 != <class 'numpy.integer'>
        * connection_type
        \t- dtype int32 != <U0
        * connection_bed_leak
        \t- dtype int32 != <class 'numpy.floating'>
        * connection_bottom_elevation
        \t- dtype int32 != <class 'numpy.floating'>
        * connection_top_elevation
        \t- dtype int32 != <class 'numpy.floating'>
        * connection_width
        \t- dtype int32 != <class 'numpy.floating'>
        * connection_length
        \t- dtype int32 != <class 'numpy.floating'>"""
    )

    # this replace action is needed because the default int type on teamcity
    # appears to be different from that on developer machines
    expected = expected.replace("int32", str(lake_starting_stage.dtype))
    assert str(error.value) == expected


def test_lake_write_validation_sign_mismatch():
    lake_numbers = xr.DataArray([-1], dims=LAKE_DIM)
    lake_starting_stage = xr.DataArray([1.0], dims=LAKE_DIM)
    lake_boundname = xr.DataArray(["lake1"], dims=LAKE_DIM)
    connection_lake_number = xr.DataArray([-1], dims=CONNECTION_DIM)
    connection_cell_id = xr.DataArray(
        data=[
            [-1, -3],
        ],
        coords={"celldim": ["layer", "cell2d"]},
        dims=("boundary", "celldim"),
    )
    connection_type = xr.DataArray(["vertical"], dims=CONNECTION_DIM)
    connection_bed_leak = xr.DataArray([1.0], dims=CONNECTION_DIM)
    connection_bottom_elevation = xr.DataArray([1.0], dims=CONNECTION_DIM)
    connection_top_elevation = xr.DataArray([1.0], dims=CONNECTION_DIM)
    connection_width = xr.DataArray([-1.0], dims=CONNECTION_DIM)
    connection_length = xr.DataArray([-1.0], dims=CONNECTION_DIM)

    connection_width = xr.DataArray([-1.0], dims=CONNECTION_DIM)
    connection_length = xr.DataArray([-1.0], dims=CONNECTION_DIM)

    outlet_lakein = xr.DataArray([-1], dims=OUTLET_DIM)
    outlet_lakeout = xr.DataArray([-1], dims=OUTLET_DIM)
    outlet_couttype = xr.DataArray(["manning"], dims=OUTLET_DIM)
    outlet_invert = xr.DataArray([-1.0], dims=OUTLET_DIM)
    outlet_roughness = xr.DataArray([-1.0], dims=OUTLET_DIM)
    outlet_width = xr.DataArray([-1.0], dims=OUTLET_DIM)
    outlet_slope = xr.DataArray([-1.0], dims=OUTLET_DIM)

    # this setup contains input errors in the data type of the different arrays

    lake = Lake(
        lake_numbers,
        lake_starting_stage,
        lake_boundname,
        connection_lake_number,
        connection_cell_id,
        connection_type,
        connection_bed_leak,
        connection_bottom_elevation,
        connection_top_elevation,
        connection_width,
        connection_length,
        outlet_lakein,
        outlet_lakeout,
        outlet_couttype,
        outlet_invert,
        outlet_roughness,
        outlet_width,
        outlet_slope,
    )

    errors = lake._validate(Lake._write_schemata)

    assert (
        str(errors["lake_number"])
        == "[ValidationError('values exceed condition: > 0')]"
    )
    assert (
        str(errors["connection_lake_number"])
        == "[ValidationError('values exceed condition: > 0')]"
    )
    assert (
        str(errors["connection_cell_id"])
        == "[ValidationError('values exceed condition: > 0')]"
    )
    assert (
        str(errors["connection_width"])
        == "[ValidationError('values exceed condition: > 0')]"
    )
    assert (
        str(errors["connection_length"])
        == "[ValidationError('values exceed condition: > 0')]"
    )
    assert (
        str(errors["outlet_lakein"])
        == "[ValidationError('values exceed condition: > 0')]"
    )
    assert (
        str(errors["outlet_lakeout"])
        == "[ValidationError('values exceed condition: > 0')]"
    )
    assert (
        str(errors["outlet_width"])
        == "[ValidationError('values exceed condition: > 0')]"
    )
    assert len(errors) == 8
