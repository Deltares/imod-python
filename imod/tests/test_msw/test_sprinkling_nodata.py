import numpy as np
import pytest
import xarray as xr

from imod.mf6.utilities.imod5_converter import validate_sprinkling_grid_cap_data


def _grid(values: list[list[float]], source: str | None = None) -> xr.DataArray:
    da = xr.DataArray(
        np.array(values, dtype=float),
        dims=("y", "x"),
        coords={"y": [2.0, 1.0], "x": [1.0, 2.0], "dx": 1.0, "dy": -1.0},
    )
    if source is not None:
        da.attrs["source_paths"] = [source]
    return da


def test_validate_sprinkling_grid_cap_data_valid():
    # NoData in layer is fine where there is no sprinkling.
    cap_data = {
        "artificial_recharge": _grid([[0, 1], [2, 0]]),
        "artificial_recharge_layer": _grid([[np.nan, 1], [1, np.nan]]),
    }
    validate_sprinkling_grid_cap_data(cap_data)


def test_validate_sprinkling_grid_cap_data_nodata_in_type_outside_boundary():
    # NoData in type is tolerated outside the active boundary, as long as the
    # layer is valid.
    cap_data = {
        "boundary": _grid([[0, 1], [1, 1]]),
        "artificial_recharge": _grid([[np.nan, 1], [2, 0]]),
        "artificial_recharge_layer": _grid([[1, 1], [1, 1]]),
    }
    validate_sprinkling_grid_cap_data(cap_data)


def test_validate_sprinkling_grid_cap_data_nodata_in_type_inside_boundary():
    cap_data = {
        "boundary": _grid([[1, 1], [1, 1]]),
        "artificial_recharge": _grid([[np.nan, 1], [2, 0]], source="c:/data/type.idf"),
        "artificial_recharge_layer": _grid([[1, 1], [1, 1]]),
    }
    with pytest.raises(ValueError, match="c:/data/type.idf"):
        validate_sprinkling_grid_cap_data(cap_data)
    # Type is not checked when explicitly disabled
    validate_sprinkling_grid_cap_data(cap_data, check_type=False)


def test_validate_sprinkling_grid_cap_data_nodata_in_layer():
    cap_data = {
        "artificial_recharge": _grid([[0, 1], [2, 0]], source="c:/data/type.idf"),
        "artificial_recharge_layer": _grid(
            [[1, 1], [np.nan, 1]], source="c:/data/layer.idf"
        ),
    }
    with pytest.raises(
        ValueError, match="'artificial_recharge_layer': 1 cell"
    ) as excinfo:
        validate_sprinkling_grid_cap_data(cap_data)
    message = str(excinfo.value)
    assert "(1.0, 1.0)" in message
    assert "c:/data/layer.idf" in message
    assert "'artificial_recharge':" not in message


def test_validate_sprinkling_grid_cap_data_nodata_in_type():
    # Layer is valid everywhere, but NoData in the type is still not supported.
    cap_data = {
        "artificial_recharge": _grid([[np.nan, 1], [2, 0]], source="c:/data/type.idf"),
        "artificial_recharge_layer": _grid(
            [[1, 1], [1, 1]], source="c:/data/layer.idf"
        ),
    }
    with pytest.raises(ValueError, match="'artificial_recharge': 1 cell") as excinfo:
        validate_sprinkling_grid_cap_data(cap_data)
    message = str(excinfo.value)
    assert "(1.0, 2.0)" in message
    assert "c:/data/type.idf" in message
    assert "'artificial_recharge_layer':" not in message


def test_validate_sprinkling_grid_cap_data_nodata_in_type_and_layer():
    cap_data = {
        "artificial_recharge": _grid([[np.nan, 1], [2, 0]], source="c:/data/type.idf"),
        "artificial_recharge_layer": _grid(
            [[np.nan, 1], [1, 1]], source="c:/data/layer.idf"
        ),
    }
    with pytest.raises(ValueError, match="NoData") as excinfo:
        validate_sprinkling_grid_cap_data(cap_data)
    message = str(excinfo.value)
    assert "c:/data/type.idf" in message
    assert "c:/data/layer.idf" in message


def test_validate_sprinkling_grid_cap_data_source_from_other_data():
    cap_data = {
        "artificial_recharge": _grid([[np.nan, 1], [2, 0]]),
        "artificial_recharge_layer": _grid([[np.nan, 1], [1, 1]]),
    }
    source_data = {
        "artificial_recharge": _grid([[0, 0], [0, 0]], source="c:/data/type.idf"),
        "artificial_recharge_layer": _grid(
            [[0, 0], [0, 0]], source="c:/data/layer.idf"
        ),
    }
    with pytest.raises(ValueError, match="c:/data/layer.idf"):
        validate_sprinkling_grid_cap_data(cap_data, source_data)


def test_validate_sprinkling_grid_cap_data_without_source():
    cap_data = {
        "artificial_recharge": _grid([[np.nan, 1], [2, 0]]),
        "artificial_recharge_layer": _grid([[np.nan, 1], [1, 1]]),
    }
    with pytest.raises(ValueError, match="unknown"):
        validate_sprinkling_grid_cap_data(cap_data)
