import numpy as np
import xarray as xr

from imod.common.utilities.nodata import describe_nodata, describe_source


def _grid(values: list[list[float]], source: str | None = None) -> xr.DataArray:
    da = xr.DataArray(
        np.array(values, dtype=float),
        dims=("y", "x"),
        coords={"y": [2.0, 1.0], "x": [10.0, 20.0], "dx": 10.0, "dy": -1.0},
    )
    if source is not None:
        da.attrs["source_paths"] = [source]
    return da


def test_describe_source():
    assert describe_source(_grid([[1, 1], [1, 1]], "a.idf")) == "a.idf"
    assert "unknown" in describe_source(_grid([[1, 1], [1, 1]]))
    assert "unknown" in describe_source(None)
    assert "unknown" in describe_source(xr.DataArray(1.0))


def test_describe_nodata__locations_and_source():
    data = {"var": _grid([[1, np.nan], [np.nan, 1]], "c:/data/var.idf")}
    required = xr.ones_like(data["var"], dtype=bool)

    message = describe_nodata("var", data, required)

    assert message is not None
    assert "'var': 2 cell(s)" in message
    assert "(20.0, 2.0), (10.0, 1.0)" in message
    assert "c:/data/var.idf" in message


def test_describe_nodata__max_locations():
    data = {"var": _grid([[np.nan, np.nan], [np.nan, np.nan]])}
    required = xr.ones_like(data["var"], dtype=bool)

    message = describe_nodata("var", data, required, n_max_locations=1)

    assert message is not None
    assert "4 cell(s)" in message
    assert "(10.0, 2.0), ..." in message


def test_describe_nodata__not_required():
    data = {"var": _grid([[1, np.nan], [1, 1]])}
    required = _grid([[1, 0], [1, 1]]).astype(bool)

    assert describe_nodata("var", data, required) is None


def test_describe_nodata__singleton_layer_dim():
    da = _grid([[1, np.nan], [1, 1]]).expand_dims("layer")
    required = xr.ones_like(da, dtype=bool)

    message = describe_nodata("var", {"var": da}, required)

    assert message is not None
    assert "(20.0, 2.0)" in message


def test_describe_nodata__missing_variable():
    required = xr.ones_like(_grid([[1, 1], [1, 1]]), dtype=bool)

    assert describe_nodata("var", {}, required) is None


def test_describe_nodata__source_from_other_data():
    data = {"var": _grid([[np.nan, 1], [1, 1]])}
    source_data = {"var": _grid([[0, 0], [0, 0]], "c:/data/var.idf")}
    required = xr.ones_like(data["var"], dtype=bool)

    message = describe_nodata("var", data, required, source_data)

    assert message is not None
    assert "c:/data/var.idf" in message
