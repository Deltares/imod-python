import pytest
import xarray as xr

from imod import schemata as sch
from imod.schemata import ValidationError


def test_option_schema():
    # Test for integer values
    schema = sch.OptionSchema([1, 2])
    # None values should be skipped
    assert schema.validate(xr.DataArray(None)) is None
    assert schema.validate(xr.DataArray(1)) is None
    assert schema.validate(xr.DataArray(2)) is None

    with pytest.raises(ValidationError):
        schema.validate(xr.DataArray(3))

    # Test for string values
    schema = sch.OptionSchema(["a", "b"])
    # None values should be skipped
    assert schema.validate(xr.DataArray(None)) is None
    assert schema.validate(xr.DataArray("a")) is None
    # String values are treated case insensitive
    assert schema.validate(xr.DataArray("B")) is None

    with pytest.raises(
        ValidationError, match="Invalid option: c. Valid options are: a, b"
    ):
        schema.validate(xr.DataArray("c"))


def test_AllCoordValues_schema():
    schema = sch.AllCoordsValueSchema("layer", ">", 0)

    da = xr.DataArray([1, 1, 1], coords={"layer": [1, 2, 3]}, dims=("layer",))
    assert schema.validate(da) is None
    assert schema.validate(da.rename(layer="ignore")) is None

    with pytest.raises(
        ValidationError,
        match="Not all values of coordinate layer comply with criterion: > 0",
    ):
        assert schema.validate(da.assign_coords(layer=[0, 1, 2]))
