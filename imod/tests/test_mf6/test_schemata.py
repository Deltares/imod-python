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
