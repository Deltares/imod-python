from textwrap import dedent

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imod.mf6.ats import AdaptiveTimeStepping
from imod.schemata import ValidationError

def _to_time_da(data):
    periods = len(data)
    coords = {"time": pd.date_range("2023-01-02", periods=periods, freq="2D")}
    dims = ("time",)
    return xr.DataArray(
        data,
        coords=coords,
        dims=dims,
    )

@pytest.fixture(scope = "function")
def ats_dict():
    return {
        "dt_init": _to_time_da(np.array([1.0, 2.0])),
        "dt_min": _to_time_da(np.array([0.5, 1.0])),
        "dt_max": _to_time_da(np.array([2.0, 4.0])),
        "dt_multiplier": _to_time_da(np.array([1.5, 2.0])),
        "dt_fail_multiplier": _to_time_da(np.array([2.0, 3.0])),
    }



def test_render(ats_dict):
    globaltimes = pd.date_range("2023-01-01", periods=10, freq="D")

    ats = AdaptiveTimeStepping(
        validate=False, **ats_dict
    )

    rendered = ats._render("test_dir", "test_pkg", globaltimes, False)

    expected = dedent("""\
        begin dimensions
          maxats 2
        end dimensions

        begin perioddata
          2 1.0 0.5 2.0 1.5 2.0
          4 2.0 1.0 4.0 2.0 3.0
        end perioddata
        """)

    assert isinstance(rendered, str)
    assert rendered == expected


def test_validate_init_schemata(ats_dict):
    # Test that the validation of the init schemata works correctly
    ats = AdaptiveTimeStepping(
        validate=False, **ats_dict
    )
    assert ats._validate_init_schemata(validate=True) is None

    ats_dict_copy = ats_dict.copy()
    ats_dict_copy.pop("dt_init")  # Remove dt_init to trigger validation error
    with pytest.raises(ValidationError, match="dt_init"):
        AdaptiveTimeStepping(validate=True, dt_init=xr.DataArray(3.0), **ats_dict_copy)
    
    with pytest.raises(ValidationError, match="dt_init"):
        AdaptiveTimeStepping(validate=True, dt_init=_to_time_da(np.array([2, 3])), **ats_dict_copy)

def test_validate_write_schemata(ats_dict):
    # Test that the validation of the write schemata works correctly
    ats_dict = ats_dict.copy()
    ats_dict["dt_init"] = _to_time_da(np.array([-1.0, 2.0]))
    ats_dict["dt_min"] = _to_time_da(np.array([-0.9, -0.9]))
    ats_dict["dt_max"] = _to_time_da(np.array([-1.0, -1.0]))
    ats_dict["dt_multiplier"] = _to_time_da(np.array([0.5, 0.5]))

    ats = AdaptiveTimeStepping(
        validate=False, **ats_dict
    )

    errors = ats._validate(ats._write_schemata)

    assert len(errors) == 3
    expected_keys = {"dt_init", "dt_min", "dt_multiplier"}
    assert len(set(errors.keys()) - expected_keys) == 0
    assert len(errors["dt_min"]) == 2