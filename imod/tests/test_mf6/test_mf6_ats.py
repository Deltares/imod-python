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


@pytest.fixture(scope="function")
def ats_dict():
    return {
        "dt_init": _to_time_da(np.array([1.0, 2.0])),
        "dt_min": _to_time_da(np.array([0.5, 1.0])),
        "dt_max": _to_time_da(np.array([2.0, 4.0])),
        "dt_multiplier": _to_time_da(np.array([1.5, 2.0])),
        "dt_fail_divisor": _to_time_da(np.array([2.0, 3.0])),
    }


def test_render_defaults(ats_dict):
    """Render with default values for dt_multiplier and dt_fail_divisor"""
    globaltimes = pd.date_range("2023-01-01", periods=5, freq="D")

    ats_dict.pop("dt_multiplier")
    ats_dict.pop("dt_fail_divisor")

    ats = AdaptiveTimeStepping(validate=True, **ats_dict)

    rendered = ats._render("test_dir", "test_pkg", globaltimes, False)

    expected = dedent("""\
        begin dimensions
          maxats 5
        end dimensions

        begin perioddata
          1 1.0 0.5 2.0 0.0 0.0
          2 1.0 0.5 2.0 0.0 0.0
          3 1.0 0.5 2.0 0.0 0.0
          4 2.0 1.0 4.0 0.0 0.0
          5 2.0 1.0 4.0 0.0 0.0
        end perioddata
        """)

    assert isinstance(rendered, str)
    assert rendered == expected


def test_render_all_transient(ats_dict):
    """Render with all transient values"""
    globaltimes = pd.date_range("2023-01-01", periods=5, freq="D")

    ats = AdaptiveTimeStepping(validate=True, **ats_dict)

    rendered = ats._render("test_dir", "test_pkg", globaltimes, False)

    expected = dedent("""\
        begin dimensions
          maxats 5
        end dimensions

        begin perioddata
          1 1.0 0.5 2.0 1.5 2.0
          2 1.0 0.5 2.0 1.5 2.0
          3 1.0 0.5 2.0 1.5 2.0
          4 2.0 1.0 4.0 2.0 3.0
          5 2.0 1.0 4.0 2.0 3.0
        end perioddata
        """)

    assert isinstance(rendered, str)
    assert rendered == expected


def test_render_mixed(ats_dict):
    """Render with mixed constant and transient values"""
    globaltimes = pd.date_range("2023-01-01", periods=5, freq="D")

    ats_dict["dt_multiplier"] = 1.0
    ats_dict["dt_fail_divisor"] = 0.5

    ats = AdaptiveTimeStepping(validate=True, **ats_dict)

    rendered = ats._render("test_dir", "test_pkg", globaltimes, False)

    expected = dedent("""\
        begin dimensions
          maxats 5
        end dimensions

        begin perioddata
          1 1.0 0.5 2.0 1.0 0.5
          2 1.0 0.5 2.0 1.0 0.5
          3 1.0 0.5 2.0 1.0 0.5
          4 2.0 1.0 4.0 1.0 0.5
          5 2.0 1.0 4.0 1.0 0.5
        end perioddata
        """)

    assert isinstance(rendered, str)
    assert rendered == expected


def test_render_constants():
    """Render with all constant values"""
    ats_dict = {
        "dt_init": 2.0,
        "dt_min": 1.0,
        "dt_max": 4.0,
        "dt_multiplier": 2.0,
        "dt_fail_divisor": 3.0,
    }

    globaltimes = pd.date_range("2023-01-01", periods=5, freq="D")

    ats = AdaptiveTimeStepping(validate=True, **ats_dict)

    rendered = ats._render("test_dir", "test_pkg", globaltimes, False)

    expected = dedent("""\
        begin dimensions
          maxats 5
        end dimensions

        begin perioddata
          1 2.0 1.0 4.0 2.0 3.0
          2 2.0 1.0 4.0 2.0 3.0
          3 2.0 1.0 4.0 2.0 3.0
          4 2.0 1.0 4.0 2.0 3.0
          5 2.0 1.0 4.0 2.0 3.0
        end perioddata
        """)

    assert isinstance(rendered, str)
    assert rendered == expected


def test_validate_init_schemata(ats_dict):
    # Test that the validation of the init schemata works correctly
    ats = AdaptiveTimeStepping(validate=False, **ats_dict)
    assert ats._validate_init_schemata(validate=True) is None

    ats_dict_copy = ats_dict.copy()
    ats_dict_copy.pop("dt_init")  # Remove dt_init to trigger validation error
    erronous_dt_init = xr.DataArray(
        [3.0],
        coords={"wrong_name": [np.datetime64("2023-01-02")]},
        dims=("wrong_name",),
    )
    with pytest.raises(ValidationError, match="dt_init"):
        AdaptiveTimeStepping(validate=True, dt_init=erronous_dt_init, **ats_dict_copy)

    with pytest.raises(ValidationError, match="dt_init"):
        AdaptiveTimeStepping(
            validate=True, dt_init=_to_time_da(np.array([2, 3])), **ats_dict_copy
        )


def test_validate_write_schemata(ats_dict):
    # Test that the validation of the write schemata works correctly
    ats_dict = ats_dict.copy()
    ats_dict["dt_init"] = _to_time_da(np.array([-1.0, 2.0]))
    ats_dict["dt_min"] = _to_time_da(np.array([-0.9, -0.9]))
    ats_dict["dt_max"] = _to_time_da(np.array([-1.0, -1.0]))
    ats_dict["dt_multiplier"] = _to_time_da(np.array([0.5, 0.5]))

    ats = AdaptiveTimeStepping(validate=False, **ats_dict)

    errors = ats._validate(ats._write_schemata)

    assert len(errors) == 3
    expected_keys = {"dt_init", "dt_min", "dt_multiplier"}
    assert len(set(errors.keys()) - expected_keys) == 0
    assert len(errors["dt_min"]) == 2
