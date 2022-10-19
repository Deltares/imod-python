import pathlib
import re
import textwrap

import numpy as np
import pytest
import xarray as xr

import imod
from imod.schemata import ValidationError


def test_render():
    layer = np.array([1, 2, 3])
    icelltype = xr.DataArray([1, 0, 0], {"layer": layer}, ("layer",))
    k = xr.DataArray([1.0e-3, 1.0e-4, 2.0e-4], {"layer": layer}, ("layer",))
    k33 = xr.DataArray([2.0e-8, 2.0e-8, 2.0e-8], {"layer": layer}, ("layer",))

    npf = imod.mf6.NodePropertyFlow(
        icelltype=icelltype,
        k=k,
        k33=k33,
        variable_vertical_conductance=True,
        dewatered=True,
        perched=True,
        save_flows=True,
    )
    directory = pathlib.Path("mymodel")
    actual = npf.render(directory, "npf", None, True)
    expected = textwrap.dedent(
        """\
        begin options
          save_flows
          variablecv dewatered
          perched
        end options

        begin griddata
          icelltype layered
            constant 1
            constant 0
            constant 0
          k layered
            constant 0.001
            constant 0.0001
            constant 0.0002
          k33 layered
            constant 2e-08
            constant 2e-08
            constant 2e-08
        end griddata
        """
    )
    assert actual == expected


def test_wrong_dtype():
    layer = np.array([1, 2, 3])
    icelltype = xr.DataArray([1, 0, 0], {"layer": layer}, ("layer",))
    k = xr.DataArray([1.0e-3, 1.0e-4, 2.0e-4], {"layer": layer}, ("layer",))
    k33 = xr.DataArray([2.0e-8, 2.0e-8, 2.0e-8], {"layer": layer}, ("layer",))

    with pytest.raises(ValidationError):
        imod.mf6.NodePropertyFlow(
            icelltype=icelltype.astype(np.float64),
            k=k,
            k33=k33,
            variable_vertical_conductance=True,
            dewatered=True,
            perched=True,
            save_flows=True,
        )


def test_wrong_dim():
    layer = np.array([1, 2, 3])
    icelltype = xr.DataArray(
        [[1, 0, 0]],
        {"time": [1], "layer": layer},
        (
            "time",
            "layer",
        ),
    )
    k = xr.DataArray([1.0e-3, 1.0e-4, 2.0e-4], {"layer": layer}, ("layer",))
    k33 = xr.DataArray([2.0e-8, 2.0e-8, 2.0e-8], {"layer": layer}, ("layer",))

    message = textwrap.dedent(
        """
        * icelltype
        \t- No option succeeded:
        \tlength of dims does not match: 2 != 3
        \tlength of dims does not match: 2 != 1
        \tlength of dims does not match: 2 != 0"""
    )

    with pytest.raises(ValidationError, match=re.escape(message)):
        imod.mf6.NodePropertyFlow(
            icelltype=icelltype,
            k=k,
            k33=k33,
            variable_vertical_conductance=True,
            dewatered=True,
            perched=True,
            save_flows=True,
        )
