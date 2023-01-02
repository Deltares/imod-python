import pathlib
import re
import textwrap

import numpy as np
import pytest
import xarray as xr

import imod
from imod.schemata import ValidationError


def test_render_string():
    oc = imod.mf6.OutputControl(save_head="first", save_budget="last")
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    actual = oc.render(directory, "outputcontrol", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
          budget fileout mymodel/mymodel.cbc
          head fileout mymodel/mymodel.hds
        end options

        begin period 1
          save head first
          save budget last
        end period
        """
    )
    assert actual == expected


def test_render_string_two_timesteps():
    globaltimes = [np.datetime64("2000-01-01"), np.datetime64("2000-01-02")]
    save_head = xr.DataArray(
        ["last", "first"], coords={"time": globaltimes}, dims=("time")
    )

    oc = imod.mf6.OutputControl(save_head=save_head, save_budget="last")
    directory = pathlib.Path("mymodel")
    actual = oc.render(directory, "outputcontrol", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
          budget fileout mymodel/mymodel.cbc
          head fileout mymodel/mymodel.hds
        end options

        begin period 1
          save head last
          save budget last
        end period
        begin period 2
          save head first
        end period
        """
    )
    assert actual == expected


def test_render_int():
    oc = imod.mf6.OutputControl(save_head=4, save_budget=3)
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    actual = oc.render(directory, "outputcontrol", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
          budget fileout mymodel/mymodel.cbc
          head fileout mymodel/mymodel.hds
        end options

        begin period 1
          save head frequency 4
          save budget frequency 3
        end period
        """
    )
    assert actual == expected


def test_render_bool_fail():
    message = textwrap.dedent(
        """
        * save_head
        \t- No option succeeded:
        \tdtype bool != <class 'numpy.integer'>
        \tdtype bool != <U0
        \tdtype bool != object
        * save_budget
        \t- No option succeeded:
        \tdtype bool != <class 'numpy.integer'>
        \tdtype bool != <U0
        \tdtype bool != object"""
    )

    with pytest.raises(ValidationError, match=re.escape(message)):
        imod.mf6.OutputControl(save_head=True, save_budget=False)


def test_render_string_fail():
    oc = imod.mf6.OutputControl(save_head="foo", save_budget="bar")
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]

    expected_message = "Output Control received wrong string. String should be one of ['first', 'last', 'all'], instead got foo"
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        _ = oc.render(directory, "outputcontrol", globaltimes, True)


def test_render_mixed_two_timesteps():
    globaltimes = [np.datetime64("2000-01-01"), np.datetime64("2000-01-02")]
    # Note that we need to create an object array in numpy first,
    # Because xarray automatically converts everything to strings
    # if not dtype=object.
    data = np.array(["last", 5], dtype="object")
    save_head = xr.DataArray(data, coords={"time": globaltimes}, dims=("time"))

    oc = imod.mf6.OutputControl(save_head=save_head, save_budget=None)
    directory = pathlib.Path("mymodel")
    actual = oc.render(directory, "outputcontrol", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
          head fileout mymodel/mymodel.hds
        end options

        begin period 1
          save head last
        end period
        begin period 2
          save head frequency 5
        end period
        """
    )
    assert actual == expected
