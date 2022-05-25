import pathlib
import textwrap

import numpy as np
import pytest
import xarray as xr

import imod


def test_render_string():
    oc = imod.mf6.OutputControl(save_concentration="first", save_budget="last")
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    actual = oc.render(directory, "outputcontrol", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
          budget fileout mymodel/mymodel.cbc
          concentration fileout mymodel/mymodel.ucn
        end options

        begin period 1
          save concentration first
          save budget last
        end period
        """
    )
    assert actual == expected


def test_render_string_two_timesteps():
    globaltimes = [np.datetime64("2000-01-01"), np.datetime64("2000-01-02")]
    save_concentration = xr.DataArray(
        ["last", "first"], coords={"time": globaltimes}, dims=("time")
    )

    oc = imod.mf6.OutputControl(
        save_concentration=save_concentration, save_budget="last"
    )
    directory = pathlib.Path("mymodel")
    actual = oc.render(directory, "outputcontrol", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
          budget fileout mymodel/mymodel.cbc
          concentration fileout mymodel/mymodel.ucn
        end options

        begin period 1
          save concentration last
          save budget last
        end period
        begin period 2
          save concentration first
        end period
        """
    )
    assert actual == expected


def test_wrong_arguments():
    globaltimes = [np.datetime64("2000-01-01"), np.datetime64("2000-01-02")]
    save_concentration = xr.DataArray(
        ["last", "first"], coords={"time": globaltimes}, dims=("time")
    )
    save_head = xr.DataArray(
        ["last", "first"], coords={"time": globaltimes}, dims=("time")
    )
    with pytest.raises(ValueError):
        imod.mf6.OutputControl(
            save_concentration=save_concentration, save_head=save_head
        )
