import pathlib
import textwrap

import numpy as np
import xarray as xr

import imod
import imod.mf6.model


def test_dispersion_default():
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    disp = imod.mf6.Dispersion(1e-4, 1.0, 10.0)

    actual = disp.render(directory, "dsp", globaltimes, True)
    expected = textwrap.dedent(
        """\
      begin options
      end options

      begin griddata
        diffc
          constant 0.0001
        alh
          constant 1.0
        ath1
          constant 10.0
      end griddata"""
    )

    assert actual == expected


def test_dispersion_options():
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    disp = imod.mf6.Dispersion(1e-4, 1.0, 10.0, 1.0, 2.0, 3.0, True, True)
    actual = disp.render(directory, "dsp", globaltimes, True)
    expected = textwrap.dedent(
        """\
      begin options
        xt3d_off
        xt3d_rhs
      end options

      begin griddata
        diffc
          constant 0.0001
        alh
          constant 1.0
        ath1
          constant 10.0
        alv
          constant 1.0
        ath2
          constant 2.0
        atv
          constant 3.0
      end griddata"""
    )

    assert actual == expected


def test_dispersion_layered():
    def layered(data):
        return xr.DataArray(data, {"layer": [1, 2, 3]}, ("layer",))

    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    disp = imod.mf6.Dispersion(
        diffusion_coefficient=layered([1.0, 2.0, 3.0]),
        longitudinal_horizontal=layered([4.0, 5.0, 6.0]),
        transversal_horizontal1=layered([7.0, 8.0, 9.0]),
        longitudinal_vertical=layered([10.0, 11.0, 12.0]),
        transversal_horizontal2=layered([13.0, 14.0, 15.0]),
        transversal_vertical=layered([16.0, 17.0, 18.0]),
        xt3d_off=True,
        xt3d_rhs=True,
    )

    actual = disp.render(directory, "dsp", globaltimes, True)
    expected = textwrap.dedent(
        """\
      begin options
        xt3d_off
        xt3d_rhs
      end options

      begin griddata
        diffc layered
          constant 1.0
          constant 2.0
          constant 3.0
        alh layered
          constant 4.0
          constant 5.0
          constant 6.0
        ath1 layered
          constant 7.0
          constant 8.0
          constant 9.0
        alv layered
          constant 10.0
          constant 11.0
          constant 12.0
        ath2 layered
          constant 13.0
          constant 14.0
          constant 15.0
        atv layered
          constant 16.0
          constant 17.0
          constant 18.0
      end griddata"""
    )
    assert actual == expected
