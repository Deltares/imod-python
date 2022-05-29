import pathlib
import textwrap

import numpy as np

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
       XT3D_OFF
       XT3D_RHS
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

    print(actual)
    assert actual == expected
