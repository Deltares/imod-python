import numpy as np
import pathlib
import imod
import imod.mf6.model
import textwrap

def test_dispersion_default():
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]      
    disp = imod.mf6.Dispersion(False, False, 1e-4, 1, 10) 
    actual =disp.render(directory, "dsp", globaltimes, True)
    expected=textwrap.dedent(
      '''\
      begin options


      end options

      begin griddata
        diffc
          constant 0.0001

        alh
          constant 1

        ath1
          constant 10


      end griddata''')
      
    assert actual == expected


def test_dispersion_options():
  directory = pathlib.Path("mymodel")
  globaltimes = [np.datetime64("2000-01-01")]      
  disp= imod.mf6.Dispersion(True, True, 1e-4, 1, 10, 1,2,3) 
  actual =disp.render(directory, "dsp", globaltimes, True)
  expected=textwrap.dedent(
      '''\
      begin options
       XT3D_OFF
       XT3D_RHS
      end options

      begin griddata
        diffc
          constant 0.0001

        alh
          constant 1

        ath1
          constant 10

        alv
          constant 1

        ath2
          constant 2


        atv
          constant 3

      end griddata''')
      
   
  print(actual)
  assert actual == expected
