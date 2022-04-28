import numpy as np
import pathlib
import textwrap
import imod
import imod.mf6.model


def test_transportModel_rendering():
  directory = pathlib.Path("mymodel")
  globaltimes = [np.datetime64("2000-01-01")]     
  adv = imod.mf6.AdvectionCentral() 
  disp= imod.mf6.Dispersion(True, True, 1e-4, 1, 10, 1,2,3) 
  m = imod.mf6.model.GroundwaterTransportModel()
  m["dsp"] = disp
  m["adv"] = adv
  actual =m.render(directory, "dsp", globaltimes, True)
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
  assert actual == expected