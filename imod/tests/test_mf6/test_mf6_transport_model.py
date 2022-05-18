import textwrap

import imod
import imod.mf6.model


def test_transport_model_rendering():

    adv = imod.mf6.AdvectionCentral()
    disp = imod.mf6.Dispersion(1e-4, 1.0, 10.0, 1.0, 2.0, 3.0, True, True)
    m = imod.mf6.model.GroundwaterTransportModel()
    m["dsp"] = disp
    m["adv"] = adv
    actual = m.render("transport")
    expected = textwrap.dedent(
        """\
      begin options
      end options

      begin packages
        dsp6 transport/dsp.dsp dsp
        adv6 transport/adv.adv adv
      end packages
      """
    )
    assert actual == expected
    )
    assert actual == expected
