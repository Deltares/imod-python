import textwrap

import imod


def test_render():
    timestep_duration = [0.001, 7, 365]
    timedis = imod.mf6.TimeDiscretization(
        timestep_duration, n_timesteps=2, timestep_multiplier=1.1
    )
    actual = timedis.render()
    expected = textwrap.dedent(
        """\
        begin options
          time_units days
        end options

        begin dimensions
          nper 3
        end dimensions

        begin perioddata
          0.001 2 1.1
          7.0 2 1.1
          365.0 2 1.1
        end perioddata
        """
    )
    assert actual == expected
