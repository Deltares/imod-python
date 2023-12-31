import textwrap

import pytest

import imod
from imod.schemata import ValidationError


def test_render():
    ims = imod.mf6.Solution(
        modelnames=["GWF_1"],
        print_option="summary",
        csv_output=False,
        no_ptc=True,
        outer_dvclose=1.0e-4,
        outer_maximum=500,
        under_relaxation=None,
        inner_dvclose=1.0e-4,
        inner_rclose=0.001,
        inner_maximum=100,
        linear_acceleration="cg",
        scaling_method=None,
        reordering_method=None,
        relaxation_factor=0.97,
    )
    actual = ims.render(None, None, None, None)
    expected = textwrap.dedent(
        """\
        begin options
          print_option summary
        end options

        begin nonlinear
          outer_dvclose 0.0001
          outer_maximum 500
        end nonlinear

        begin linear
          inner_maximum 100
          inner_dvclose 0.0001
          inner_rclose 0.001
          linear_acceleration cg
          relaxation_factor 0.97
        end linear
        """
    )
    assert expected == actual


def test_wrong_dtype():
    with pytest.raises(ValidationError):
        imod.mf6.Solution(
            modelnames=["GWF_1"],
            print_option="summary",
            csv_output=False,
            no_ptc=True,
            outer_dvclose=4,
            outer_maximum=500,
            under_relaxation=None,
            inner_dvclose=1.0e-4,
            inner_rclose=0.001,
            inner_maximum=100,
            linear_acceleration="cg",
            scaling_method=None,
            reordering_method=None,
            relaxation_factor=0.97,
        )
