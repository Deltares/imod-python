import re
import textwrap

import pytest

import imod
from imod.schemata import ValidationError


def create_ims() -> imod.mf6.Solution:
    return imod.mf6.Solution(
        modelnames=["GWF_1"],
        print_option="summary",
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


def test_render():
    ims = imod.mf6.Solution(
        modelnames=["GWF_1"],
        print_option="summary",
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
    actual = ims._render(None, None, None, None)
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


def test_drop_and_add_model():
    ims = create_ims()
    ims.remove_model_from_solution("GWF_1")
    assert "modelnames" not in ims.dataset.keys()
    ims.add_model_to_solution("GWF_2")
    assert "GWF_2" in ims.dataset["modelnames"].values


def test_remove_non_present_model():
    ims = create_ims()
    ims.remove_model_from_solution("GWF_1")
    with pytest.raises(ValueError):
        ims.remove_model_from_solution("GWF_1")


def test_add_already_present_model():
    ims = create_ims()
    with pytest.raises(ValueError):
        ims.add_model_to_solution("GWF_1")


def test_ims_options():
    """
    Ensure inner/outer csvfile, no_ptc, and ats_outer_maximum_fraction are
    written.
    """
    ims = imod.mf6.Solution(
        modelnames=["GWF_1"],
        outer_dvclose=0.001,
        outer_maximum=20,
        inner_maximum=100,
        inner_dvclose=0.0001,
        inner_rclose=1.0,
        linear_acceleration="cg",
        inner_csvfile="inner.csv",
        outer_csvfile="outer.csv",
        no_ptc="first",
        ats_outer_maximum_fraction=0.25,
    )
    actual = ims._render(None, None, None, None)
    expected = textwrap.dedent(
        """\
        begin options
          print_option summary
          outer_csvfile fileout outer.csv
          inner_csvfile fileout inner.csv
          no_ptc first
          ats_outer_maximum_fraction 0.25
        end options

        begin nonlinear
          outer_dvclose 0.001
          outer_maximum 20
        end nonlinear

        begin linear
          inner_maximum 100
          inner_dvclose 0.0001
          inner_rclose 1.0
          linear_acceleration cg
        end linear
        """
    )
    assert expected == actual


def test_ims_option_validation():
    expected = textwrap.dedent(
        """
        - linear_acceleration
            - Invalid option: abc. Valid options are: cg, bicgstab
        - rclose_option
            - Invalid option: any. Valid options are: strict, l2norm_rclose, relative_rclose
        - scaling_method
            - Invalid option: random. Valid options are: diagonal, l2norm
        - reordering_method
            - Invalid option: alphabetical. Valid options are: rcm, md
        - print_option
            - Invalid option: whatever. Valid options are: none, summary, all
        - no_ptc
            - Invalid option: last. Valid options are: first, all
        - ats_outer_maximum_fraction
            - not all values comply with criterion: <= 0.5"""
    )

    with pytest.raises(ValidationError, match=re.escape(expected)):
        imod.mf6.Solution(
            modelnames=["GWF_1"],
            outer_dvclose=0.001,
            outer_maximum=20,
            inner_maximum=100,
            inner_dvclose=0.0001,
            inner_rclose=1.0,
            rclose_option="any",
            linear_acceleration="abc",
            scaling_method="random",
            reordering_method="alphabetical",
            print_option="whatever",
            no_ptc="last",
            ats_outer_maximum_fraction=1.0,
        )
