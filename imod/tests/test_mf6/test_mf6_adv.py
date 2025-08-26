import pathlib

import numpy as np
import pytest

import imod
from imod.schemata import ValidationError


def test_advection_upstream():
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    adv = imod.mf6.AdvectionUpstream()
    actual = adv._render(directory, "adv", globaltimes, True)
    expected = "begin options\n  scheme upstream\nend options"
    assert actual == expected


def test_advection_central():
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    adv = imod.mf6.AdvectionCentral()
    actual = adv._render(directory, "adv", globaltimes, True)
    expected = "begin options\n  scheme central\nend options"
    assert actual == expected


def test_advection_TVD():
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    adv = imod.mf6.AdvectionTVD()
    actual = adv._render(directory, "adv", globaltimes, True)
    expected = "begin options\n  scheme TVD\nend options"
    assert actual == expected


def test_advection_ats_percel():
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    adv = imod.mf6.AdvectionTVD(ats_percel=0.5)
    actual = adv._render(directory, "adv", globaltimes, True)
    expected = "begin options\n  scheme TVD\n  ats_percel 0.5\nend options"
    assert actual == expected


def test_advection_ats_percel_invalid():
    """Test that invalid ats_percel values raise ValidationError."""
    # Test for wrong type (int)
    with pytest.raises(ValidationError):
        imod.mf6.AdvectionTVD(ats_percel=1)
    # Test for out of range values (<= 0.0)
    with pytest.raises(ValidationError):
        imod.mf6.AdvectionTVD(ats_percel=-0.1)
    with pytest.raises(ValidationError):
        imod.mf6.AdvectionTVD(ats_percel=0.0)
