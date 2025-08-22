import pathlib

import numpy as np

import imod


def test_advection_upstream():
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    a = imod.mf6.AdvectionUpstream()
    actual = a._render(directory, "adv", globaltimes, True)
    expected = "begin options\n  scheme upstream\nend options"
    assert actual == expected


def test_advection_central():
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    a = imod.mf6.AdvectionCentral()
    actual = a._render(directory, "adv", globaltimes, True)
    expected = "begin options\n  scheme central\nend options"
    assert actual == expected


def test_advection_TVD():
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    a = imod.mf6.AdvectionTVD()
    actual = a._render(directory, "adv", globaltimes, True)
    expected = "begin options\n  scheme TVD\nend options"
    assert actual == expected


def test_advection_ats_percel():
    directory = pathlib.Path("mymodel")
    globaltimes = [np.datetime64("2000-01-01")]
    a = imod.mf6.AdvectionTVD(ats_percel=0.5)
    actual = a._render(directory, "adv", globaltimes, True)
    expected = "begin options\n  scheme TVD\n  ats_percel 0.5\nend options"
    assert actual == expected
