import pathlib

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imod.wq import Well


@pytest.fixture(scope="module")
def well():
    """Five separate wells, each starting at a different time."""
    datetimes = pd.date_range("2000-01-01", "2000-01-05")
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    id_name = [f"well_{i}" for i in range(x.size)]
    return Well(id_name=id_name, x=x, y=y, rate=5.0, layer=2, time=datetimes)


@pytest.fixture(scope="module")
def well_conc():
    """
    Five separate wells, each starting at a different time, with a single
    species.
    """
    datetimes = pd.date_range("2000-01-01", "2000-01-05")
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    id_name = [f"well_{i}" for i in range(x.size)]
    return Well(
        id_name=id_name, x=x, y=y, rate=5.0, layer=2, time=datetimes, concentration=2.5
    )


@pytest.fixture(scope="module")
def well_conc_multiple_species():
    """
    Five separate wells, each starting at a different time, with two species.
    """
    datetimes = pd.date_range("2000-01-01", "2000-01-05")
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    id_name = [f"well_{i}" for i in range(x.size)]
    w = Well(
        id_name=id_name, x=x, y=y, rate=5.0, layer=2, time=datetimes, concentration=2.5
    )
    conc1 = w["concentration"].assign_coords(species=1)
    conc2 = w["concentration"].assign_coords(species=2)
    w["concentration"] = xr.concat([conc1, conc2], dim="species")
    return w


def test_render(well):
    wel = well
    directory = pathlib.Path("well")
    compare = """
    wel_p?_s1_l2 = well/rate_l2.ipf"""

    actual = wel._render(
        directory, globaltimes=wel["time"].values, system_index=1, nlayer=3
    )
    assert actual == compare


def test_render__notime_nolayer(well):
    # Necessary because using drop return a pandas.DataFrame instead of a Well
    # object
    d = {
        k: v
        for k, v in well.dataset.copy().drop_vars("layer").drop_vars("time").items()
    }
    wel = Well(**d)
    directory = pathlib.Path("well")
    compare = """
    wel_p?_s1_l$ = well/rate.ipf"""

    actual = wel._render(directory, globaltimes=["?"], system_index=1, nlayer=3)
    assert actual == compare


def test_render__notime_layer(well):
    d = {k: v for k, v in well.dataset.copy().drop_vars("time").items()}
    d["layer"] = [1, 2, 3, 4, 5]
    wel = Well(**d)
    directory = pathlib.Path("well")
    compare = """
    wel_p?_s1_l1:5 = well/rate_l:.ipf"""

    actual = wel._render(directory, globaltimes=["?"], system_index=1, nlayer=3)
    assert actual == compare


def test_render__time_nolayer(well):
    d = {k: v for k, v in well.dataset.copy().drop_vars("layer").items()}
    wel = Well(**d)
    directory = pathlib.Path("well")
    compare = """
    wel_p?_s1_l$ = well/rate.ipf"""

    assert (
        wel._render(directory, globaltimes=wel["time"].values, system_index=1, nlayer=3)
        == compare
    )


def test_render__time_layer(well):
    d = {k: v for k, v in well.dataset.items()}
    d["layer"] = [1, 2, 3, 4, 5]
    wel = Well(**d)
    directory = pathlib.Path("well")
    compare = """
    wel_p?_s1_l1:5 = well/rate_l:.ipf"""

    actual = wel._render(
        directory, globaltimes=wel["time"].values, system_index=1, nlayer=3
    )
    assert actual == compare


def test_render_concentration(well_conc, tmp_path):
    wel = well_conc
    directory = pathlib.Path("well")
    # Test rate
    compare = """
    wel_p?_s1_l2 = well/rate_l2.ipf"""

    actual = wel._render(
        directory, globaltimes=wel["time"].values, system_index=1, nlayer=3
    )
    assert actual == compare

    # Test concentration
    compare = """
    cwel_t1_p?_l2 = well/concentration_l2.ipf"""

    actual = wel._render_ssm(directory, globaltimes=wel["time"].values, nlayer=3)
    assert actual == compare


def test_render_concentration_multiple_species(well_conc_multiple_species, tmp_path):
    wel = well_conc_multiple_species
    directory = pathlib.Path("well")
    # Test rate, not affected by multiple species
    compare = """
    wel_p?_s1_l2 = well/rate_l2.ipf"""

    actual = wel._render(
        directory, globaltimes=wel["time"].values, system_index=1, nlayer=3
    )
    assert actual == compare

    # Test concentration
    compare = """
    cwel_t1_p?_l2 = well/concentration_c1_l2.ipf
    cwel_t2_p?_l2 = well/concentration_c2_l2.ipf"""

    actual = wel._render_ssm(directory, globaltimes=wel["time"].values, nlayer=3)
    assert actual == compare


def test_save(well, tmp_path):
    wel = well
    wel.save(tmp_path / "well")

    files = [
        "rate_l2.ipf",
        "rate_l2/well_0.txt",
        "rate_l2/well_1.txt",
        "rate_l2/well_2.txt",
        "rate_l2/well_3.txt",
        "rate_l2/well_4.txt",
    ]
    for file in files:
        assert (tmp_path / "well" / file).is_file()


def test_save_concentration(well_conc, tmp_path):
    wel = well_conc
    wel.save(tmp_path / "well")

    files = [
        "rate_l2.ipf",
        "concentration_l2.ipf",
        "rate_l2/well_0.txt",
        "rate_l2/well_1.txt",
        "rate_l2/well_2.txt",
        "rate_l2/well_3.txt",
        "rate_l2/well_4.txt",
        "concentration_l2/well_0.txt",
        "concentration_l2/well_1.txt",
        "concentration_l2/well_2.txt",
        "concentration_l2/well_3.txt",
        "concentration_l2/well_4.txt",
    ]
    for file in files:
        assert (tmp_path / "well" / file).is_file()


def test_save_multiple_species(well_conc_multiple_species, tmp_path):
    wel = well_conc_multiple_species
    wel.save(tmp_path / "well")

    files = [
        "rate_l2.ipf",
        "rate_l2/well_0.txt",
        "rate_l2/well_1.txt",
        "rate_l2/well_2.txt",
        "rate_l2/well_3.txt",
        "rate_l2/well_4.txt",
        "concentration_c1_l2.ipf",
        "concentration_c2_l2.ipf",
        "concentration_c1_l2/well_0.txt",
        "concentration_c1_l2/well_1.txt",
        "concentration_c1_l2/well_2.txt",
        "concentration_c1_l2/well_3.txt",
        "concentration_c1_l2/well_4.txt",
        "concentration_c2_l2/well_0.txt",
        "concentration_c2_l2/well_1.txt",
        "concentration_c2_l2/well_2.txt",
        "concentration_c2_l2/well_3.txt",
        "concentration_c2_l2/well_4.txt",
    ]
    for file in files:
        assert (tmp_path / "well" / file).is_file()


def test_save__time_nolayer(well, tmp_path):
    d = {k: v for k, v in well.dataset.copy().drop_vars("layer").items()}
    wel = Well(**d)
    wel.save(tmp_path / "well")

    files = [
        "rate.ipf",
        "well_0.txt",
        "well_1.txt",
        "well_2.txt",
        "well_3.txt",
        "well_4.txt",
    ]
    for file in files:
        assert (tmp_path / "well" / file).is_file()


def test_repeat_stress_error(well):
    wel = well
    with pytest.raises(NotImplementedError):
        wel.repeat_stress({})
