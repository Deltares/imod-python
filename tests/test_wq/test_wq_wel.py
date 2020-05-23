import pathlib
import shutil

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imod.wq import Well


@pytest.fixture(scope="module")
def well():
    datetimes = pd.date_range("2000-01-01", "2000-01-05")
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    return Well(id_name="well", x=x, y=y, rate=5.0, layer=2, time=datetimes)


@pytest.fixture(scope="module")
def well_conc():
    datetimes = pd.date_range("2000-01-01", "2000-01-05")
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    return Well(
        id_name="well", x=x, y=y, rate=5.0, layer=2, time=datetimes, concentration=2.5
    )


@pytest.fixture(scope="module")
def well_conc_multiple_species():
    datetimes = pd.date_range("2000-01-01", "2000-01-05")
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    w = Well(
        id_name="well", x=x, y=y, rate=5.0, layer=2, time=datetimes, concentration=2.5
    )
    conc1 = w["concentration"].assign_coords(species=1)
    conc2 = w["concentration"].assign_coords(species=2)
    w["concentration"] = xr.concat([conc1, conc2], dim="species")
    return w


def test_render(well):
    wel = well
    directory = pathlib.Path("well")
    compare = """
    wel_p1_s1_l2 = well/well_20000101000000_l2.ipf
    wel_p2_s1_l2 = well/well_20000102000000_l2.ipf
    wel_p3_s1_l2 = well/well_20000103000000_l2.ipf
    wel_p4_s1_l2 = well/well_20000104000000_l2.ipf
    wel_p5_s1_l2 = well/well_20000105000000_l2.ipf"""

    actual = wel._render(directory, globaltimes=wel["time"].values, system_index=1)
    assert actual == compare


def test_render__notime_nolayer(well):
    # Necessary because using drop return a pandas.DataFrame instead of a Well
    # object
    d = {k: v for k, v in well.copy().drop("layer").drop("time").items()}
    wel = Well(**d)
    directory = pathlib.Path("well")
    compare = """
    wel_p?_s1_l? = well/well.ipf"""

    actual = wel._render(directory, globaltimes=["?"], system_index=1)
    assert actual == compare


def test_render__notime_layer(well):
    d = {k: v for k, v in well.copy().drop("time").items()}
    d["layer"] = [1, 2, 3, 4, 5]
    wel = Well(**d)
    directory = pathlib.Path("well")
    compare = """
    wel_p?_s1_l1:5 = well/well_l:.ipf"""

    actual = wel._render(directory, globaltimes=["?"], system_index=1)
    assert actual == compare


def test_render__time_nolayer(well):
    d = {k: v for k, v in well.copy().drop("layer").items()}
    wel = Well(**d)
    directory = pathlib.Path("well")
    compare = """
    wel_p1_s1_l? = well/well_20000101000000.ipf
    wel_p2_s1_l? = well/well_20000102000000.ipf
    wel_p3_s1_l? = well/well_20000103000000.ipf
    wel_p4_s1_l? = well/well_20000104000000.ipf
    wel_p5_s1_l? = well/well_20000105000000.ipf"""

    assert (
        wel._render(directory, globaltimes=wel["time"].values, system_index=1)
        == compare
    )


def test_render__time_layer(well):
    d = {k: v for k, v in well.items()}
    d["layer"] = [1, 2, 3, 4, 5]
    wel = Well(**d)
    directory = pathlib.Path("well")
    compare = """
    wel_p1_s1_l1 = well/well_20000101000000_l1.ipf
    wel_p2_s1_l2 = well/well_20000102000000_l2.ipf
    wel_p3_s1_l3 = well/well_20000103000000_l3.ipf
    wel_p4_s1_l4 = well/well_20000104000000_l4.ipf
    wel_p5_s1_l5 = well/well_20000105000000_l5.ipf"""

    actual = wel._render(directory, globaltimes=wel["time"].values, system_index=1)
    assert actual == compare


def test_timemap__single_layer():
    datetimes = pd.date_range("2000-01-01", "2000-01-05")
    x = [1.0, 1.0]
    y = [1.0, 1.0]
    time = [datetimes[0], datetimes[1]]
    wel = Well(id_name="well", x=x, y=y, rate=5.0, layer=1, time=time)
    # Set a periodic boundary condition
    timemap = {
        datetimes[2]: datetimes[0],
        datetimes[3]: datetimes[1],
        datetimes[4]: datetimes[0],
    }
    wel.add_timemap(timemap)

    directory = pathlib.Path("well")
    compare = """
    wel_p1_s1_l1 = well/well_20000101000000_l1.ipf
    wel_p2_s1_l1 = well/well_20000102000000_l1.ipf
    wel_p3_s1_l1 = well/well_20000101000000_l1.ipf
    wel_p4_s1_l1 = well/well_20000102000000_l1.ipf
    wel_p5_s1_l1 = well/well_20000101000000_l1.ipf"""

    actual = wel._render(directory, globaltimes=datetimes, system_index=1)
    assert actual == compare


def test_timemap__multiple_layers():
    datetimes = pd.date_range("2000-01-01", "2000-01-04")
    layer = [1, 1, 2]
    x = [1.0, 1.0, 1.0]
    y = [1.0, 1.0, 1.0]
    time = [datetimes[0], datetimes[1], datetimes[0]]
    wel = Well(id_name="well", x=x, y=y, rate=5.0, layer=layer, time=time)
    # Set a periodic boundary condition
    # test number of well layers is not constant between periods
    timemap = {datetimes[2]: datetimes[0], datetimes[3]: datetimes[1]}
    wel.add_timemap(timemap)

    directory = pathlib.Path("well")
    compare = """
    wel_p1_s1_l1:2 = well/well_20000101000000_l:.ipf
    wel_p2_s1_l1 = well/well_20000102000000_l1.ipf
    wel_p3_s1_l1:2 = well/well_20000101000000_l:.ipf
    wel_p4_s1_l1 = well/well_20000102000000_l1.ipf"""

    actual = wel._render(directory, globaltimes=datetimes, system_index=1)
    assert actual == compare


def test_render_concentration(well_conc, tmp_path):
    wel = well_conc
    directory = pathlib.Path("well")
    # Test rate
    compare = """
    wel_p1_s1_l2 = well/well_20000101000000_l2.ipf
    wel_p2_s1_l2 = well/well_20000102000000_l2.ipf
    wel_p3_s1_l2 = well/well_20000103000000_l2.ipf
    wel_p4_s1_l2 = well/well_20000104000000_l2.ipf
    wel_p5_s1_l2 = well/well_20000105000000_l2.ipf"""

    actual = wel._render(directory, globaltimes=wel["time"].values, system_index=1)
    assert actual == compare

    # Test concentration
    compare = """
    cwel_t1_p1_l2 = well/well-concentration_20000101000000_l2.ipf
    cwel_t1_p2_l2 = well/well-concentration_20000102000000_l2.ipf
    cwel_t1_p3_l2 = well/well-concentration_20000103000000_l2.ipf
    cwel_t1_p4_l2 = well/well-concentration_20000104000000_l2.ipf
    cwel_t1_p5_l2 = well/well-concentration_20000105000000_l2.ipf"""

    actual = wel._render_ssm(directory, globaltimes=wel["time"].values)
    assert actual == compare


def test_render_concentration_multiple_species(well_conc_multiple_species, tmp_path):
    wel = well_conc_multiple_species
    directory = pathlib.Path("well")
    # Test rate, not affected by multiple species
    compare = """
    wel_p1_s1_l2 = well/well_20000101000000_l2.ipf
    wel_p2_s1_l2 = well/well_20000102000000_l2.ipf
    wel_p3_s1_l2 = well/well_20000103000000_l2.ipf
    wel_p4_s1_l2 = well/well_20000104000000_l2.ipf
    wel_p5_s1_l2 = well/well_20000105000000_l2.ipf"""

    actual = wel._render(directory, globaltimes=wel["time"].values, system_index=1)
    assert actual == compare

    # Test concentration
    compare = """
    cwel_t1_p1_l2 = well/well-concentration_c1_20000101000000_l2.ipf
    cwel_t1_p2_l2 = well/well-concentration_c1_20000102000000_l2.ipf
    cwel_t1_p3_l2 = well/well-concentration_c1_20000103000000_l2.ipf
    cwel_t1_p4_l2 = well/well-concentration_c1_20000104000000_l2.ipf
    cwel_t1_p5_l2 = well/well-concentration_c1_20000105000000_l2.ipf
    cwel_t2_p1_l2 = well/well-concentration_c2_20000101000000_l2.ipf
    cwel_t2_p2_l2 = well/well-concentration_c2_20000102000000_l2.ipf
    cwel_t2_p3_l2 = well/well-concentration_c2_20000103000000_l2.ipf
    cwel_t2_p4_l2 = well/well-concentration_c2_20000104000000_l2.ipf
    cwel_t2_p5_l2 = well/well-concentration_c2_20000105000000_l2.ipf"""

    actual = wel._render_ssm(directory, globaltimes=wel["time"].values)
    assert actual == compare


def test_save(well, tmp_path):
    wel = well
    wel.save(tmp_path / "well")

    files = [
        "well_20000101000000_l2.ipf",
        "well_20000102000000_l2.ipf",
        "well_20000103000000_l2.ipf",
        "well_20000104000000_l2.ipf",
        "well_20000105000000_l2.ipf",
    ]
    for file in files:
        assert (tmp_path / "well" / file).is_file()


def test_save(well_conc, tmp_path):
    wel = well_conc
    wel.save(tmp_path / "well")

    files = [
        "well_20000101000000_l2.ipf",
        "well_20000102000000_l2.ipf",
        "well_20000103000000_l2.ipf",
        "well_20000104000000_l2.ipf",
        "well_20000105000000_l2.ipf",
        "well-concentration_20000101000000_l2.ipf",
        "well-concentration_20000102000000_l2.ipf",
        "well-concentration_20000103000000_l2.ipf",
        "well-concentration_20000104000000_l2.ipf",
        "well-concentration_20000105000000_l2.ipf",
    ]
    for file in files:
        assert (tmp_path / "well" / file).is_file()


def test_save_multiple_species(well_conc_multiple_species, tmp_path):
    wel = well_conc_multiple_species
    wel.save(tmp_path / "well")

    files = [
        "well_20000101000000_l2.ipf",
        "well_20000102000000_l2.ipf",
        "well_20000103000000_l2.ipf",
        "well_20000104000000_l2.ipf",
        "well_20000105000000_l2.ipf",
        "well-concentration_c1_20000101000000_l2.ipf",
        "well-concentration_c1_20000102000000_l2.ipf",
        "well-concentration_c1_20000103000000_l2.ipf",
        "well-concentration_c1_20000104000000_l2.ipf",
        "well-concentration_c1_20000105000000_l2.ipf",
        "well-concentration_c2_20000101000000_l2.ipf",
        "well-concentration_c2_20000102000000_l2.ipf",
        "well-concentration_c2_20000103000000_l2.ipf",
        "well-concentration_c2_20000104000000_l2.ipf",
        "well-concentration_c2_20000105000000_l2.ipf",
    ]
    for file in files:
        assert (tmp_path / "well" / file).is_file()


def test_save__time_nolayer(well, tmp_path):
    d = {k: v for k, v in well.copy().drop("layer").items()}
    wel = Well(**d)
    wel.save(tmp_path / "well")

    files = [
        "well_20000101000000.ipf",
        "well_20000102000000.ipf",
        "well_20000103000000.ipf",
        "well_20000104000000.ipf",
        "well_20000105000000.ipf",
    ]
    for file in files:
        assert (tmp_path / "well" / file).is_file()
