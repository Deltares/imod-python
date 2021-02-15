import pathlib
import shutil

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imod.wq import Well


@pytest.fixture(scope="function")
def well_time_single_layer():
    datetimes = pd.date_range("2000-01-01", "2000-01-05")
    return Well(id_name="well", x=1.0, y=2.0, rate=5.0, layer=2, time=datetimes)


@pytest.fixture(scope="function")
def well_conc():
    datetimes = pd.date_range("2000-01-01", "2000-01-05")
    x = [1.0, 2.0]
    y = [3.0, 4.0]
    return Well(
        id_name="well", x=x, y=y, rate=5.0, layer=2, time=datetimes, concentration=2.5
    )


@pytest.fixture(scope="function")
def well_conc_multiple_species():
    datetimes = pd.date_range("2000-01-01", "2000-01-05")
    x = [1.0, 2.0, 1.0, 2.0]
    y = [3.0, 4.0, 3.0, 4.0]
    species = [1, 1, 2, 2]
    w = Well(
        id_name="well",
        x=x,
        y=y,
        rate=5.0,
        layer=2,
        time=datetimes,
        concentration=2.5,
        species=species,
    )
    return w


def test_render():
    datetimes = pd.date_range("2000-01-01", "2000-01-05")
    wel = Well(id_name="well", x=1.0, y=2.0, rate=5.0, layer=2, time=datetimes)
    directory = pathlib.Path("well")
    compare = """
    wel_p1_s1_l2 = well/well_20000101000000_l2.ipf
    wel_p2_s1_l2 = well/well_20000102000000_l2.ipf
    wel_p3_s1_l2 = well/well_20000103000000_l2.ipf
    wel_p4_s1_l2 = well/well_20000104000000_l2.ipf
    wel_p5_s1_l2 = well/well_20000105000000_l2.ipf"""

    actual = wel._render(
        directory, globaltimes=wel["time"].values, system_index=1, nlayer=3
    )
    assert actual == compare


def test_render__notime_nolayer():
    wel = Well(id_name="well", x=1.0, y=2.0, rate=5.0)
    directory = pathlib.Path("well")
    compare = """
    wel_p?_s1_l? = well/well.ipf"""

    actual = wel._render(directory, globaltimes=["?"], system_index=1, nlayer=3)
    assert actual == compare


def test_render__notime_layer():
    wel = Well(id_name="well", x=1.0, y=2.0, rate=5.0, layer=[1, 2, 3])
    directory = pathlib.Path("well")

    # Test layer 1:3 for a total of 5 layers in the model
    compare = """
    wel_p?_s1_l1:3 = well/well_l:.ipf"""
    actual = wel._render(directory, globaltimes=["?"], system_index=1, nlayer=5)
    assert actual == compare

    # Test layer $ for a total of 3 layers in the model
    # So a well is present in every layer, encoded with the $ token
    compare = """
    wel_p?_s1_l$ = well/well_l$.ipf"""
    actual = wel._render(directory, globaltimes=["?"], system_index=1, nlayer=3)
    assert actual == compare


def test_render__time_nolayer():
    datetimes = pd.date_range("2000-01-01", "2000-01-05")
    wel = Well(id_name="well", x=1.0, y=2.0, rate=5.0, time=datetimes)
    directory = pathlib.Path("well")
    compare = """
    wel_p1_s1_l? = well/well_20000101000000.ipf
    wel_p2_s1_l? = well/well_20000102000000.ipf
    wel_p3_s1_l? = well/well_20000103000000.ipf
    wel_p4_s1_l? = well/well_20000104000000.ipf
    wel_p5_s1_l? = well/well_20000105000000.ipf"""

    assert (
        wel._render(directory, globaltimes=wel["time"].values, system_index=1, nlayer=3)
        == compare
    )


def test_render__time_layer():
    datetimes = pd.date_range("2000-01-01", "2000-01-05")
    layer = [1, 2, 3, 4, 5]
    wel = Well(id_name="well", x=1.0, y=2.0, rate=5.0, layer=layer, time=datetimes)
    directory = pathlib.Path("well")
    compare = """
    wel_p1_s1_l1 = well/well_20000101000000_l1.ipf
    wel_p2_s1_l2 = well/well_20000102000000_l2.ipf
    wel_p3_s1_l3 = well/well_20000103000000_l3.ipf
    wel_p4_s1_l4 = well/well_20000104000000_l4.ipf
    wel_p5_s1_l5 = well/well_20000105000000_l5.ipf"""

    actual = wel._render(
        directory, globaltimes=wel["time"].values, system_index=1, nlayer=3
    )
    assert actual == compare


def test_timemap__single_layer():
    datetimes = pd.date_range("2000-01-01", "2000-01-05")
    time = [datetimes[0], datetimes[1]]
    wel = Well(id_name="well", x=1.0, y=2.0, rate=5.0, layer=1, time=time)
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

    actual = wel._render(directory, globaltimes=datetimes, system_index=1, nlayer=3)
    assert actual == compare


def test_timemap__multiple_layers():
    datetimes = pd.date_range("2000-01-01", "2000-01-04")
    layer = [1, 1, 2]
    time = [datetimes[0], datetimes[1], datetimes[0]]
    wel = Well(id_name="well", x=1.0, y=2.0, rate=5.0, layer=layer, time=time)
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

    actual = wel._render(directory, globaltimes=datetimes, system_index=1, nlayer=3)
    assert actual == compare


def test_render_concentration():
    datetimes = pd.date_range("2000-01-01", "2000-01-05")
    wel = Well(
        id_name="well",
        x=1.0,
        y=2.0,
        rate=5.0,
        layer=2,
        time=datetimes,
        concentration=2.5,
    )
    directory = pathlib.Path("well")
    # Test rate
    compare = """
    wel_p1_s1_l2 = well/well_20000101000000_l2.ipf
    wel_p2_s1_l2 = well/well_20000102000000_l2.ipf
    wel_p3_s1_l2 = well/well_20000103000000_l2.ipf
    wel_p4_s1_l2 = well/well_20000104000000_l2.ipf
    wel_p5_s1_l2 = well/well_20000105000000_l2.ipf"""

    actual = wel._render(
        directory, globaltimes=wel["time"].values, system_index=1, nlayer=3
    )
    assert actual == compare

    # Test concentration
    compare = """
    cwel_t1_p1_l2 = well/well-concentration_20000101000000_l2.ipf
    cwel_t1_p2_l2 = well/well-concentration_20000102000000_l2.ipf
    cwel_t1_p3_l2 = well/well-concentration_20000103000000_l2.ipf
    cwel_t1_p4_l2 = well/well-concentration_20000104000000_l2.ipf
    cwel_t1_p5_l2 = well/well-concentration_20000105000000_l2.ipf"""

    actual = wel._render_ssm(directory, globaltimes=wel["time"].values, nlayer=3)
    assert actual == compare


def test_render_concentration_multiple_species():
    datetimes = pd.date_range("2000-01-01", "2000-01-05")
    concentration = xr.DataArray(
        data=[[1.0, 1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0, 2.0]],
        coords={"species": [1, 2]},
        dims=["species", "_"],
    )
    wel = Well(
        id_name="well",
        x=1.0,
        y=2.0,
        rate=5.0,
        layer=2,
        time=datetimes,
        concentration=concentration,
    )
    directory = pathlib.Path("well")
    # Test rate, not affected by multiple species
    compare = """
    wel_p1_s1_l2 = well/well_20000101000000_l2.ipf
    wel_p2_s1_l2 = well/well_20000102000000_l2.ipf
    wel_p3_s1_l2 = well/well_20000103000000_l2.ipf
    wel_p4_s1_l2 = well/well_20000104000000_l2.ipf
    wel_p5_s1_l2 = well/well_20000105000000_l2.ipf"""

    actual = wel._render(
        directory, globaltimes=wel["time"].values, system_index=1, nlayer=3
    )
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

    actual = wel._render_ssm(directory, globaltimes=wel["time"].values, nlayer=3)

    # Make sure the result is the same when concentration is a 2D numpy array
    wel = Well(
        id_name="well",
        x=1.0,
        y=2.0,
        rate=5.0,
        layer=2,
        time=datetimes,
        concentration=concentration.values,
    )
    actual = wel._render_ssm(directory, globaltimes=wel["time"].values, nlayer=3)
    assert actual == compare


def test_render_concentration_multiple_species():
    datetimes = pd.date_range("2000-01-01", "2000-01-04")
    x = [1.0, 2.0, 1.0, 2.0]
    y = [3.0, 4.0, 3.0, 4.0]
    concentration = xr.DataArray(
        data=[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        coords={"species": [1, 2]},
        dims=["species", "_"],
    )
    wel = Well(
        id_name="well",
        x=x,
        y=y,
        rate=5.0,
        layer=2,
        time=datetimes,
        concentration=concentration,
    )
    directory = pathlib.Path("well")
    # Test rate, not affected by multiple species
    compare = """
    wel_p1_s1_l2 = well/well_20000101000000_l2.ipf
    wel_p2_s1_l2 = well/well_20000102000000_l2.ipf
    wel_p3_s1_l2 = well/well_20000103000000_l2.ipf
    wel_p4_s1_l2 = well/well_20000104000000_l2.ipf"""

    actual = wel._render(
        directory, globaltimes=wel["time"].values, system_index=1, nlayer=3
    )
    print(actual)
    print(compare)
    assert actual == compare

    # Test concentration
    compare = """
    cwel_t1_p1_l2 = well/well-concentration_c1_20000101000000_l2.ipf
    cwel_t1_p2_l2 = well/well-concentration_c1_20000102000000_l2.ipf
    cwel_t1_p3_l2 = well/well-concentration_c1_20000103000000_l2.ipf
    cwel_t1_p4_l2 = well/well-concentration_c1_20000104000000_l2.ipf
    cwel_t2_p1_l2 = well/well-concentration_c2_20000101000000_l2.ipf
    cwel_t2_p2_l2 = well/well-concentration_c2_20000102000000_l2.ipf
    cwel_t2_p3_l2 = well/well-concentration_c2_20000103000000_l2.ipf
    cwel_t2_p4_l2 = well/well-concentration_c2_20000104000000_l2.ipf"""

    actual = wel._render_ssm(directory, globaltimes=wel["time"].values, nlayer=3)
    assert actual == compare


def test_sparse_conversion():
    wel = Well(id_name="well", x=1.0, y=2.0, rate=5.0)
    assert wel.identical(Well.from_sparse_dataset(wel.to_sparse_dataset()))

    datetimes = pd.date_range("2000-01-01", "2000-01-04")
    layer = [1, 1, 2]
    time = [datetimes[0], datetimes[1], datetimes[0]]
    wel = Well(id_name="well", x=1.0, y=2.0, rate=5.0, layer=layer, time=time)
    assert wel.identical(Well.from_sparse_dataset(wel.to_sparse_dataset()))

    datetimes = pd.date_range("2000-01-01", "2000-01-05")
    concentration = [1.0, 2.0, 3.0, 4.0, 5.0]
    wel = Well(
        id_name="well",
        x=1.0,
        y=2.0,
        rate=5.0,
        layer=2,
        time=datetimes,
        concentration=concentration,
    )
    Well.from_sparse_dataset(wel.to_sparse_dataset())
    assert wel.identical(Well.from_sparse_dataset(wel.to_sparse_dataset()))

    concentration = xr.DataArray(
        data=[[1.0, 1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0, 2.0]],
        coords={"species": [1, 2]},
        dims=["species", "_"],
    )
    wel = Well(
        id_name="well",
        x=1.0,
        y=2.0,
        rate=5.0,
        layer=2,
        time=datetimes,
        concentration=concentration,
    )
    Well.from_sparse_dataset(wel.to_sparse_dataset())
    assert wel.identical(Well.from_sparse_dataset(wel.to_sparse_dataset()))


def test_save(tmp_path):
    datetimes = pd.date_range("2000-01-01", "2000-01-05")
    wel = Well(id_name="well", x=1.0, y=2.0, rate=5.0, layer=2, time=datetimes)
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


def test_save_concentration(tmp_path):
    datetimes = pd.date_range("2000-01-01", "2000-01-05")
    wel = Well(
        id_name="well",
        x=1.0,
        y=2.0,
        rate=5.0,
        layer=2,
        time=datetimes,
        concentration=2.5,
    )
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


def test_save_multiple_species(tmp_path):
    datetimes = pd.date_range("2000-01-01", "2000-01-04")
    x = [1.0, 2.0, 1.0, 2.0]
    y = [3.0, 4.0, 3.0, 4.0]
    concentration = xr.DataArray(
        data=[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        coords={"species": [1, 2]},
        dims=["species", "_"],
    )
    wel = Well(
        id_name="well",
        x=x,
        y=y,
        rate=5.0,
        layer=2,
        time=datetimes,
        concentration=concentration,
    )
    wel.save(tmp_path / "well")

    files = [
        "well_20000101000000_l2.ipf",
        "well_20000102000000_l2.ipf",
        "well_20000103000000_l2.ipf",
        "well_20000104000000_l2.ipf",
        "well-concentration_c1_20000101000000_l2.ipf",
        "well-concentration_c1_20000102000000_l2.ipf",
        "well-concentration_c1_20000103000000_l2.ipf",
        "well-concentration_c1_20000104000000_l2.ipf",
        "well-concentration_c2_20000101000000_l2.ipf",
        "well-concentration_c2_20000102000000_l2.ipf",
        "well-concentration_c2_20000103000000_l2.ipf",
        "well-concentration_c2_20000104000000_l2.ipf",
    ]
    for file in files:
        assert (tmp_path / "well" / file).is_file()


def test_save__time_nolayer(tmp_path):
    datetimes = pd.date_range("2000-01-01", "2000-01-05")
    wel = Well(id_name="well", x=1.0, y=2.0, rate=5.0, time=datetimes)
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


def test_sel():
    datetimes = pd.date_range("2000-01-01", "2000-02-01", freq="5D")
    wel = Well(
        id_name=["well1"] * 7 + ["well2"] * 7,
        x=1.0,
        y=1.0,
        rate=np.linspace(1.0, 10.0, 14),
        layer=2,
        time=list(datetimes) + list(datetimes),
    )
    selection = wel.sel(id_name="well2")
    assert len(selection["index"]) == 7
    assert (selection["id_name"].values == "well2").all()

    selection = wel.sel(layer=2)
    assert len(selection.index) == 14


def test_sel_time():
    datetimes = pd.date_range("2000-01-01", "2000-02-01", freq="5D")
    wel = Well(
        id_name=["well1"] * 7 + ["well2"] * 7,
        x=1.0,
        y=1.0,
        rate=np.linspace(1.0, 10.0, 14),
        layer=2,
        time=list(datetimes) + list(datetimes),
    )
    selection = wel.sel(time=slice("2000-01-02", "2000-02-01"))
    assert len(selection.index) == 14
    assert selection.time[0] == pd.Timestamp("2000-01-02")
    assert selection.time[-1] == pd.Timestamp("2000-01-31")

    selection = wel.sel(time=slice("2000-01-12", None))
    assert len(selection.index) == 10
    assert selection.time[0] == pd.Timestamp("2000-01-12")
    assert selection.time[-1] == pd.Timestamp("2000-01-31")

    selection = wel.sel(time="2000-01-07")
    assert len(selection.index) == 2
    assert selection.time[0] == pd.Timestamp("2000-01-07")
    assert np.allclose(selection.rate, [1.692, 6.538], atol=0.001)

    selection = wel.sel(time="2000-01-07", id_name="well2")
    assert len(selection.index) == 1
    assert (np.unique(selection.id_name) == np.array(["well2"])).all()
    assert np.allclose(selection.rate, [6.538], atol=0.001)

    selection = wel.sel(time=["2000-01-01", "2000-01-06"])
    assert len(selection.index) == 4
    assert selection.time[0] == pd.Timestamp("2000-01-01")
    assert np.allclose(selection.rate, [1.0, 1.692, 5.846, 6.538], atol=0.001)

    with pytest.raises(KeyError):
        selection = wel.sel(time=["1990-01-01", "2000-01-06"])


def test_sel_multiple_species():
    datetimes = pd.date_range("2000-01-01", "2000-02-01", freq="5D")
    id_name = ["well1"] * 7 + ["well2"] * 7
    rate = np.linspace(1.0, 10.0, 14)
    time = list(datetimes) + list(datetimes)
    wel = Well(
        id_name=id_name, x=1.0, y=1.0, rate=rate, layer=2, time=time, concentration=2.5
    )
    conc1 = wel["concentration"].assign_coords(species=1)
    conc2 = wel["concentration"].assign_coords(species=2)
    wel["concentration"] = xr.concat([conc1, conc2], dim="species")

    selection = wel.sel(species=1)
    assert len(selection.index) == 14
    assert len(selection["concentration"].dims) == 1

    selection = wel.sel(time=slice("2000-01-07", "2000-02-01"), species=1)
    assert len(selection.index) == 12
    assert len(selection["concentration"].dims) == 1
    assert selection.time[0] == pd.Timestamp("2000-01-07")
    assert selection.time[-1] == pd.Timestamp("2000-01-31")
