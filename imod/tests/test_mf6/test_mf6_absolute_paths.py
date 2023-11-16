import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xugrid as xu

import imod
from imod.mf6.write_context import WriteContext


@pytest.fixture(scope="function")
def uzf_test_data():
    shape = nlay, nrow, ncol = 2, 2, 2
    dx, dy = 10, -10
    xmin = 0.0
    xmax = dx * ncol
    ymin = 0.0
    ymax = abs(dy) * nrow
    dims = ("layer", "y", "x")

    nper = 4
    time = pd.date_range("2018-01-01", periods=nper, freq="H")
    layer = np.arange(1, nlay + 1)
    y = np.arange(ymax, ymin, dy) + 0.5 * dy
    x = np.arange(xmin, xmax, dx) + 0.5 * dx
    coords = {"layer": layer, "y": y, "x": x}

    idomain = xr.DataArray(np.ones(shape), coords=coords, dims=dims)
    idomain[:, 0, 0] = 0
    active = idomain.where(idomain == 1)
    active_time = (
        xr.DataArray(np.ones(time.shape), coords={"time": time}, dims=("time",))
        * active
    )

    d = {}
    d["kv_sat"] = active * 10.0
    d["theta_sat"] = active * 0.1
    d["theta_res"] = active * 0.05
    d["theta_init"] = active * 0.08
    d["epsilon"] = active * 7.0
    d["surface_depression_depth"] = active * 1.0
    d["infiltration_rate"] = active_time * 0.003
    d["et_pot"] = active_time * 0.002
    d["extinction_depth"] = active_time * 1.0
    d["groundwater_ET_function"] = "linear"
    return d


@pytest.mark.usefixtures("circle_model")
def test_simulation_writes_full_paths_if_requested(circle_model, tmp_path):
    simulation = circle_model
    sim_dir = tmp_path / "circle"

    # add grid-data to the storage package to increase test sensitivity
    idomain = simulation["GWF_1"].domain
    specific_storage = xu.full_like(idomain, dtype=float, fill_value=1e-5)
    simulation["GWF_1"]["sto"] = imod.mf6.SpecificStorage(
        specific_storage=specific_storage,
        specific_yield=0.15,
        transient=False,
        convertible=0,
    )

    # add grid-data to the IC package to increase test sensitivity
    start = xu.full_like(idomain, dtype=float, fill_value=2.0)
    simulation["GWF_1"]["ic"] = imod.mf6.InitialConditions(start=start)
    for path_setting in [True, False]:
        simulation.write(sim_dir, binary=False, use_absolute_paths=path_setting)

        # Define some counters of the expected number of slashes in the output file
        # This includes slashes in paths, but also slashes in keywords
        # such as "open/close" found in mf6 input.
        if path_setting:
            simdir_separator_count = str(sim_dir.as_posix()).count("/")
            modeldir_separator_count = simdir_separator_count + 2
            pkgdir_separator_count = simdir_separator_count + 3
        else:
            simdir_separator_count = 0
            modeldir_separator_count = 1
            pkgdir_separator_count = simdir_separator_count + 2

        toplevel_files = {"mfsim.nam": modeldir_separator_count}
        gwf1_files = {
            "chd.chd": pkgdir_separator_count + 1,
            "disv.disv": 2 * pkgdir_separator_count + 2,
            "GWF_1.nam": 7 * modeldir_separator_count,
            "ic.ic": pkgdir_separator_count + 1,
            "npf.npf": 3 * pkgdir_separator_count + 3,
            "oc.oc": 2 * modeldir_separator_count,
            "rch.rch": pkgdir_separator_count + 1,
            "sto.sto": pkgdir_separator_count + 1,
        }

        for fname, expected_separator_count in toplevel_files.items():
            with open(sim_dir / fname, "r") as f:
                content = f.read()
                assert content.count("/") == expected_separator_count

        for fname, expected_separator_count in gwf1_files.items():
            with open(sim_dir / "GWF_1" / fname, "r") as f:
                content = f.read()
                assert content.count("/") == expected_separator_count


def test_uzf_writes_full_paths_if_requested(uzf_test_data, tmp_path):
    uzf = imod.mf6.UnsaturatedZoneFlow(**uzf_test_data)

    global_times = np.array(
        [
            np.datetime64("1999-01-01"),
            np.datetime64("2000-01-01"),
        ]
    )

    for path_setting in [True, False]:
        if path_setting:
            simdir_separator_count = str(tmp_path.as_posix()).count("/")
            pkgdir_separator_count = simdir_separator_count + 3
        else:
            pkgdir_separator_count = 2

        write_context = WriteContext(tmp_path, False, path_setting)
        uzf.write("uzf", global_times, write_context)

        with open(tmp_path / "uzf.uzf", "r") as f:
            content = f.read()
            assert content.count("/") == pkgdir_separator_count * 2
    pass
