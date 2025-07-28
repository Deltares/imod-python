"""
LHM user acceptance , these are pytest-marked with 'user_acceptance'.

These require the LHM model to be available on the local drive. The test plan
describes how to set this up.
"""

import os
import sys
from pathlib import Path
import dask

import pandas as pd
import pytest

import imod
from imod.formats.prj.prj import open_projectfile_data
from imod.logging.config import LoggerType
from imod.logging.loglevel import LogLevel
from imod.mf6.ims import Solution
from imod.mf6.oc import OutputControl
from imod.mf6.simulation import Modflow6Simulation


def convert_imod5_to_mf6_sim(lhm_prjfile: Path, times: list) -> Modflow6Simulation:
    imod5_data, period_data = open_projectfile_data(lhm_prjfile)

    simulation = Modflow6Simulation.from_imod5_data(
        imod5_data,
        period_data,
        times,
    )
    simulation._validation_context = imod.mf6.ValidationSettings(ignore_time=True)
    # Set settings so that the simulation behaves like iMOD5
    simulation["imported_model"]["oc"] = OutputControl(
        save_head="last", save_budget="last"
    )
    # Mimic iMOD5's "Moderate" settings
    solution = Solution(
        modelnames=["imported_model"],
        print_option="summary",
        outer_csvfile=None,
        inner_csvfile=None,
        no_ptc=None,
        outer_dvclose=0.001,
        outer_maximum=150,
        under_relaxation="dbd",
        under_relaxation_theta=0.9,
        under_relaxation_kappa=0.0001,
        under_relaxation_gamma=0.0,
        under_relaxation_momentum=0.0,
        backtracking_number=0,
        backtracking_tolerance=0.0,
        backtracking_reduction_factor=0.0,
        backtracking_residual_limit=0.0,
        inner_maximum=30,
        inner_dvclose=0.001,
        inner_rclose=100.0,
        rclose_option="strict",
        linear_acceleration="bicgstab",
        relaxation_factor=0.97,
        preconditioner_levels=0,
        preconditioner_drop_tolerance=0.0,
        number_orthogonalizations=0,
    )
    simulation["ims"] = solution

    simulation["imported_model"]["npf"]["xt3d_option"] = True

    return simulation


def cleanup_mf6_sim(simulation: Modflow6Simulation) -> None:
    """
    Cleanup the simulation of erronous package data
    """
    model = simulation["imported_model"]
    for pkg in model.values():
        pkg.dataset.load()

    mask = model.domain
    simulation.mask_all_models(mask)
    dis = model["dis"]

    pkgs_to_cleanup = [
        "riv-1riv",
        "riv-1drn",
        "riv-2riv",
        "riv-2drn",
        "riv-3riv",
        "riv-3drn",
        "riv-4riv",
        "riv-4drn",
        "riv-5riv",
        "riv-5drn",
        "drn-1",
        "drn-2",
        "drn-3",
        "ghb",
    ]

    for pkgname in pkgs_to_cleanup:
        if pkgname in model.keys():
            model[pkgname].cleanup(dis)

    wel_keys = [key for key in model.keys() if "wel-" in key]
    # Account for edge case where iMOD5 allocates to left-hand column of edge,
    # and iMOD Python to right-hand.
    for pkgname in wel_keys:
        model[pkgname].dataset["x"] -= 1e-10

    for pkgname in wel_keys:
        model[pkgname].cleanup(dis)


@pytest.mark.user_acceptance
def test_import_lhm_mf6():
    dask.config.set(num_workers=1)
    user_acceptance_dir = Path(os.environ["USER_ACCEPTANCE_DIR"])
    lhm_dir = user_acceptance_dir / "LHM_transient"
    lhm_prjfile = lhm_dir / "model" / "LHM_transient_test.PRJ"
    logfile_path = lhm_dir / "logfile_mf6.txt"

    out_dir = lhm_dir / "mf6_imod-python"
    out_dir.mkdir(parents=True, exist_ok=True)

    # notes: M/D/Y and convert to list of datetime.
    times = pd.date_range(start="1/1/2011", end="1/1/2012", freq="D").tolist()
    #    os.chdir(lhm_dir)

    with open(logfile_path, "w") as sys.stdout:
        imod.logging.configure(
            LoggerType.PYTHON,
            log_level=LogLevel.DEBUG,
            add_default_file_handler=False,
            add_default_stream_handler=True,
        )
        simulation = convert_imod5_to_mf6_sim(lhm_prjfile, times)
        # Cleanup
        cleanup_mf6_sim(simulation)

        simulation.write(out_dir, False, False)
