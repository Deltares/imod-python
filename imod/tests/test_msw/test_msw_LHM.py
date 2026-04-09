"""
LHM user acceptance tests, these are pytest-marked with 'user_acceptance'.

These require the LHM model to be available on the local drive. The test plan
describes how to set this up.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

import imod
from imod.formats.prj import open_projectfile_data
from imod.logging.config import LoggerType
from imod.logging.loglevel import LogLevel
from imod.mf6 import StructuredDiscretization
from imod.mf6.mf6_wel_adapter import Mf6Wel


def convert_imod5_to_msw_model(
    lhm_prjfile: Path, times: list
) -> tuple[imod.msw.MetaSwapModel, Mf6Wel, StructuredDiscretization]:
    imod5_data, _ = open_projectfile_data(lhm_prjfile)

    dis_pkg = imod.mf6.StructuredDiscretization.from_imod5_data(
        imod5_data, validate=False
    )
    dis_pkg.dataset.load()  # Load dataset to speed up import
    npf_pkg = imod.mf6.NodePropertyFlow.from_imod5_data(imod5_data, dis_pkg["idomain"])
    npf_pkg.dataset.load()  # Load dataset to speed up import
    wel_pkg = imod.mf6.LayeredWell.from_imod5_cap_data(imod5_data)
    active = dis_pkg["idomain"] == 1
    mf6_wel_pkg = wel_pkg.to_mf6_pkg(
        active, dis_pkg["top"], dis_pkg["bottom"], npf_pkg["k"]
    )

    msw_model = imod.msw.MetaSwapModel.from_imod5_data(imod5_data, dis_pkg, times)
    msw_model["oc"] = imod.msw.VariableOutputControl()

    return msw_model, mf6_wel_pkg, dis_pkg


@pytest.mark.user_acceptance
def test_import_lhm_msw():
    user_acceptance_dir = Path(os.environ["USER_ACCEPTANCE_DIR"])
    lhm_dir = user_acceptance_dir / "LHM_transient"
    lhm_prjfile = lhm_dir / "model" / "LHM_transient_test.PRJ"
    logfile_path = lhm_dir / "logfile_msw.txt"

    out_dir = lhm_dir / "msp_imod-python"
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

        msw_model, mf6_wel_pkg, dis_pkg = convert_imod5_to_msw_model(lhm_prjfile, times)

        msw_model.write(out_dir, dis_pkg, mf6_wel_pkg, validate=False)
