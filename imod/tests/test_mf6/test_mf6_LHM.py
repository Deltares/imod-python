import sys
from pathlib import Path

import pandas as pd

import imod
from imod.formats.prj.prj import open_projectfile_data
from imod.logging.config import LoggerType
from imod.logging.loglevel import LogLevel
from imod.mf6.oc import OutputControl
from imod.mf6.regrid.regrid_schemes import (
    DiscretizationRegridMethod,
    NodePropertyFlowRegridMethod,
    StorageCoefficientRegridMethod,
)
from imod.mf6.simulation import Modflow6Simulation
from imod.mf6.utilities.mf6hfb import merge_hfb_packages
from imod.mf6.write_context import WriteContext
from imod.prepare.topsystem.default_allocation_methods import (
    SimulationAllocationOptions,
    SimulationDistributingOptions,
)

LHM_DIR = Path(
    r"c:\Users\engelen\projects_wdir\imod-python\imod5_converter\MODFLOW6_MODEL"
)

# @pytest.fixture(scope="module")
def LHM_imod5_data():
    data = open_projectfile_data(LHM_DIR / "LHM4.3_test.prj")

    imod5_data = data[0]
    period_data = data[1]
    default_simulation_allocation_options = SimulationAllocationOptions
    default_simulation_distributing_options = SimulationDistributingOptions

    regridding_option = {}
    regridding_option["npf"] = NodePropertyFlowRegridMethod()
    regridding_option["dis"] = DiscretizationRegridMethod()
    regridding_option["sto"] = StorageCoefficientRegridMethod()
    times = pd.date_range(start="1/1/2018", end="12/1/2018", freq="ME")

    simulation = Modflow6Simulation.from_imod5_data(
        imod5_data,
        period_data,
        default_simulation_allocation_options,
        default_simulation_distributing_options,
        times,
        regridding_option,
    )
    simulation["imported_model"]["oc"] = OutputControl(
        save_head="last", save_budget="last"
    )
    return simulation


def test_mf6_LHM(tmp_path):
    logfile_path = tmp_path / "logfile.txt"
    with open(logfile_path, "w") as sys.stdout:
        imod.logging.configure(
            LoggerType.PYTHON,
            log_level=LogLevel.DEBUG,
            add_default_file_handler=False,
            add_default_stream_handler=True,
        )
        simulation = LHM_imod5_data()

        for k, package in simulation["imported_model"].items():
            package.dataset.load()
        simulation.write(tmp_path, binary=False, validate=True)


def test_mf6_LHM_write_HFB(tmp_path):
    logfile_path = tmp_path / "logfile.txt"
    with open(logfile_path, "w") as sys.stdout:
        imod.logging.configure(
            LoggerType.PYTHON,
            log_level=LogLevel.DEBUG,
            add_default_file_handler=False,
            add_default_stream_handler=True,
        )
        simulation = LHM_imod5_data()
        model = simulation["imported_model"]

        mf6_hfb_ls = []
        for key, pkg in model.items():
            if issubclass(type(pkg), imod.mf6.HorizontalFlowBarrierBase):
                mf6_hfb_ls.append(pkg)
            pkg.dataset.load()

        top, bottom, idomain = model._Modflow6Model__get_domain_geometry()
        k = model._Modflow6Model__get_k()

        mf6_hfb = merge_hfb_packages(mf6_hfb_ls, idomain, top, bottom, k)

        times = pd.date_range(start="1/1/2018", end="12/1/2018", freq="ME")

        out_dir = tmp_path / "LHM"
        out_dir.mkdir(parents=True, exist_ok=True)
        write_context = WriteContext(out_dir, use_binary=True, use_absolute_paths=False)

        mf6_hfb.write("hfb", times, write_context)


