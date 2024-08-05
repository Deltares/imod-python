
import sys
from pathlib import Path

import pandas as pd

import imod
from imod.formats.prj.prj import open_projectfile_data
from imod.logging.config import LoggerType
from imod.logging.loglevel import LogLevel
from imod.mf6.dis import StructuredDiscretization
from imod.mf6.oc import OutputControl
from imod.mf6.regrid.regrid_schemes import (
    DiscretizationRegridMethod,
    NodePropertyFlowRegridMethod,
    StorageCoefficientRegridMethod,
)
from imod.mf6.simulation import Modflow6Simulation
from imod.prepare.topsystem.allocation import ALLOCATION_OPTION
from imod.prepare.topsystem.conductance import DISTRIBUTING_OPTION
from imod.prepare.topsystem.default_allocation_methods import (
    SimulationAllocationOptions,
    SimulationDistributingOptions,
)

prj_dir = Path("D:\\dev\\imod_python-gh\\vanFrans\\e400_iMOD5\\f03_regional_tests\\c05_Mipwa_01")
data = open_projectfile_data(
    prj_dir / "c05_Mipwa_01__root__.PRJ"
)

tmp_path = "D:\\dev\\imod_python-gh\\tmp"
logfile_path = tmp_path / "logfile.txt"
with open(logfile_path, "w") as sys.stdout:
    imod.logging.configure(
        LoggerType.PYTHON,
        log_level=LogLevel.DEBUG,
        add_default_file_handler=False,
        add_default_stream_handler=True,
    )    

    imod5_data = data[0]
    period_data = data[1]
    default_simulation_allocation_options = SimulationAllocationOptions
    default_simulation_distributing_options = SimulationDistributingOptions

    regridding_option = {}
    regridding_option["npf"] = NodePropertyFlowRegridMethod()
    regridding_option["dis"] = DiscretizationRegridMethod()
    regridding_option["sto"] = StorageCoefficientRegridMethod()
    times = pd.date_range(start="1/1/1989", end="12/1/1991", freq="ME")

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

    for k, package in simulation["imported_model"].items():
        package.dataset.load()
    simulation.write(tmp_path, binary=False, validate=True)
