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


def test_mf6_LHM(tmp_path):
    logfile_path = tmp_path / "logfile.txt"
    with open(logfile_path, "w") as sys.stdout:
        imod.logging.configure(
            LoggerType.PYTHON,
            log_level=LogLevel.DEBUG,
            add_default_file_handler=False,
            add_default_stream_handler=True,
        )
        LHM_dir = Path("D:\\tmp\\LHM\\MODFLOW6_MODEL\\MODFLOW6_MODEL")
        data = open_projectfile_data(LHM_dir / "LHM4.3_withriv.prj")

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

        for k, package in simulation["imported_model"].items():
            package.dataset.load()
        simulation.write(tmp_path, binary=False, validate=True)
    pass


def test_mf6_river(tmp_path):
    LHM_dir = Path("D:\\tmp\\LHM\\MODFLOW6_MODEL\\MODFLOW6_MODEL")
    data = open_projectfile_data(
        LHM_dir / "LHM4.3_stationair_gekalibreerd_bruinkool_fluxes_mf6.prj"
    )



    imod5_data = data[0]
    period_data = data[1]
    default_simulation_allocation_options = SimulationAllocationOptions
    default_simulation_distributing_options = SimulationDistributingOptions

    regridding_option = {}
    regridding_option["npf"] = NodePropertyFlowRegridMethod()
    regridding_option["dis"] = DiscretizationRegridMethod()
    regridding_option["sto"] = StorageCoefficientRegridMethod()
    times = pd.date_range(start="1/1/2018", end="12/1/2018", freq="ME")
    target_dis = StructuredDiscretization.from_imod5_data(imod5_data)
    (riv, drn) = imod.mf6.River.from_imod5_data(
        "riv-1",
        imod5_data,
        period_data,
        target_dis,
        time_min=times[0],
        time_max=times[-1],
        allocation_option_riv=ALLOCATION_OPTION.at_elevation,
        distributing_option_riv=DISTRIBUTING_OPTION.by_crosscut_thickness,
        regridder_types=None,
    )
    riv._validate()


def test_mf6_wel1(tmp_path):
    prj_dir = Path("D:\\dev\\imod_python-gh\\vanFrans\\e400_iMOD5\\f01_basic_tests\\c01_WEL")
    data = open_projectfile_data(
        prj_dir / "c01_WEL_T1__root__.prj"
    )

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

        for k, package in simulation["imported_model"].items():
            package.dataset.load()
        simulation.write(tmp_path, binary=False, validate=True)


def test_mf6_drn1(tmp_path):
    prj_dir = Path("D:\\dev\\imod_python-gh\\vanFrans\\e400_iMOD5\\f01_basic_tests\\c04_DRN")
    data = open_projectfile_data(
        prj_dir / "c04_DRN_T1__root__.prj."
    )

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

        for k, package in simulation["imported_model"].items():
            package.dataset.load()
        simulation.write(tmp_path, binary=False, validate=True)



def test_mf6_riv1(tmp_path):
    prj_dir = Path("D:\\dev\\imod_python-gh\\vanFrans\\e400_iMOD5\\f01_basic_tests\\c05_RIV")
    data = open_projectfile_data(
        prj_dir / "c05_RIV_T1__root__.prj."
    )

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

        for k, package in simulation["imported_model"].items():
            package.dataset.load()
        simulation.write(tmp_path, binary=False, validate=True)



def test_mf6_rch1(tmp_path):
    prj_dir = Path("D:\\dev\\imod_python-gh\\vanFrans\\e400_iMOD5\\f01_basic_tests\\c07_RCH")
    data = open_projectfile_data(
        prj_dir / "c07_RCH_T1__root__.prj."
    )

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

        for k, package in simulation["imported_model"].items():
            package.dataset.load()
        simulation.write(tmp_path, binary=False, validate=True)




def test_mf6_hfb(tmp_path):
    prj_dir = Path("D:\\dev\\imod_python-gh\\vanFrans\\e400_iMOD5\\f01_basic_tests\\c10_HFB")
    data = open_projectfile_data(
        prj_dir / "c10_HFB_T1__root__.prj."
    )

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

        for k, package in simulation["imported_model"].items():
            package.dataset.load()
        simulation.write(tmp_path, binary=False, validate=True)
