
from pathlib import Path

import pandas as pd
import imod
from imod.formats.prj.prj import open_projectfile_data
from imod.mf6.oc import OutputControl
from imod.mf6.regrid.regrid_schemes import DiscretizationRegridMethod, NodePropertyFlowRegridMethod, StorageCoefficientRegridMethod
from imod.mf6.simulation import Modflow6Simulation
from imod.prepare.topsystem.default_allocation_methods import SimulationAllocationOptions, SimulationDistributingOptions

def test_mf6_LHM( tmp_path):

    LHM_dir = Path("D:\\tmp\\LHM\\MODFLOW6_MODEL\\MODFLOW6_MODEL")
    data = open_projectfile_data(LHM_dir / "LHM4.3_stationair_gekalibreerd_bruinkool_fluxes_mf6.prj")

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
    simulation.write(tmp_path, binary=False, validate=True)    
    pass