from copy import deepcopy
from pathlib import Path

import pytest

import imod
from imod.tests.fixtures.mf6_modelrun_fixture import assert_simulation_can_run


def test_mask_simulation(
    tmp_path: Path,
    flow_transport_simulation: imod.mf6.Modflow6Simulation,
):
    mask = deepcopy(flow_transport_simulation["flow"].domain)
    mask.loc[1, 0.5, 15] = 0
    flow_transport_simulation.mask_all_models(mask)
    assert flow_transport_simulation["flow"].domain.sel({"layer":1, "y":0.5, "x":15}) == 0
    assert flow_transport_simulation["tpt_a"].domain.sel({"layer":1, "y":0.5, "x":15})== 0    
    assert flow_transport_simulation["tpt_b"].domain.sel({"layer":1, "y":0.5, "x":15}) == 0 
    assert flow_transport_simulation["tpt_c"].domain.sel({"layer":1, "y":0.5, "x":15}) == 0    
    assert flow_transport_simulation["tpt_d"].domain.sel({"layer":1, "y":0.5, "x":15})== 0    

    assert_simulation_can_run(flow_transport_simulation, "dis", tmp_path)

def test_mask_split_simulation(
    split_transient_twri_model: imod.mf6.Modflow6Simulation,
):
    mask = deepcopy(split_transient_twri_model["GWF_1_2"].domain)
    mask.loc[2, 57500, 12500] = 0
    with pytest.raises(ValueError , match="Apply masking before splitting."):
        split_transient_twri_model.mask_all_models(mask)
