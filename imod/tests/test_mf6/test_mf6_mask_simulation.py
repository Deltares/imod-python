from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import imod
from imod.tests.fixtures.mf6_modelrun_fixture import assert_simulation_can_run


@pytest.mark.parametrize("ignore_time", [True, False])
def test_mask_simulation(
    tmp_path: Path,
    flow_transport_simulation: imod.mf6.Modflow6Simulation,
    ignore_time: bool,
):
    mask = deepcopy(flow_transport_simulation["flow"].domain)
    mask.loc[1, 0.5, 15] = 0
    flow_transport_simulation.mask_all_models(mask, ignore_time_purge_empty=ignore_time)
    assert (
        flow_transport_simulation["flow"].domain.sel({"layer": 1, "y": 0.5, "x": 15})
        == 0
    )
    assert (
        flow_transport_simulation["tpt_a"].domain.sel({"layer": 1, "y": 0.5, "x": 15})
        == 0
    )
    assert (
        flow_transport_simulation["tpt_b"].domain.sel({"layer": 1, "y": 0.5, "x": 15})
        == 0
    )
    assert (
        flow_transport_simulation["tpt_c"].domain.sel({"layer": 1, "y": 0.5, "x": 15})
        == 0
    )
    assert (
        flow_transport_simulation["tpt_d"].domain.sel({"layer": 1, "y": 0.5, "x": 15})
        == 0
    )

    assert_simulation_can_run(flow_transport_simulation, tmp_path)


def test_mask_split_simulation(
    split_transient_twri_model: imod.mf6.Modflow6Simulation,
):
    mask = deepcopy(split_transient_twri_model["GWF_1_2"].domain)
    mask.loc[2, 57500, 12500] = 0
    with pytest.raises(ValueError, match="Apply masking before splitting."):
        split_transient_twri_model.mask_all_models(mask)


def test_mask_simulation_different_domains(
    structured_flow_simulation_2_flow_models: imod.mf6.Modflow6Simulation,
    tmp_path: Path,
):
    model_2 = structured_flow_simulation_2_flow_models["flow_copy"]

    # initially the simulation has 2 models, each with the same domain.
    # we'll regrid one of them so that they no longer use the same domain
    y = np.arange(10, -1, -1)
    x = np.arange(0, 11, 1)

    nlayer = 3

    shape = nlayer, len(x), len(y)
    dims = ("layer", "y", "x")
    layer = np.arange(1, 4)

    coords = {"layer": layer, "y": y, "x": x, "dx": 1, "dy": 1}

    new_grid = xr.DataArray(np.ones(shape, dtype=float) * 1, coords=coords, dims=dims)

    model_2_refined = model_2.regrid_like(new_grid)
    structured_flow_simulation_2_flow_models["flow_copy"] = model_2_refined
    mask = new_grid

    # with this setup, masking should no longer be allowed
    with pytest.raises(
        ValueError, match="when all the models in the simulation use the same grid."
    ):
        structured_flow_simulation_2_flow_models.mask_all_models(mask)
