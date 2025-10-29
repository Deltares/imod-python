from copy import deepcopy
from datetime import datetime

import pytest
import numpy as np
import xarray as xr

import imod
from imod.mf6.dis import StructuredDiscretization
from imod.mf6.npf import NodePropertyFlow
from imod.mf6.write_context import WriteContext
from imod.prepare.topsystem.allocation import ALLOCATION_OPTION
from imod.prepare.topsystem.conductance import DISTRIBUTING_OPTION


@pytest.mark.unittest_jit
def test_from_imod5_non_planar(imod5_dataset_periods, tmp_path):
    period_data = imod5_dataset_periods[1]
    imod5_dataset = imod5_dataset_periods[0]
    target_dis = StructuredDiscretization.from_imod5_data(imod5_dataset, validate=False)
    target_npf = NodePropertyFlow.from_imod5_data(
        imod5_dataset, target_dis.dataset["idomain"]
    )

    ghb = imod.mf6.GeneralHeadBoundary.from_imod5_data(
        "ghb",
        imod5_dataset,
        period_data,
        target_dis,
        target_npf,
        time_min=datetime(2002, 2, 2),
        time_max=datetime(2022, 2, 2),
        allocation_option=ALLOCATION_OPTION.at_elevation,
        distributing_option=DISTRIBUTING_OPTION.by_crosscut_thickness,
    )

    assert isinstance(ghb, imod.mf6.GeneralHeadBoundary)

    ghb_time = ghb.dataset.coords["time"].data
    expected_times = np.array(
        [
            np.datetime64("2002-02-02"),
            np.datetime64("2002-04-01"),
            np.datetime64("2002-10-01"),
        ]
    )
    np.testing.assert_array_equal(ghb_time, expected_times)
    ghb_repeat_stress = ghb.dataset["repeat_stress"].data
    assert np.all(ghb_repeat_stress[:, 1][::2] == np.datetime64("2002-04-01"))
    assert np.all(ghb_repeat_stress[:, 1][1::2] == np.datetime64("2002-10-01"))

    # write the packages for write validation
    write_context = WriteContext(simulation_directory=tmp_path, use_binary=False)
    ghb._write("ghb", [1], write_context)


@pytest.mark.unittest_jit
def test_from_imod5_and_cleanup_non_planar(imod5_dataset_periods, tmp_path):
    period_data = imod5_dataset_periods[1]
    imod5_dataset = imod5_dataset_periods[0]
    target_dis = StructuredDiscretization.from_imod5_data(imod5_dataset, validate=False)
    target_npf = NodePropertyFlow.from_imod5_data(
        imod5_dataset, target_dis.dataset["idomain"]
    )

    ghb = imod.mf6.GeneralHeadBoundary.from_imod5_data(
        "ghb",
        imod5_dataset,
        period_data,
        target_dis,
        target_npf,
        time_min=datetime(2002, 2, 2),
        time_max=datetime(2022, 2, 2),
        allocation_option=ALLOCATION_OPTION.at_elevation,
        distributing_option=DISTRIBUTING_OPTION.by_crosscut_thickness,
    )

    ghb.cleanup(target_dis)


@pytest.mark.unittest_jit
def test_from_imod5_constant(imod5_dataset_periods, tmp_path):
    period_data = imod5_dataset_periods[1]
    imod5_dataset = imod5_dataset_periods[0]
    target_dis = StructuredDiscretization.from_imod5_data(imod5_dataset, validate=False)
    target_npf = NodePropertyFlow.from_imod5_data(
        imod5_dataset, target_dis.dataset["idomain"]
    )
    original_ghb = deepcopy(imod5_dataset["ghb"])
    layer = imod5_dataset["ghb"]["conductance"].coords["layer"].data
    imod5_dataset["ghb"]["conductance"] = xr.DataArray(
        [1.0], coords={"layer": layer}, dims=("layer",)
    )
    ghb = imod.mf6.GeneralHeadBoundary.from_imod5_data(
        "ghb",
        imod5_dataset,
        period_data,
        target_dis,
        target_npf,
        time_min=datetime(2002, 2, 2),
        time_max=datetime(2022, 2, 2),
        allocation_option=ALLOCATION_OPTION.at_elevation,
        distributing_option=DISTRIBUTING_OPTION.by_crosscut_thickness,
    )

    assert isinstance(ghb, imod.mf6.GeneralHeadBoundary)

    ghb_time = ghb.dataset.coords["time"].data
    expected_times = np.array(
        [
            np.datetime64("2002-02-02"),
            np.datetime64("2002-04-01"),
            np.datetime64("2002-10-01"),
        ]
    )
    np.testing.assert_array_equal(ghb_time, expected_times)
    ghb_repeat_stress = ghb.dataset["repeat_stress"].data
    assert np.all(ghb_repeat_stress[:, 1][::2] == np.datetime64("2002-04-01"))
    assert np.all(ghb_repeat_stress[:, 1][1::2] == np.datetime64("2002-10-01"))

    # write the packages for write validation
    write_context = WriteContext(simulation_directory=tmp_path, use_binary=False)
    ghb._write("ghb", [1], write_context)

    # teardown
    imod5_dataset["ghb"] = original_ghb


@pytest.mark.unittest_jit
def test_from_imod5_and_cleanup_constant(imod5_dataset_periods, tmp_path):
    period_data = imod5_dataset_periods[1]
    imod5_dataset = imod5_dataset_periods[0]
    target_dis = StructuredDiscretization.from_imod5_data(imod5_dataset, validate=False)
    target_npf = NodePropertyFlow.from_imod5_data(
        imod5_dataset, target_dis.dataset["idomain"]
    )
    original_ghb = deepcopy(imod5_dataset["ghb"])
    layer = imod5_dataset["ghb"]["conductance"].coords["layer"].data
    imod5_dataset["ghb"]["conductance"] = xr.DataArray(
        [1.0], coords={"layer": layer}, dims=("layer",)
    )
    ghb = imod.mf6.GeneralHeadBoundary.from_imod5_data(
        "ghb",
        imod5_dataset,
        period_data,
        target_dis,
        target_npf,
        time_min=datetime(2002, 2, 2),
        time_max=datetime(2022, 2, 2),
        allocation_option=ALLOCATION_OPTION.at_elevation,
        distributing_option=DISTRIBUTING_OPTION.by_crosscut_thickness,
    )

    ghb.cleanup(target_dis)
    # teardown
    imod5_dataset["ghb"] = original_ghb


@pytest.mark.unittest_jit
def test_from_imod5_planar(imod5_dataset_periods, tmp_path):
    period_data = imod5_dataset_periods[1]
    imod5_dataset = imod5_dataset_periods[0]
    target_dis = StructuredDiscretization.from_imod5_data(imod5_dataset, validate=False)
    target_npf = NodePropertyFlow.from_imod5_data(
        imod5_dataset, target_dis.dataset["idomain"]
    )

    original_ghb = deepcopy(imod5_dataset["ghb"])
    imod5_dataset["ghb"]["conductance"] = imod5_dataset["ghb"][
        "conductance"
    ].assign_coords({"layer": [0]})
    imod5_dataset["ghb"]["head"] = imod5_dataset["ghb"]["head"].isel({"layer": 0})

    ghb = imod.mf6.GeneralHeadBoundary.from_imod5_data(
        "ghb",
        imod5_dataset,
        period_data,
        target_dis,
        target_npf,
        time_min=datetime(2002, 2, 2),
        time_max=datetime(2022, 2, 2),
        allocation_option=ALLOCATION_OPTION.at_elevation,
        distributing_option=DISTRIBUTING_OPTION.by_layer_thickness,
    )

    assert isinstance(ghb, imod.mf6.GeneralHeadBoundary)

    ghb_time = ghb.dataset.coords["time"].data
    expected_times = np.array(
        [
            np.datetime64("2002-02-02"),
            np.datetime64("2002-04-01"),
            np.datetime64("2002-10-01"),
        ]
    )
    np.testing.assert_array_equal(ghb_time, expected_times)
    ghb_repeat_stress = ghb.dataset["repeat_stress"].data
    assert np.all(ghb_repeat_stress[:, 1][::2] == np.datetime64("2002-04-01"))
    assert np.all(ghb_repeat_stress[:, 1][1::2] == np.datetime64("2002-10-01"))

    # write the packages for write validation
    write_context = WriteContext(simulation_directory=tmp_path, use_binary=False)
    ghb._write("ghb", [1], write_context)

    # teardown
    imod5_dataset["ghb"] = original_ghb


@pytest.mark.unittest_jit
def test_from_imod5_and_cleanup_planar(imod5_dataset_periods, tmp_path):
    period_data = imod5_dataset_periods[1]
    imod5_dataset = imod5_dataset_periods[0]
    target_dis = StructuredDiscretization.from_imod5_data(imod5_dataset, validate=False)
    target_npf = NodePropertyFlow.from_imod5_data(
        imod5_dataset, target_dis.dataset["idomain"]
    )

    original_ghb = deepcopy(imod5_dataset["ghb"])
    imod5_dataset["ghb"]["conductance"] = imod5_dataset["ghb"][
        "conductance"
    ].assign_coords({"layer": [0]})
    imod5_dataset["ghb"]["head"] = imod5_dataset["ghb"]["head"].isel({"layer": 0})

    ghb = imod.mf6.GeneralHeadBoundary.from_imod5_data(
        "ghb",
        imod5_dataset,
        period_data,
        target_dis,
        target_npf,
        time_min=datetime(2002, 2, 2),
        time_max=datetime(2022, 2, 2),
        allocation_option=ALLOCATION_OPTION.at_elevation,
        distributing_option=DISTRIBUTING_OPTION.by_layer_thickness,
    )

    ghb.cleanup(target_dis)

    # teardown
    imod5_dataset["ghb"] = original_ghb
