from datetime import datetime

import imod
from imod.mf6.dis import StructuredDiscretization
from imod.mf6.npf import NodePropertyFlow
from imod.mf6.write_context import WriteContext
from imod.prepare.topsystem.allocation import ALLOCATION_OPTION
from imod.prepare.topsystem.conductance import DISTRIBUTING_OPTION


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

    # write the packages for write validation
    write_context = WriteContext(simulation_directory=tmp_path, use_binary=False)
    ghb._write("ghb", [1], write_context)


def test_from_imod5_planar(imod5_dataset_periods, tmp_path):
    period_data = imod5_dataset_periods[1]
    imod5_dataset = imod5_dataset_periods[0]
    target_dis = StructuredDiscretization.from_imod5_data(imod5_dataset, validate=False)
    target_npf = NodePropertyFlow.from_imod5_data(
        imod5_dataset, target_dis.dataset["idomain"]
    )
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

    # write the packages for write validation
    write_context = WriteContext(simulation_directory=tmp_path, use_binary=False)
    ghb._write("ghb", [1], write_context)
