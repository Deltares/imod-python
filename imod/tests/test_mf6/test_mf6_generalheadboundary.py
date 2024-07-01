from datetime import datetime

import imod
from imod.formats.prj.prj import open_projectfile_data
from imod.mf6.dis import StructuredDiscretization
from imod.mf6.write_context import WriteContext


def test_from_imod5(tmp_path):
    testdir = (
        "D:\\dev\\imod_python-gh\\imod-python\\imod\\tests\\imod5_data\\iMOD5_model.prj"
    )
    imod5_dataset = open_projectfile_data(testdir)
    period_data = imod5_dataset[1]
    imod5_dataset = imod5_dataset[0]
    target_dis = StructuredDiscretization.from_imod5_data(imod5_dataset, validate=False)

    ghb = imod.mf6.GeneralHeadBoundary.from_imod5_data(
        "ghb",
        imod5_dataset,
        period_data,
        target_dis,
        time_min=datetime(2002, 2, 2),
        time_max=datetime(2022, 2, 2),
        regridder_types=None,
    )

    assert isinstance(ghb, imod.mf6.GeneralHeadBoundary)

    # write the packages for write validation
    write_context = WriteContext(simulation_directory=tmp_path, use_binary=False)
    ghb.write("ghb", [1], write_context)
