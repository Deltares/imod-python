import pytest
import pandas as pd
import numpy as np
import pathlib
import os

from imod.flow import Well


def test_wel(well_df, three_days, get_render_dict):
    well = Well(**well_df)

    directory = str(pathlib.Path(".").resolve())

    nlayer = 3
    times = three_days

    time_composed = {
        "1": "2018-01-01 00:00:00",
        "2": "2018-01-02 00:00:00",
        "3": "2018-01-03 00:00:00",
    }

    to_render = get_render_dict(well, directory, times, nlayer)
    to_render["n_entry"] = 1
    to_render["times"] = time_composed

    compare = (
        "0003, (wel), 1, Well, ['rate']\n"
        "2018-01-01 00:00:00\n"
        "001, 001\n"
        f"1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}wel_20180101000000_l2.ipf\n"
        "2018-01-02 00:00:00\n"
        "001, 001\n"
        f"1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}wel_20180102000000_l2.ipf\n"
        "2018-01-03 00:00:00\n"
        "001, 001\n"
        f"1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}wel_20180103000000_l2.ipf"
    )
    rendered = well._render_projectfile(**to_render)

    assert compare == rendered
