from imod.flow import River
import pathlib
import os
import numpy as np
import xarray as xr
import pytest

@pytest.fixture(scope="module")
def river(basic_dis, two_days):
    ibound, _, _ = basic_dis
    x = ibound.x.values

    times = two_days 

    #Boundary_conditions
    trend = np.ones(times.shape)
    trend = np.cumsum(trend)

    head = ibound.where(ibound.x.isin([x[0], x[-1]]))
    head = xr.DataArray(trend, 
                coords = {"time" : times}, 
                dims = ["time"]
                ) * head

    return River(
        stage=head, 
        conductance=10., 
        bottom_elevation=head-1,
        infiltration_factor=1.)


def test_river(river, get_render_dict, two_days):
    # Resolve in advance, so that comparisons have the same directory
    # See e.g. https://github.com/omarkohl/pytest-datafiles/issues/6
    directory = str(pathlib.Path(".").resolve())

    nlayer = len(river["layer"])
    times = two_days 

    time_composed = {
        "1" : "2018-01-01 00:00:00",
        "2" : "2018-01-02 00:00:00",
        }

    to_render = get_render_dict(river, directory, times, nlayer)
    to_render["n_entry"] = nlayer
    to_render["times"] = time_composed

    compare = (
    "0002, (riv), 1, River, ['conductance', 'stage', 'bottom_elevation', 'infiltration_factor']\n"
    "2018-01-01 00:00:00\n"
    "004, 003\n"
    f'1, 1, 001, 1.000, 0.000, 10.0, ""\n'
    f'1, 1, 002, 1.000, 0.000, 10.0, ""\n'
    f'1, 1, 003, 1.000, 0.000, 10.0, ""\n'
    f"1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}stage_20180101000000_l1.idf\n"
    f"1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}stage_20180101000000_l2.idf\n"
    f"1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}stage_20180101000000_l3.idf\n"
    f"1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}bottom_elevation_20180101000000_l1.idf\n"
    f"1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}bottom_elevation_20180101000000_l2.idf\n"
    f"1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}bottom_elevation_20180101000000_l3.idf\n"
    f'1, 1, 001, 1.000, 0.000, 1.0, ""\n'
    f'1, 1, 002, 1.000, 0.000, 1.0, ""\n'
    f'1, 1, 003, 1.000, 0.000, 1.0, ""\n'
    "2018-01-02 00:00:00\n"
    "004, 003\n"
    f'1, 1, 001, 1.000, 0.000, 10.0, ""\n'
    f'1, 1, 002, 1.000, 0.000, 10.0, ""\n'
    f'1, 1, 003, 1.000, 0.000, 10.0, ""\n'
    f"1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}stage_20180102000000_l1.idf\n"
    f"1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}stage_20180102000000_l2.idf\n"
    f"1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}stage_20180102000000_l3.idf\n"
    f"1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}bottom_elevation_20180102000000_l1.idf\n"
    f"1, 2, 002, 1.000, 0.000, -9999., {directory}{os.sep}bottom_elevation_20180102000000_l2.idf\n"
    f"1, 2, 003, 1.000, 0.000, -9999., {directory}{os.sep}bottom_elevation_20180102000000_l3.idf\n"
    f'1, 1, 001, 1.000, 0.000, 1.0, ""\n'
    f'1, 1, 002, 1.000, 0.000, 1.0, ""\n'
    f'1, 1, 003, 1.000, 0.000, 1.0, ""'
    )
    rendered = river._render_projectfile(**to_render)

    assert compare == rendered