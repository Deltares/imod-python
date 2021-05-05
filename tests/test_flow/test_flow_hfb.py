from imod.flow.hfb import HorizontalFlowBarrier
import pathlib
import pandas as pd
import os
import textwrap


def test_horizontal_flow_barrier_render(
    basic_dis, get_render_dict, horizontal_flow_barrier_gdf
):
    # Resolve in advance, so that comparisons have the same directory
    # See e.g. https://github.com/omarkohl/pytest-datafiles/issues/6
    directory = pathlib.Path(".").resolve()

    ibound, _, _ = basic_dis
    horizontal_flow_barrier = HorizontalFlowBarrier(**horizontal_flow_barrier_gdf)
    nlayer = len(ibound["layer"])

    to_render = get_render_dict(horizontal_flow_barrier, directory, None, nlayer)
    to_render["n_entry"] = len(pd.unique(horizontal_flow_barrier["layer"]))

    # {directory.stem} because this is taken in both compose as well as save
    compare = textwrap.dedent(f"""\
        0001, (hfb), 1, HorizontalFlowBarrier, ['resistance']
        001, 002
        1, 2, 003, 100.0, 0.000, -9999., {directory}{os.sep}{directory.stem}_l3.gen
        1, 2, 004, 100.0, 0.000, -9999., {directory}{os.sep}{directory.stem}_l4.gen"""
    )

    rendered = horizontal_flow_barrier._render_projectfile(**to_render)

    assert rendered == compare
