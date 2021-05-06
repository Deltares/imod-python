from imod.flow import Recharge
import pathlib
import os


def test_recharge_no_time(three_days):
    # Resolve in advance, so that comparisons have the same directory
    # See e.g. https://github.com/omarkohl/pytest-datafiles/issues/6
    directory = str(pathlib.Path(".").resolve())

    rch = Recharge(rate=10.0)

    nlayer = 3  # Model has three layers, but should only render 1 layer for Recharge!
    times = three_days

    time_composed = {
        "1": "2018-01-01 00:00:00",
        "2": "2018-01-02 00:00:00",
        "3": "2018-01-03 00:00:00",
    }

    composition = rch.compose(
        directory,
        times,
        nlayer,
    )

    to_render = dict(
        pkg_id=rch._pkg_id,
        name=rch.__class__.__name__,
        variable_order=rch._variable_order,
        package_data=composition[rch._pkg_id],
        n_entry=1,
        times=time_composed,
    )

    compare = (
        "0001, (rch), 1, Recharge, ['rate']\n"
        "2018-01-01 00:00:00\n"
        "001, 001\n"
        '1, 1, 001, 1.000, 0.000, 10.0, ""'
    )
    rendered = rch._render_projectfile(**to_render)

    assert compare == rendered
