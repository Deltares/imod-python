import pathlib

from imod.flow import EvapoTranspiration


def test_evapotranspiration_no_time(three_days):
    # Resolve in advance, so that comparisons have the same directory
    # See e.g. https://github.com/omarkohl/pytest-datafiles/issues/6
    directory = str(pathlib.Path(".").resolve())

    evt = EvapoTranspiration(rate=10.0, top_elevation=0.0, extinction_depth=1.0)

    nlayer = 3  # Model has three layers, but should only render 1 layer for Recharge!
    times = three_days

    time_composed = {
        "1": "2018-01-01 00:00:00",
        "2": "2018-01-02 00:00:00",
        "3": "2018-01-03 00:00:00",
        "steady-state": "steady-state",
    }

    composition = evt.compose(
        directory,
        times,
        nlayer,
    )

    to_render = dict(
        pkg_id=evt._pkg_id,
        name=evt.__class__.__name__,
        variable_order=evt._variable_order,
        package_data=composition[evt._pkg_id],
        n_entry=1,
        times=time_composed,
    )

    compare = (
        "0001, (evt), 1, EvapoTranspiration, ['rate', 'top_elevation', 'extinction_depth']\n"
        "steady-state\n"
        "003, 001\n"
        '1, 1, 001, 1.000, 0.000, 10.0, ""\n'
        '1, 1, 001, 1.000, 0.000, 0.0, ""\n'
        '1, 1, 001, 1.000, 0.000, 1.0, ""'
    )
    rendered = evt._render_projectfile(**to_render)

    assert compare == rendered
