from imod.wq import MassLoading
import numpy as np
import pandas as pd
import xarray as xr


def test_render_mal__scalar():
    mal = MassLoading(concentration=1.0)
    actual = mal._render_ssm(".", ["?"], nlayer=3)
    compare = "\n    cmal_t1_p?_l? = 1.0"
    assert actual == compare


def test_render_mal__array():
    time = pd.date_range("2000-01-01", "2000-01-03", freq="D")
    conc = xr.DataArray(
        np.ones((3, 5, 3, 4)),
        {
            "time": time,
            "layer": [1, 2, 3, 4, 5],
            "y": [0.5, 1.5, 2.5],
            "x": [0.5, 1.5, 2.5, 3.5],
        },
        dims=("time", "layer", "y", "x"),
    )
    mal = MassLoading(concentration=conc)
    actual = mal._render_ssm("mal", globaltimes=time, nlayer=5)
    compare = """
    cmal_t1_p1_l$ = mal/concentration_20000101000000_l$.idf
    cmal_t1_p2_l$ = mal/concentration_20000102000000_l$.idf
    cmal_t1_p3_l$ = mal/concentration_20000103000000_l$.idf"""
    assert actual == compare
