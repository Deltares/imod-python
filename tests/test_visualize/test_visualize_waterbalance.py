import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import imod


@pytest.fixture(scope="module")
def waterbalance_df():
    df = pd.DataFrame(
        np.random.rand(10, 6) * 100.0,
        index=pd.date_range("2001-01-01", "2001-01-10"),
        columns=["P", "ET", "Seepage", "Riv", "Wel", "Drainage"],
    )
    return df


def test_waterbalance_barchart(waterbalance_df):
    df = waterbalance_df
    inflows = ["P", "Riv", "Seepage"]
    outflows = ["ET", "Wel", "Drainage"]

    # Without optional arguments
    imod.visualize.waterbalance_barchart(
        df=df, inflows=inflows, outflows=outflows,
    )

    # With optional arguments
    _, ax = plt.subplots()
    df["date"] = df.index
    colors = ["#b2182b", "#ef8a62", "#fddbc7", "#d1e5f0", "#67a9cf", "#2166ac"]
    kwargs = dict(
        df=df,
        inflows=inflows,
        outflows=outflows,
        datecolumn="date",
        format="%Y-%m-%d",
        ax=ax,
        unit="m3/d",
        colors=colors,
    )
    imod.visualize.waterbalance_barchart(**kwargs)

    # Errors
    with pytest.raises(ValueError):
        faulty_kwargs = kwargs.copy()
        faulty_kwargs["datecolumn"] = "datetime"
        imod.visualize.waterbalance_barchart(**faulty_kwargs)
    with pytest.raises(ValueError):
        faulty_kwargs = kwargs.copy()
        faulty_kwargs["inflows"] = ["Precip", "ET", "Seepage"]
        imod.visualize.waterbalance_barchart(**faulty_kwargs)
    with pytest.raises(ValueError):
        faulty_kwargs = kwargs.copy()
        faulty_kwargs["colors"] = colors[:-1]
        imod.visualize.waterbalance_barchart(**faulty_kwargs)
