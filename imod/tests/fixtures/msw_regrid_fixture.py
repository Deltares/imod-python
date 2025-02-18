import numpy as np
import xarray as xr

nan = np.nan


def get_3x3_area():
    subunit = [0, 1]
    x = [1.0, 2.0, 3.0]
    y = [3.0, 2.0, 1.0]
    dx = 1.0
    dy = -1.0
    area = xr.DataArray(
        np.array(
            [
                [[0.5, 0.5, 0.5], [nan, nan, nan], [1.0, 1.0, 1.0]],
                [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [nan, nan, nan]],
            ]
        ),
        dims=("subunit", "y", "x"),
        coords={"subunit": subunit, "y": y, "x": x, "dx": dx, "dy": dy},
    )

    return area


def get_5x5_new_grid():
    x = [1.0, 1.5, 2.0, 2.5, 3.0]
    y = [3.0, 2.5, 2.0, 1.5, 1.0]
    dx = 0.5
    dy = -0.5
    layer = [1, 2, 3]

    idomain = xr.DataArray(
        1,
        dims=("layer", "y", "x"),
        coords={"layer": layer, "y": y, "x": x, "dx": dx, "dy": dy},
    )
    return idomain
