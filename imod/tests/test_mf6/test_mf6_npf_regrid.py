import numpy as np
import pytest
import xarray as xr
import xugrid as xu

import imod


def test_regrid(tmp_path):
    grid = imod.data.circle()
    nlayer = 15

    nface = grid.n_face
    layer = np.arange(nlayer, dtype=int) + 1
    k_value = 10.0
    idomain = xu.UgridDataArray(
        xr.DataArray(
            np.ones((nlayer, nface), dtype=np.int32),
            coords={"layer": layer},
            dims=["layer", grid.face_dimension],
        ),
        grid=grid,
    )
    k = xu.full_like(idomain, k_value, dtype=float)

    npf = imod.mf6.NodePropertyFlow(
        icelltype=idomain,
        k=k,
        variable_vertical_conductance=True,
        dewatered=True,
        perched=True,
        save_flows=True,
        alternative_cell_averaging="AMT-HMK",
    )

    new_npf = npf.regrid_like(k)

    # check the rendered versions are the same, they contain the options
    new_rendered = new_npf.render(tmp_path, "regridded", None, False)
    original_rendered = npf.render(tmp_path, "original", None, False)

    new_rendered = new_rendered.replace("regridded", "original")
    assert new_rendered == original_rendered

    # check the arrays
    k_new = new_npf.dataset["k"]
    k_diff = k_new - k
    max_diff = k_diff.max().values[()]
    min_diff = k_diff.min().values[()]
    abs_tol = 1e-13

    assert abs(min_diff) < abs_tol and max_diff < abs_tol
