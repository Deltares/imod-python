import textwrap

import numpy as np
import pytest
import xarray as xr

import imod


@pytest.mark.usefixtures("transient_twri_model")
def test_render_exchange_file(tmp_path):
    # arrange
    cell_id1 = np.array([(1, 1), (2, 1), (3, 1)], dtype="i,i")
    cell_id2 = np.array([(1, 2), (2, 2), (3, 2)], dtype="i,i")
    layer = np.array([12, 13, 14])
    gwfgwf = imod.mf6.GWFGWF("name1", "name2", cell_id1, cell_id2, layer)

    # act
    actual = gwfgwf.render(tmp_path, "gwfgwf", [], False)

    # assert

    expected = textwrap.dedent(
        """\
    begin options
    end options

    begin dimensions
      nexg 3
    end dimensions

    begin exchangedata
    12 1 1 12 1 2
    13 2 1 13 2 2
    14 3 1 14 3 2

    end exchangedata
    """
    )

    assert actual == expected


def test_error_clip():
    # arrange
    cell_id1 = np.array([(1, 1), (2, 1), (3, 1)], dtype="i,i")
    cell_id2 = np.array([(1, 2), (2, 2), (3, 2)], dtype="i,i")
    layer = np.array([12, 13, 14])
    gwfgwf = imod.mf6.GWFGWF("name1", "name2", cell_id1, cell_id2, layer)

    # test error
    with pytest.raises(NotImplementedError):
        gwfgwf.clip_box(0, 100, 1, 12, 0, 100, 0, 100)


def test_error_regrid():
    # arrange
    cell_id1 = np.array([(1, 1), (2, 1), (3, 1)], dtype="i,i")
    cell_id2 = np.array([(1, 2), (2, 2), (3, 2)], dtype="i,i")
    layer = np.array([12, 13, 14])
    gwfgwf = imod.mf6.GWFGWF("name1", "name2", cell_id1, cell_id2, layer)

    dims = ("layer", "y", "x")
    coords = {"layer": [1, 2, 3], "y": [109, 11], "x": [12, 13], "dx": 1, "dy": 1}
    target_grid = xr.DataArray(np.ones((3, 2, 2), dtype=int), coords=coords, dims=dims)

    # test error
    assert not gwfgwf.is_regridding_supported()
    with pytest.raises(NotImplementedError):
        gwfgwf.regrid_like(target_grid)
