import textwrap
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import imod


@pytest.fixture(scope="function")
def sample_gwfgwf_structured():
    ds = xr.Dataset()
    ds["cell_id1"] = xr.DataArray(
        [[1, 1], [2, 1], [3, 1]],
        dims=("index", "cell_dims1"),
        coords={"cell_dims1": ["row_1", "column_1"]},
    )
    ds["cell_id2"] = xr.DataArray(
        [[1, 2], [2, 2], [3, 2]],
        dims=("index", "cell_dims2"),
        coords={"cell_dims2": ["row_2", "column_2"]},
    )
    ds["layer"] = xr.DataArray([12, 13, 14], dims="layer")
    ds["cl1"] = xr.DataArray(np.ones(3), dims="index")
    ds["cl2"] = xr.DataArray(np.ones(3), dims="index")
    ds["hwva"] = ds["cl1"] + ds["cl2"]

    ds = ds.stack(cell_id=("layer", "index"), create_index=False).reset_coords()
    ds["cell_id1"] = ds["cell_id1"].T
    ds["cell_id2"] = ds["cell_id2"].T

    return imod.mf6.GWFGWF("name1", "name2", **ds)


@pytest.fixture(scope="function")
def sample_gwfgwf_unstructured():
    ds = xr.Dataset()
    ds["cell_id1"] = xr.DataArray([[1], [2], [3]], dims=("index", "cell_dims1"))
    ds["cell_id2"] = xr.DataArray([[2], [2], [4]], dims=("index", "cell_dims2"))
    ds["layer"] = xr.DataArray([12, 13, 14], dims="layer")
    ds["cl1"] = xr.DataArray(np.ones(3), dims="index")
    ds["cl2"] = xr.DataArray(np.ones(3), dims="index")
    ds["hwva"] = ds["cl1"] + ds["cl2"]

    ds["angldegx"] = xr.DataArray(np.ones(3), dims="index") * 90.0
    ds["cdist"] = xr.DataArray(np.ones(3), dims="index") * 2.0

    ds = ds.stack(cell_id=("layer", "index"), create_index=False).reset_coords()
    ds["cell_id1"] = ds["cell_id1"].T
    ds["cell_id2"] = ds["cell_id2"].T

    return imod.mf6.GWFGWF("name1", "name2", **ds)


class TestGwfgwf:
    @pytest.mark.usefixtures("sample_gwfgwf_structured")
    def test_render_exchange_file_structured(
        self, sample_gwfgwf_structured: imod.mf6.GWFGWF, tmp_path: Path
    ):
        # act
        actual = sample_gwfgwf_structured.render(tmp_path, "gwfgwf", [], False)

        # assert

        expected = textwrap.dedent(
            """\
        begin options
          save_flows
        end options

        begin dimensions
          nexg 9
        end dimensions

        begin exchangedata
          12 1 1 12 1 2 1 1.0 1.0 2.0
          12 2 1 12 2 2 1 1.0 1.0 2.0
          12 3 1 12 3 2 1 1.0 1.0 2.0
          13 1 1 13 1 2 1 1.0 1.0 2.0
          13 2 1 13 2 2 1 1.0 1.0 2.0
          13 3 1 13 3 2 1 1.0 1.0 2.0
          14 1 1 14 1 2 1 1.0 1.0 2.0
          14 2 1 14 2 2 1 1.0 1.0 2.0
          14 3 1 14 3 2 1 1.0 1.0 2.0
        end exchangedata
        """
        )

        assert actual == expected

    @pytest.mark.usefixtures("sample_gwfgwf_unstructured")
    def test_render_exchange_file_unstructured(
        self, sample_gwfgwf_unstructured: imod.mf6.GWFGWF, tmp_path: Path
    ):
        # act
        actual = sample_gwfgwf_unstructured.render(tmp_path, "gwfgwf", [], False)

        # assert

        expected = textwrap.dedent(
            """\
        begin options
          save_flows
          auxiliary angldegx cdist
        end options

        begin dimensions
          nexg 9
        end dimensions

        begin exchangedata
          12 1 12 2 1 1.0 1.0 2.0 90.0 2.0
          12 2 12 2 1 1.0 1.0 2.0 90.0 2.0
          12 3 12 4 1 1.0 1.0 2.0 90.0 2.0
          13 1 13 2 1 1.0 1.0 2.0 90.0 2.0
          13 2 13 2 1 1.0 1.0 2.0 90.0 2.0
          13 3 13 4 1 1.0 1.0 2.0 90.0 2.0
          14 1 14 2 1 1.0 1.0 2.0 90.0 2.0
          14 2 14 2 1 1.0 1.0 2.0 90.0 2.0
          14 3 14 4 1 1.0 1.0 2.0 90.0 2.0
        end exchangedata
        """
        )

        assert actual == expected

    @pytest.mark.usefixtures("sample_gwfgwf_structured")
    def test_error_clip(self, sample_gwfgwf_structured: imod.mf6.GWFGWF):
        # test error
        with pytest.raises(NotImplementedError):
            sample_gwfgwf_structured.clip_box(0, 100, 1, 12, 0, 100, 0, 100)

    @pytest.mark.usefixtures("sample_gwfgwf_structured")
    def test_error_regrid(self, sample_gwfgwf_structured: imod.mf6.GWFGWF):
        assert not sample_gwfgwf_structured.is_regridding_supported()
