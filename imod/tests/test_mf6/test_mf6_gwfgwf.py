import textwrap
from pathlib import Path

import numpy as np
import pytest

import imod


@pytest.fixture(scope="function")
def sample_gwfgwf_structured():
    cell_id1 = np.array([(1, 1), (2, 1), (3, 1)], dtype="i,i")
    cell_id2 = np.array([(1, 2), (2, 2), (3, 2)], dtype="i,i")
    layer = np.array([12, 13, 14])
    return imod.mf6.GWFGWF("name1", "name2", cell_id1, cell_id2, layer)


@pytest.fixture(scope="function")
def sample_gwfgwf_unstructured():
    cell_id1 = np.array([(1), (2), (31)], dtype="i")
    cell_id2 = np.array([(2), (2), (4)], dtype="i")
    layer = np.array([12, 13, 14])
    return imod.mf6.GWFGWF("name1", "name2", cell_id1, cell_id2, layer)


class testGwfgwf:
    @pytest.mark.usefixtures("sample_gwfgwf_structured")
    def test_render_exchange_file_structured(
        sample_gwfgwf_structured: imod.mf6.GWFGWF, tmp_path: Path
    ):
        # act
        actual = sample_gwfgwf_structured.render(tmp_path, "gwfgwf", [], False)

        # assert

        expected = textwrap.dedent(
            """\
        begin options
        end options

        begin dimensions
        nexg 3
        end dimensions

        begin exchangedata
        12 1 1 12 1 2 1.0 1.0 1.0 1.0
        13 2 1 13 2 2 1.0 1.0 1.0 1.0
        14 3 1 14 3 2 1.0 1.0 1.0 1.0

        end exchangedata
        """
        )

        assert actual == expected

    @pytest.mark.usefixtures("sample_gwfgwf_unstructured")
    def test_render_exchange_file_unstructured(
        sample_gwfgwf_unstructured: imod.mf6.GWFGWF, tmp_path: Path
    ):
        # act
        actual = sample_gwfgwf_unstructured.render(tmp_path, "gwfgwf", [], False)

        # assert

        expected = textwrap.dedent(
            """\
        begin options
        end options

        begin dimensions
        nexg 3
        end dimensions

        begin exchangedata
        12 1 12 2 1.0 1.0 1.0 1.0
        13 2 13 2 1.0 1.0 1.0 1.0
        14 31 14 4 1.0 1.0 1.0 1.0

        end exchangedata
        """
        )

        assert actual == expected

    @pytest.mark.usefixtures("sample_gwfgwf_structured")
    def test_error_clip(sample_gwfgwf_structured: imod.mf6.GWFGWF):
        # test error
        with pytest.raises(NotImplementedError):
            sample_gwfgwf_structured.clip_box(0, 100, 1, 12, 0, 100, 0, 100)

    @pytest.mark.usefixtures("sample_gwfgwf_structured")
    def test_error_regrid(sample_gwfgwf_structured: imod.mf6.GWFGWF):
        assert not sample_gwfgwf_structured.is_regridding_supported()
