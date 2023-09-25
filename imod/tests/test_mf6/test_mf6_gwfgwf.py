import textwrap

import numpy as np
import pytest

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
