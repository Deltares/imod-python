from pathlib import Path
import numpy as np
import pytest
import xarray as xr
from imod.pkg import ConstantHead


@pytest.fixture(scope="module")
def constanthead(request):
    layer = np.arange(1, 4)
    y = np.arange(4.5, 0.0, -1.0)
    x = np.arange(0.5, 5.0, 1.0)
    head = xr.DataArray(
        np.full((3, 5, 5), 1.0),
        coords={"layer": layer, "y": y, "x": x, "dx": 1.0, "dy": -1.0},
        dims=("layer", "y", "x"),
    )

    chd = ConstantHead(head_start=head, head_end=head.copy(), concentration=head.copy())
    return chd

def test_render(constanthead):
    chd = constanthead
    directory = Path(".")

    compare = (
        "\n"
        "    shead_p?_s1_l1 = head_start_l1.idf\n"
        "    shead_p?_s1_l2 = head_start_l2.idf\n"
        "    shead_p?_s1_l3 = head_start_l3.idf\n"
        "    ehead_p?_s1_l1 = head_end_l1.idf\n"
        "    ehead_p?_s1_l2 = head_end_l2.idf\n"
        "    ehead_p?_s1_l3 = head_end_l3.idf"
    )

    assert chd._render(directory, globaltimes=["?"], system_index=1) == compare
