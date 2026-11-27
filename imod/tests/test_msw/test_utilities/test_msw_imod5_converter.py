import numpy as np

import imod
from imod.msw.utilities.imod5_converter import get_cell_area_from_imod5_data


def test_get_cell_area_from_imod5_data__urban_area_too_large():
    grid = imod.util.empty_2d(
        dx=10.0, dy=10.0, xmin=0.0, xmax=20.0, ymin=0.0, ymax=10.0
    )
    # First cell: fully urban. Second cell: urban area too large for the cell
    # area minus the wetted area.
    imod5_cap = {
        "boundary": grid.fillna(1).astype(int),
        "wetted_area": grid.copy(data=np.array([[0.0, 50.0]])),
        "urban_area": grid.copy(data=np.array([[100.0, 80.0]])),
    }

    area = get_cell_area_from_imod5_data(imod5_cap)

    rural_area = area.sel(subunit=0).values.ravel()
    urban_area = area.sel(subunit=1).values.ravel()
    np.testing.assert_allclose(rural_area, [0.0, 50.0])
    np.testing.assert_allclose(urban_area, [100.0, 0.0])
