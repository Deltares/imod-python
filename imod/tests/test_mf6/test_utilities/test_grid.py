import numpy as np

from imod.mf6.utilities.grid import get_smallest_target_grid
from imod.util.spatial import empty_2d


def test_get_smallest_target_grid():
    # Three grids with aligned cell edges at each 100m
    grid1 = empty_2d(dx=25.0, xmin=100.0, xmax=300.0, dy=-25.0, ymin=100.0, ymax=300.0)
    grid2 = empty_2d(dx=50.0, xmin=0.0, xmax=200.0, dy=-50.0, ymin=0.0, ymax=200.0)
    grid3 = empty_2d(dx=20.0, xmin=0.0, xmax=400.0, dy=-20.0, ymin=0.0, ymax=400.0)

    actual = get_smallest_target_grid(grid1, grid2, grid3)

    assert actual.coords["dx"] == 20.0
    assert actual.coords["dy"] == -20.0
    assert np.all(actual.coords["x"].values == [110.0, 130.0, 150.0, 170.0, 190.0])
    assert np.all(actual.coords["y"].values == [190.0, 170.0, 150.0, 130.0, 110.0])

    # Two grids with barely aligned cell edges.
    grid1 = empty_2d(dx=50.0, xmin=110.0, xmax=220.0, dy=-50.0, ymin=110.0, ymax=220.0)
    grid2 = empty_2d(dx=20.0, xmin=0.0, xmax=400.0, dy=-20.0, ymin=0.0, ymax=400.0)

    actual = get_smallest_target_grid(grid1, grid2)

    assert actual.coords["dx"] == 20.0
    assert actual.coords["dy"] == -20.0
    assert np.all(actual.coords["x"].values == [120.0, 140.0, 160.0, 180.0, 200.0])
    assert np.all(actual.coords["y"].values == [210.0, 190.0, 170.0, 150.0, 130.0])
