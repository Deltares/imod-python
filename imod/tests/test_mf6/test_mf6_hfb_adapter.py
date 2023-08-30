import pathlib
import textwrap

import numpy as np
import xarray as xr
from pytest_cases import parametrize_with_cases

from imod.mf6.mf6_hfb_adapter import Mf6HorizontalFlowBarrier
from imod.mf6.write_context import WriteContext

class GridBarriers:
    def case_structured(self):
        row_1 = [1, 2]
        column_1 = [1, 1]
        row_2 = [1, 2]
        column_2 = [2, 2]
        layer = [1, 2, 3]

        cell_indices = np.arange(len(row_1)) + 1

        barrier = xr.Dataset()
        barrier["cell_id1"] = xr.DataArray(
            [row_1, column_1],
            coords={"cell_idx": cell_indices, "cell_dims1": ["row_1", "column_1"]},
        )
        barrier["cell_id2"] = xr.DataArray(
            [row_2, column_2],
            coords={"cell_idx": cell_indices, "cell_dims2": ["row_2", "column_2"]},
        )
        barrier["hydraulic_characteristic"] = xr.DataArray(
            np.full((len(layer), len(cell_indices)), 1e-3),
            coords={"layer": layer, "cell_idx": cell_indices},
        )
        barrier = (
            barrier.stack(cell_id=("layer", "cell_idx"), create_index=False)
            .drop_vars("cell_idx")
            .reset_coords()
        )

        return barrier

    def case_untructured(self):
        cell2d_id1 = [1, 2]
        cell2d_id2 = [3, 4]
        layer = [1, 2, 3]

        cell_indices = np.arange(len(cell2d_id1)) + 1

        barrier = xr.Dataset()
        barrier["cell_id1"] = xr.DataArray(
            np.array([cell2d_id1]).T,
            coords={"cell_idx": cell_indices, "cell_dims1": ["cell2d_1"]},
        )
        barrier["cell_id2"] = xr.DataArray(
            np.array([cell2d_id2]).T,
            coords={"cell_idx": cell_indices, "cell_dims2": ["cell2d_2"]},
        )
        barrier["hydraulic_characteristic"] = xr.DataArray(
            np.full((len(layer), len(cell_indices)), 1e-3),
            coords={"layer": layer, "cell_idx": cell_indices},
        )

        barrier = (
            barrier.stack(cell_id=("layer", "cell_idx"), create_index=False)
            .drop_vars("cell_idx")
            .reset_coords()
        )

        return barrier


@parametrize_with_cases("barrier", cases=GridBarriers)
def test_hfb_render(barrier):
    # Arrange
    hfb = Mf6HorizontalFlowBarrier(**barrier)

    expected = textwrap.dedent(
        """\
        begin options

        end options

        begin dimensions
          maxhfb 6
        end dimensions

        begin period 1
          open/close mymodel/hfb/hfb.dat
        end period"""
    )

    # Act
    directory = pathlib.Path("mymodel")
    actual = hfb.render(directory, "hfb", None, False)

    # Assert
    assert actual == expected


@parametrize_with_cases("barrier", cases=GridBarriers)
def test_hfb_writing_one_layer__unstructured(barrier, tmp_path):
    # Arrange
    hfb = Mf6HorizontalFlowBarrier(**barrier)
    write_context = WriteContext(tmp_path)

    expected_hfb_data = np.row_stack(
        (
            barrier["layer"],
            barrier["cell_id1"],
            barrier["layer"],
            barrier["cell_id2"],
            barrier["hydraulic_characteristic"],
        )
    ).T

    # Act
    hfb.write( "hfb", None, write_context)

    # Assert
    data = np.loadtxt(tmp_path / "hfb" / "hfb.dat")
    np.testing.assert_almost_equal(data, expected_hfb_data)
