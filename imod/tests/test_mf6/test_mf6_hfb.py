import geopandas as gpd
import numpy as np
import pytest
import shapely
import xugrid as xu
import textwrap
import pathlib

import imod


def get_hfb_data_one_layer(grid_xy):
    """
    Line at cell edges of unstructured flow model
    """

    x = [-1.0, 1.0, 1.0, 3.0, 3.0, 5.0]
    y = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    indices = np.repeat(np.arange(3), 2)
    linestrings = shapely.linestrings(x, y, indices=indices)
    lines = gpd.GeoDataFrame(geometry=linestrings)
    lines["linedata"] = 10.0

    uda, _ = xu.snap_to_grid(lines, grid_xy, 0.2)

    line_as_dataarray = uda["linedata"]
    line_as_dataarray = line_as_dataarray.expand_dims("layer")
    line_as_dataarray = line_as_dataarray.assign_coords(layer=[1])

    return line_as_dataarray


@pytest.mark.parametrize(
    "hfb_class",
    [
        imod.mf6.HorizontalFlowBarrierResistance,
        imod.mf6.HorizontalFlowBarrierMultiplier,
        imod.mf6.HorizontalFlowBarrierHydraulicCharacteristic,
    ],
)
def test_hfb_render_one_layer(
    hfb_class, unstructured_flow_model,
):
    # Arrange
    idomain = unstructured_flow_model["disv"]["idomain"]
    hfb_data_one_layer = get_hfb_data_one_layer(idomain.sel(layer=1))
    hfb = hfb_class(hfb_data_one_layer, idomain)

    expected = textwrap.dedent(
        """\
        begin options

        end options

        begin dimensions
          maxhfb 3
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


@pytest.mark.parametrize(
    "hfb_specialization",
    [
        (imod.mf6.HorizontalFlowBarrierResistance, 0.1),
        (imod.mf6.HorizontalFlowBarrierMultiplier, -10.0),
        (imod.mf6.HorizontalFlowBarrierHydraulicCharacteristic, 10.0),
    ],
)
def test_hfb_writing_one_layer(
    hfb_specialization, tmp_path, unstructured_flow_model,
):
    hfb_class, expected_value = hfb_specialization
    # Arrange
    idomain = unstructured_flow_model["disv"]["idomain"]
    hfb_data_one_layer = get_hfb_data_one_layer(idomain.sel(layer=1))
    hfb = hfb_class(hfb_data_one_layer, idomain)

    expected_hfb_data = np.array(
        [
            [1.0, 13.0, 1.0, 19.0, expected_value],
            [1.0, 14.0, 1.0, 20.0, expected_value],
            [1.0, 15.0, 1.0, 21.0, expected_value],
        ]
    )

    # Act
    hfb.write(tmp_path, "hfb", None, False)

    # Assert
    data = np.loadtxt(tmp_path / "hfb" / "hfb.dat")
    np.testing.assert_almost_equal(data, expected_hfb_data)
