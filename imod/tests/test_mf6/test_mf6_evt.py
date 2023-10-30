import pathlib
import textwrap

import numpy as np
import pytest
import xarray as xr

import imod
from imod.schemata import ValidationError


@pytest.mark.usefixtures(
    "rate_fc",
    "elevation_fc",
    "concentration_fc",
    "proportion_rate_fc",
    "proportion_depth_fc",
)
def test_render(
    rate_fc, elevation_fc, concentration_fc, proportion_rate_fc, proportion_depth_fc
):
    # Arrange
    directory = pathlib.Path("mymodel")
    globaltimes = np.array(
        [
            "2000-01-01",
            "2000-01-02",
            "2000-01-03",
        ],
        dtype="datetime64[ns]",
    )

    evt = imod.mf6.Evapotranspiration(
        surface=elevation_fc,
        rate=rate_fc,
        depth=elevation_fc,
        proportion_rate=proportion_rate_fc,
        proportion_depth=proportion_depth_fc,
        concentration=concentration_fc,
        concentration_boundary_type="AUX",
    )
    # Act
    actual = evt.render(directory, "evt", globaltimes, False)

    # Assert
    expected = textwrap.dedent(
        """\
        begin options
          auxiliary salinity temperature
        end options

        begin dimensions
          maxbound 2
          nseg 1
        end dimensions


        begin period 1
          open/close mymodel/evt/evt-0.dat
        end period
        begin period 2
          open/close mymodel/evt/evt-1.dat
        end period
        begin period 3
          open/close mymodel/evt/evt-2.dat
        end period"""
    )
    assert actual == expected


@pytest.mark.usefixtures(
    "rate_fc",
    "elevation_fc",
    "proportion_rate_fc",
    "proportion_depth_fc",
)
def test_get_options__no_segments(
    rate_fc, elevation_fc, proportion_rate_fc, proportion_depth_fc
):
    """Test with no segments specified, this means there implicitly is 1 segment
    in the Modflow 6 input."""

    # Arrange
    evt = imod.mf6.Evapotranspiration(
        surface=elevation_fc,
        rate=rate_fc,
        depth=elevation_fc,
        proportion_rate=proportion_rate_fc,
        proportion_depth=proportion_depth_fc,
    )

    # Act
    options = evt._get_options({})

    # Assert
    assert options["nseg"] == 1


@pytest.mark.usefixtures(
    "rate_fc",
    "elevation_fc",
    "proportion_rate_fc",
    "proportion_depth_fc",
)
def test_get_options__with_segments(
    rate_fc, elevation_fc, proportion_rate_fc, proportion_depth_fc
):
    """Test with 3 segments specified, this means that Modflow6 counts a total
    of 4 segments."""

    # Arrange
    segments = xr.DataArray(
        data=[1, 2, 3], coords={"segment": [1, 2, 3]}, dims=("segment",)
    )

    dim_order = "segment", "time", "layer", "y", "x"

    proportion_depth_fc = (proportion_depth_fc * segments).transpose(*dim_order)
    proportion_rate_fc = (proportion_rate_fc * segments).transpose(*dim_order)

    evt = imod.mf6.Evapotranspiration(
        surface=elevation_fc,
        rate=rate_fc,
        depth=elevation_fc,
        proportion_rate=proportion_rate_fc,
        proportion_depth=proportion_depth_fc,
    )

    # Act
    options = evt._get_options({})

    # Assert
    assert options["nseg"] == 4


@pytest.mark.usefixtures(
    "rate_fc",
    "elevation_fc",
    "proportion_rate_fc",
    "proportion_depth_fc",
)
def test_get_bin_ds__no_segments(
    rate_fc, elevation_fc, proportion_rate_fc, proportion_depth_fc
):
    # Arrange
    evt = imod.mf6.Evapotranspiration(
        surface=elevation_fc,
        rate=rate_fc,
        depth=elevation_fc,
        proportion_rate=proportion_rate_fc,
        proportion_depth=proportion_depth_fc,
    )

    # Act
    bin_ds = evt._get_bin_ds()

    # Assert
    expected_dims = {"time": 3, "layer": 3, "y": 15, "x": 15}
    expected_variables = [
        "surface",
        "rate",
        "depth",
        "proportion_depth",
        "proportion_rate",
    ]

    assert bin_ds.dims == expected_dims
    assert list(bin_ds.keys()) == expected_variables


@pytest.mark.usefixtures(
    "rate_fc",
    "elevation_fc",
    "proportion_rate_fc",
    "proportion_depth_fc",
)
def test_get_bin_ds__with_segments(
    rate_fc, elevation_fc, proportion_rate_fc, proportion_depth_fc
):
    # Arrange
    segments = xr.DataArray(
        data=[1, 2, 3], coords={"segment": [1, 2, 3]}, dims=("segment",)
    )

    dim_order = "segment", "time", "layer", "y", "x"

    proportion_depth_fc = (proportion_depth_fc * segments).transpose(*dim_order)
    proportion_rate_fc = (proportion_rate_fc * segments).transpose(*dim_order)

    evt = imod.mf6.Evapotranspiration(
        surface=elevation_fc,
        rate=rate_fc,
        depth=elevation_fc,
        proportion_rate=proportion_rate_fc,
        proportion_depth=proportion_depth_fc,
    )

    # Act
    bin_ds = evt._get_bin_ds()

    # Assert
    expected_dims = {"time": 3, "layer": 3, "y": 15, "x": 15}
    expected_variables = [
        "surface",
        "rate",
        "depth",
        "proportion_depth_segment_1",
        "proportion_depth_segment_2",
        "proportion_depth_segment_3",
        "proportion_rate_segment_1",
        "proportion_rate_segment_2",
        "proportion_rate_segment_3",
    ]

    assert bin_ds.dims == expected_dims
    assert list(bin_ds.keys()) == expected_variables


@pytest.mark.usefixtures(
    "rate_fc",
    "elevation_fc",
    "proportion_rate_fc",
    "proportion_depth_fc",
)
def test_wrong_dim_order(
    rate_fc, elevation_fc, proportion_rate_fc, proportion_depth_fc
):
    # Arrange
    segments = xr.DataArray(
        data=[1, 2, 3], coords={"segment": [1, 2, 3]}, dims=("segment",)
    )

    proportion_depth_fc = proportion_depth_fc * segments
    proportion_rate_fc = proportion_rate_fc * segments

    with pytest.raises(ValidationError):
        imod.mf6.Evapotranspiration(
            surface=elevation_fc,
            rate=rate_fc,
            depth=elevation_fc,
            proportion_rate=proportion_rate_fc,
            proportion_depth=proportion_depth_fc,
        )
