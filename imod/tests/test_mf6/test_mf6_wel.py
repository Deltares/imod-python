import pathlib
import tempfile
import textwrap
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import xarray as xr
import xugrid as xu
from pytest_cases import parametrize_with_cases

import imod
from imod.mf6.utilities.grid import broadcast_to_full_domain
from imod.mf6.write_context import WriteContext
from imod.schemata import ValidationError
from imod.tests.fixtures.flow_basic_fixture import BasicDisSettings


def test_to_mf6_pkg__high_lvl_stationary(basic_dis, well_high_lvl_test_data_stationary):
    # Arrange
    idomain, top, bottom = basic_dis
    wel = imod.mf6.Well(*well_high_lvl_test_data_stationary)
    active = idomain == 1
    k = xr.ones_like(idomain)

    nmax_cellid_expected = np.array(["layer", "row", "column"])
    cellid_expected = np.array(
        [
            [1, 1, 9],
            [1, 2, 9],
            [1, 1, 8],
            [1, 2, 8],
            [2, 3, 7],
            [2, 4, 7],
            [2, 3, 6],
            [2, 4, 6],
        ],
        dtype=np.int64,
    )
    rate_expected = np.array(np.ones((8,), dtype=np.float32))

    # Act
    mf6_wel = wel.to_mf6_pkg(active, top, bottom, k)
    mf6_ds = mf6_wel.dataset

    # Assert
    assert dict(mf6_ds.dims) == {
        "ncellid": 8,
        "nmax_cellid": 3,
        "species": 2,
    }
    np.testing.assert_equal(mf6_ds.coords["nmax_cellid"].values, nmax_cellid_expected)
    np.testing.assert_equal(mf6_ds["cellid"].values, cellid_expected)
    np.testing.assert_equal(mf6_ds["rate"].values, rate_expected)


def test_to_mf6_pkg__validate(well_high_lvl_test_data_stationary):
    # Arrange
    wel = imod.mf6.Well(*well_high_lvl_test_data_stationary)

    # Act
    errors = wel._validate(wel._write_schemata)
    assert len(errors) == 0

    # Set rates with index exceeding 3 to NaN.
    wel.dataset["rate"] = wel.dataset["rate"].where(wel.dataset.coords["index"] < 3)
    errors = wel._validate(wel._write_schemata)
    assert len(errors) == 1


def test_to_mf6_pkg__high_lvl_multilevel(basic_dis, well_high_lvl_test_data_stationary):
    """
    Test with stationary wells where the first 4 well screens extend over 2 layers.
    Rates are distributed based on the fraction of the screen length in each layer.
    In this case: The first layer should get 0.25, the second 0.75.
    """
    # Arrange
    idomain, top, bottom = basic_dis
    x, y, screen_top, _, rate_wel, concentration = well_high_lvl_test_data_stationary
    screen_bottom = [-20.0] * 8
    wel = imod.mf6.Well(x, y, screen_top, screen_bottom, rate_wel, concentration)
    active = idomain == 1
    k = xr.ones_like(idomain)

    nmax_cellid_expected = np.array(["layer", "row", "column"])
    cellid_expected = np.array(
        [
            [1, 1, 9],
            [1, 2, 9],
            [1, 1, 8],
            [1, 2, 8],
            [2, 1, 9],
            [2, 2, 9],
            [2, 1, 8],
            [2, 2, 8],
            [2, 3, 7],
            [2, 4, 7],
            [2, 3, 6],
            [2, 4, 6],
        ],
        dtype=np.int64,
    )
    rate_expected = np.array([0.25] * 4 + [0.75] * 4 + [1.0] * 4)

    # Act
    mf6_wel = wel.to_mf6_pkg(active, top, bottom, k)
    mf6_ds = mf6_wel.dataset

    # Assert
    assert dict(mf6_ds.dims) == {
        "ncellid": 12,
        "nmax_cellid": 3,
        "species": 2,
    }
    np.testing.assert_equal(mf6_ds.coords["nmax_cellid"].values, nmax_cellid_expected)
    np.testing.assert_equal(mf6_ds["cellid"].values, cellid_expected)
    np.testing.assert_equal(mf6_ds["rate"].values, rate_expected)


def test_to_mf6_pkg__high_lvl_transient(basic_dis, well_high_lvl_test_data_transient):
    # Arrange
    idomain, top, bottom = basic_dis
    wel = imod.mf6.Well(*well_high_lvl_test_data_transient)
    active = idomain == 1
    k = xr.ones_like(idomain)

    nmax_cellid_expected = np.array(["layer", "row", "column"])
    cellid_expected = np.array(
        [
            [1, 1, 9],
            [1, 2, 9],
            [1, 1, 8],
            [1, 2, 8],
            [2, 3, 7],
            [2, 4, 7],
            [2, 3, 6],
            [2, 4, 6],
        ],
        dtype=np.int64,
    )
    rate_expected = np.outer(np.ones((8,), dtype=np.float32), np.arange(5) + 1)

    # Act
    mf6_wel = wel.to_mf6_pkg(active, top, bottom, k)
    mf6_ds = mf6_wel.dataset

    # Assert
    assert dict(mf6_wel.dataset.dims) == {
        "ncellid": 8,
        "time": 5,
        "nmax_cellid": 3,
        "species": 2,
    }
    np.testing.assert_equal(mf6_ds.coords["nmax_cellid"].values, nmax_cellid_expected)
    np.testing.assert_equal(mf6_ds["cellid"].values, cellid_expected)
    np.testing.assert_equal(mf6_ds["rate"].values, rate_expected)


@pytest.mark.parametrize("save_flows", [True, False])
@pytest.mark.parametrize("print_input", [True, False])
@pytest.mark.parametrize("print_flows", [True, False])
def test_to_mf6_pkg__save_flows(
    basic_dis, well_high_lvl_test_data_transient, save_flows, print_input, print_flows
):
    # Arrange
    idomain, top, bottom = basic_dis

    wel = imod.mf6.Well(
        *well_high_lvl_test_data_transient,
        save_flows=save_flows,
        print_input=print_input,
        print_flows=print_flows,
    )
    active = idomain == 1
    k = xr.ones_like(idomain)

    # Act
    mf6_wel = wel.to_mf6_pkg(active, top, bottom, k)
    mf6_ds = mf6_wel.dataset

    # Assert
    mf6_ds["save_flows"].values[()] == save_flows
    mf6_ds["print_input"].values[()] == print_input
    mf6_ds["print_flows"].values[()] == print_flows


def test_is_empty(well_high_lvl_test_data_transient):
    # Arrange
    wel = imod.mf6.Well(*well_high_lvl_test_data_transient)
    empty_wel_args = ([] for i in range(len(well_high_lvl_test_data_transient)))
    wel_empty = imod.mf6.Well(*empty_wel_args, validate=False)

    # Act/Assert
    assert not wel.is_empty()
    assert wel_empty.is_empty()


class ClipBoxCases:
    @staticmethod
    def case_clip_xy(parameterizable_basic_dis):
        clip_arguments = {
            "x_min": 52.0,
            "x_max": 76.0,
            "y_max": 67.0,
        }

        expected_dims = {"index": 3, "species": 2}
        return clip_arguments, expected_dims, does_not_raise()

    @staticmethod
    def case_clip_layer_max(parameterizable_basic_dis):
        _, top, bottom = parameterizable_basic_dis
        clip_arguments = {"layer_max": 2, "bottom": bottom, "top": top}

        expected_dims = {"index": 4, "species": 2}
        return clip_arguments, expected_dims, does_not_raise()

    @staticmethod
    def case_clip_layer_min(parameterizable_basic_dis):
        _, top, bottom = parameterizable_basic_dis
        clip_arguments = {"layer_min": 5, "bottom": bottom, "top": top}

        expected_dims = {"index": 4, "species": 2}
        return clip_arguments, expected_dims, does_not_raise()

    @staticmethod
    def case_clip_layer_min_layer_max(parameterizable_basic_dis):
        _, top, bottom = parameterizable_basic_dis
        clip_arguments = {"layer_min": 1, "layer_max": 1, "bottom": bottom, "top": top}

        expected_dims = {"index": 4, "species": 2}
        return clip_arguments, expected_dims, does_not_raise()

    @staticmethod
    def case_clip_top_is_scalar(parameterizable_basic_dis):
        _, _, bottom = parameterizable_basic_dis
        top = 0.0
        clip_arguments = {"layer_max": 2, "bottom": bottom, "top": top}

        expected_dims = {"index": 4, "species": 2}
        return clip_arguments, expected_dims, does_not_raise()

    @staticmethod
    def case_clip_top_is_non_layered_structuredgrid(parameterizable_basic_dis):
        idomain, top, bottom = parameterizable_basic_dis
        top, bottom = broadcast_to_full_domain(idomain, top, bottom)
        top = top.isel(layer=0).drop_vars("layer")

        clip_arguments = {"layer_max": 2, "bottom": bottom, "top": top}

        expected_dims = {"index": 4, "species": 2}
        return clip_arguments, expected_dims, does_not_raise()

    @staticmethod
    def case_clip_top_is_layered_structuredgrid(parameterizable_basic_dis):
        idomain, top, bottom = parameterizable_basic_dis

        top, bottom = broadcast_to_full_domain(idomain, top, bottom)
        clip_arguments = {"layer_max": 2, "bottom": bottom, "top": top}

        expected_dims = {"index": 4, "species": 2}
        return clip_arguments, expected_dims, does_not_raise()

    @staticmethod
    def case_clip_top_is_non_layered_unstructuredgrid(parameterizable_basic_dis):
        idomain, top, bottom = parameterizable_basic_dis
        top, bottom = broadcast_to_full_domain(idomain, top, bottom)
        top = xu.UgridDataArray.from_structured(top)
        bottom = xu.UgridDataArray.from_structured(bottom)

        top = top.isel(layer=0).drop_vars("layer")

        clip_arguments = {"layer_max": 2, "bottom": bottom, "top": top}

        expected_dims = {"index": 4, "species": 2}
        return clip_arguments, expected_dims, does_not_raise()

    @staticmethod
    def case_clip_top_is_layered_unstructuredgrid(parameterizable_basic_dis):
        idomain, top, bottom = parameterizable_basic_dis
        top, bottom = broadcast_to_full_domain(idomain, top, bottom)
        top = xu.UgridDataArray.from_structured(top)
        bottom = xu.UgridDataArray.from_structured(bottom)

        clip_arguments = {"layer_max": 2, "bottom": bottom, "top": top}

        expected_dims = {"index": 4, "species": 2}
        return clip_arguments, expected_dims, does_not_raise()

    @staticmethod
    def case_clip_missing_top(parameterizable_basic_dis):
        _, _, bottom = parameterizable_basic_dis
        clip_arguments = {"layer_max": 2, "bottom": bottom}

        expected_dims = {}
        return clip_arguments, expected_dims, pytest.raises(ValueError)

    @staticmethod
    def case_clip_missing_bottom(parameterizable_basic_dis):
        _, top, _ = parameterizable_basic_dis
        clip_arguments = {"layer_max": 2, "top": top}

        expected_dims = {}
        return clip_arguments, expected_dims, pytest.raises(ValueError)


@pytest.mark.parametrize(
    "parameterizable_basic_dis",
    [
        BasicDisSettings(
            nlay=10,
            zstop=-10.0,
            xstart=50.0,
            xstop=100.0,
            ystart=50.0,
            ystop=100.0,
            nrow=10,
            ncol=10,
        )
    ],
    indirect=True,
)
@parametrize_with_cases(
    ("clip_box_args", "expected_dims", "expectation"), cases=ClipBoxCases
)
def test_clip_box__high_lvl_stationary(
    well_high_lvl_test_data_stationary, clip_box_args, expected_dims, expectation
):
    # Arrange
    wel = imod.mf6.Well(*well_high_lvl_test_data_stationary)

    with expectation:
        # Act
        ds = wel.clip_box(**clip_box_args).dataset

        # Assert
        assert dict(ds.dims) == expected_dims


@pytest.mark.parametrize(
    "parameterizable_basic_dis",
    [BasicDisSettings(nlay=10, zstop=-10.0)],
    indirect=True,
)
def test_clip_box__high_lvl_transient(
    well_high_lvl_test_data_transient, parameterizable_basic_dis
):
    # Arrange
    wel = imod.mf6.Well(*well_high_lvl_test_data_transient)
    _, top, bottom = parameterizable_basic_dis

    # Act & Assert
    # Test clipping x & y without specified time
    ds = wel.clip_box(x_min=52.0, x_max=76.0, y_max=67.0).dataset
    assert dict(ds.dims) == {"index": 3, "time": 5, "species": 2}

    # Test clipping with z
    ds = wel.clip_box(layer_max=2, top=top, bottom=bottom).dataset
    assert dict(ds.dims) == {"index": 4, "time": 5, "species": 2}
    ds = wel.clip_box(layer_min=5, top=top, bottom=bottom).dataset
    assert dict(ds.dims) == {"index": 4, "time": 5, "species": 2}

    # Test clipping with specified time
    timestr = "2000-01-03"
    ds = wel.clip_box(time_min=timestr).dataset
    assert dict(ds.dims) == {"index": 8, "time": 3, "species": 2}

    # Test clipping with specified time and spatial dimensions
    timestr = "2000-01-03"
    ds = wel.clip_box(x_min=52.0, x_max=76.0, y_max=67.0, time_min=timestr).dataset
    assert dict(ds.dims) == {"index": 3, "time": 3, "species": 2}

    # Test clipping with specified time inbetween timesteps
    timestr = "2000-01-03 18:00:00"
    ds = wel.clip_box(x_min=52.0, x_max=76.0, y_max=67.0, time_min=timestr).dataset
    assert dict(ds.dims) == {"index": 3, "time": 3, "species": 2}
    ds_first_time = ds.isel(time=0)
    assert ds_first_time.coords["time"] == np.datetime64(timestr)
    np.testing.assert_allclose(ds_first_time["rate"].values, np.array([3.0, 3.0, 3.0]))
    np.testing.assert_allclose(
        ds_first_time["concentration"].values,
        np.array([[30.0, 30.0, 30.0], [69.0, 69.0, 69.0]]),
    )


def test_derive_cellid_from_points(basic_dis, well_high_lvl_test_data_stationary):
    # Arrange
    idomain, _, _ = basic_dis
    x, y, _, _, _, _ = well_high_lvl_test_data_stationary
    layer = [1, 1, 1, 1, 2, 2, 2, 2]

    nmax_cellid_expected = np.array(["layer", "row", "column"])
    cellid_expected = np.array(
        [
            [1, 1, 9],
            [1, 2, 9],
            [1, 1, 8],
            [1, 2, 8],
            [2, 3, 7],
            [2, 4, 7],
            [2, 3, 6],
            [2, 4, 6],
        ],
        dtype=np.int64,
    )

    # Act
    cellid = imod.mf6.wel.Well._Well__derive_cellid_from_points(idomain, x, y, layer)

    # Assert
    np.testing.assert_array_equal(cellid, cellid_expected)
    np.testing.assert_equal(cellid.coords["nmax_cellid"].values, nmax_cellid_expected)


def test_render__stationary(well_test_data_stationary):
    layer, row, column, rate, _ = well_test_data_stationary
    wel = imod.mf6.WellDisStructured(
        layer=layer,
        row=row,
        column=column,
        rate=rate,
        print_input=False,
        print_flows=False,
        save_flows=False,
    )
    directory = pathlib.Path("mymodel")
    globaltimes = np.array(["2000-01-01"], dtype="datetime64[ns]")
    actual = wel.render(directory, "well", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin dimensions
          maxbound 15
        end dimensions

        begin period 1
          open/close mymodel/well/wel.bin (binary)
        end period
        """
    )
    assert actual == expected
    cell2d = (row - 1) * 15 + column
    wel = imod.mf6.WellDisVertices(
        layer=layer,
        cell2d=cell2d,
        rate=rate,
        print_input=False,
        print_flows=False,
        save_flows=False,
    )
    actual = wel.render(directory, "well", globaltimes, True)
    assert actual == expected


def test_render__transient(well_test_data_transient):
    layer, row, column, times, rate, _ = well_test_data_transient

    with pytest.raises(ValueError, match="time varying variable: must be 2d"):
        imod.mf6.WellDisStructured(
            layer=layer,
            row=row,
            column=column,
            rate=rate.isel(index=0),
            print_input=False,
            print_flows=False,
            save_flows=False,
        )

    wel = imod.mf6.WellDisStructured(
        layer=layer,
        row=row,
        column=column,
        rate=rate,
        print_input=False,
        print_flows=False,
        save_flows=False,
    )
    directory = pathlib.Path("mymodel")
    globaltimes = np.array(
        [
            "2000-01-01",
            "2000-02-01",
            "2000-03-01",
        ],
        dtype="datetime64[ns]",
    )
    actual = wel.render(directory, "well", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin dimensions
          maxbound 15
        end dimensions

        begin period 1
          open/close mymodel/well/wel-0.bin (binary)
        end period
        begin period 2
          open/close mymodel/well/wel-1.bin (binary)
        end period
        """
    )
    assert actual == expected

    # Test automatic transpose, where time is the second time
    wel = imod.mf6.WellDisStructured(
        layer=layer,
        row=row,
        column=column,
        rate=rate.transpose(),
        print_input=False,
        print_flows=False,
        save_flows=False,
    )
    actual = wel.render(directory, "well", globaltimes, True)
    assert actual == expected


def test_wrong_dtype():
    layer = np.array([3, 2, 2])
    row = np.array([5, 4, 6])
    column = np.array([11, 6, 12])
    rate = np.full(3, 5)
    with pytest.raises(ValidationError):
        imod.mf6.WellDisStructured(
            layer=layer,
            row=row,
            column=column,
            rate=rate,
            print_input=False,
            print_flows=False,
            save_flows=False,
        )


def test_validate_false():
    layer = np.array([3, 2, 2])
    row = np.array([5, 4, 6])
    column = np.array([11, 6, 12])
    rate = np.full(3, 5)

    imod.mf6.WellDisStructured(
        layer=layer,
        row=row,
        column=column,
        rate=rate,
        print_input=False,
        print_flows=False,
        save_flows=False,
        validate=False,
    )


def test_render__concentration_dis_structured_constant_time(well_test_data_stationary):
    layer, row, column, rate, injection_concentration = well_test_data_stationary

    concentration = xr.DataArray(
        data=injection_concentration,
        dims=["cell", "species"],
        coords={
            "cell": (range(0, 15)),
            "species": (["salinity", "temperature"]),
        },
    )

    wel = imod.mf6.WellDisStructured(
        layer=layer,
        row=row,
        column=column,
        rate=rate,
        concentration=concentration,
        concentration_boundary_type="AUX",
        print_input=False,
        print_flows=False,
        save_flows=False,
    )
    directory = pathlib.Path("mymodel")
    globaltimes = np.array(["2000-01-01"], dtype="datetime64[ns]")
    actual = wel.render(directory, "well", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
          auxiliary salinity temperature
        end options

        begin dimensions
          maxbound 15
        end dimensions

        begin period 1
          open/close mymodel/well/wel.bin (binary)
        end period
        """
    )
    assert actual == expected

    with tempfile.TemporaryDirectory() as output_dir:
        write_context = WriteContext(simulation_directory=output_dir)
        wel.write("wel", globaltimes, write_context)
        with open(output_dir + "/wel/wel.dat", "r") as f:
            data = f.read()
            assert (
                data.count(" 123 456") == 15
            )  # check salinity and temperature was written to period data


def test_render__concentration_dis_vertices_constant_time(well_test_data_stationary):
    layer, row, column, rate, injection_concentration = well_test_data_stationary

    concentration = xr.DataArray(
        data=injection_concentration,
        dims=["cell", "species"],
        coords={
            "cell": (range(0, 15)),
            "species": (["salinity", "temperature"]),
        },
    )

    cell2d = (row - 1) * 15 + column
    wel = imod.mf6.WellDisVertices(
        layer=layer,
        cell2d=cell2d,
        rate=rate,
        concentration=concentration,
        concentration_boundary_type="AUX",
        print_input=False,
        print_flows=False,
        save_flows=False,
    )
    globaltimes = np.array(["2000-01-01"], dtype="datetime64[ns]")

    with tempfile.TemporaryDirectory() as output_dir:
        write_context = WriteContext(simulation_directory=output_dir)
        wel.write("wel", globaltimes, write_context)
        with open(output_dir + "/wel/wel.dat", "r") as f:
            data = f.read()
            assert (
                data.count(" 123 456") == 15
            )  # check salinity and temperature was written to period data


def test_render__concentration_dis_vertices_transient(well_test_data_transient):
    layer, row, column, time, rate, injection_concentration = well_test_data_transient

    concentration = xr.DataArray(
        data=injection_concentration,
        dims=["time", "cell", "species"],
        coords={
            "time": time,
            "cell": (range(0, 15)),
            "species": (["salinity", "temperature"]),
        },
    )

    cell2d = (row - 1) * 15 + column
    wel = imod.mf6.WellDisVertices(
        layer=layer,
        cell2d=cell2d,
        rate=rate,
        concentration=concentration,
        concentration_boundary_type="AUX",
        print_input=False,
        print_flows=False,
        save_flows=False,
    )

    with tempfile.TemporaryDirectory() as output_dir:
        write_context = WriteContext(simulation_directory=output_dir)

        wel.write("wel", time, write_context)
        with open(output_dir + "/wel/wel-0.dat", "r") as f:
            data = f.read()
            assert (
                data.count(" 123 456") == 15
            )  # check salinity and temperature was written to period data
        with open(output_dir + "/wel/wel-1.dat", "r") as f:
            data = f.read()
            assert (
                data.count(" 246 135") == 15
            )  # check salinity and temperature was written to period data
