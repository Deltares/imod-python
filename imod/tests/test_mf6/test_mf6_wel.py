import pathlib
import tempfile
import textwrap

import numpy as np
import pytest
import xarray as xr

import imod
from imod.schemata import ValidationError


def test_render(well_test_data_stationary):
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
    globaltimes = [np.datetime64("2000-01-01")]
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


def test_render_transient(well_test_data_transient):
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
    globaltimes = [
        np.datetime64("2000-01-01"),
        np.datetime64("2000-02-01"),
        np.datetime64("2000-03-01"),
    ]
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


def test_render_concentration_dis_structured_constant_time(well_test_data_stationary):
    layer, row, column, rate, injection_concentration = well_test_data_stationary

    concentration = xr.DataArray(
        data=injection_concentration,
        dims=["cell", "species"],
        coords=dict(
            cell=(range(0, 15)),
            species=(["salinity", "temperature"]),
        ),
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
    globaltimes = [np.datetime64("2000-01-01")]
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
        wel.write(output_dir, "wel", globaltimes, False)
        with open(output_dir + "/wel/wel.dat", "r") as f:
            data = f.read()
            assert (
                data.count(" 123 456") == 15
            )  # check salinity and temperature was written to period data


def test_render_concentration_dis_vertices_constant_time(well_test_data_stationary):
    layer, row, column, rate, injection_concentration = well_test_data_stationary

    concentration = xr.DataArray(
        data=injection_concentration,
        dims=["cell", "species"],
        coords=dict(
            cell=(range(0, 15)),
            species=(["salinity", "temperature"]),
        ),
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
    globaltimes = [np.datetime64("2000-01-01")]

    with tempfile.TemporaryDirectory() as output_dir:
        wel.write(output_dir, "wel", globaltimes, False)
        with open(output_dir + "/wel/wel.dat", "r") as f:
            data = f.read()
            assert (
                data.count(" 123 456") == 15
            )  # check salinity and temperature was written to period data


def test_render_concentration_dis_vertices_transient(well_test_data_transient):
    layer, row, column, time, rate, injection_concentration = well_test_data_transient

    concentration = xr.DataArray(
        data=injection_concentration,
        dims=["time", "cell", "species"],
        coords=dict(
            time=time,
            cell=(range(0, 15)),
            species=(["salinity", "temperature"]),
        ),
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
        wel.write(output_dir, "wel", time, False)
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
