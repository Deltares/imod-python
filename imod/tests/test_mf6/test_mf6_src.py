import pathlib
import tempfile
import textwrap

import numpy as np
import pytest
import xarray as xr

import imod


def test_render(well_test_data_stationary):
    layer, row, column, rate, _ = well_test_data_stationary
    src = imod.mf6.MassSourceLoadingDisStructured(
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
    actual = src.render(directory, "src", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin dimensions
          maxbound 15
        end dimensions

        begin period 1
          open/close mymodel/src/src.bin (binary)
        end period
        """
    )
    assert actual == expected
    cell2d = (row - 1) * 15 + column
    src = imod.mf6.MassSourceLoadingDisVertices(
        layer=layer,
        cell2d=cell2d,
        rate=rate,
        print_input=False,
        print_flows=False,
        save_flows=False,
    )
    actual = src.render(directory, "src", globaltimes, True)
    assert actual == expected


def test_render_mass_source_transient(well_test_data_transient):
    layer, row, column, times, rate, _ = well_test_data_transient

    with pytest.raises(ValueError, match="time varying variable: must be 2d"):
        imod.mf6.MassSourceLoadingDisStructured(
            layer=layer,
            row=row,
            column=column,
            rate=rate.isel(index=0),
            print_input=False,
            print_flows=False,
            save_flows=False,
        )

    src = imod.mf6.MassSourceLoadingDisStructured(
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
    actual = src.render(directory, "src", globaltimes, True)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin dimensions
          maxbound 15
        end dimensions

        begin period 1
          open/close mymodel/src/src-0.bin (binary)
        end period
        begin period 2
          open/close mymodel/src/src-1.bin (binary)
        end period
        """
    )
    assert actual == expected

    # Test automatic transpose, where time is the second time
    src = imod.mf6.MassSourceLoadingDisStructured(
        layer=layer,
        row=row,
        column=column,
        rate=rate.transpose(),
        print_input=False,
        print_flows=False,
        save_flows=False,
    )
    actual = src.render(directory, "src", globaltimes, True)
    assert actual == expected


