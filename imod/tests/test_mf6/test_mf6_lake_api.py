import textwrap

import numpy as np
import xarray as xr

from imod.mf6.lake_package import lake_api


def create_gridcovering_array(idomain, lake_cells, fillvalue, dtype):
    """
    creates an array similar in dimensions/coords to idomain, but with value NaN (orr the missing value for integers)
    everywhere, except in the cells contained in list "lake_cells". In those cells, the output array has value fillvalue.
    """
    result = xr.full_like(
        idomain, fill_value=lake_api.missing_values[np.dtype(dtype).name], dtype=dtype
    )
    for cell in lake_cells:
        result.values[cell[0], cell[1], cell[2]] = fillvalue
    return result


def create_lakelake(idomain, starting_stage, boundname, lake_cells):
    """ """

    connectionType = create_gridcovering_array(
        idomain, lake_cells, lake_api.connection_types["HORIZONTAL"], np.int32
    )
    bed_leak = create_gridcovering_array(idomain, lake_cells, 0.2, np.float32)
    top_elevation = create_gridcovering_array(idomain, lake_cells, 0.3, np.float32)
    bot_elevation = create_gridcovering_array(idomain, lake_cells, 0.4, np.float32)
    connection_length = create_gridcovering_array(idomain, lake_cells, 0.5, np.float32)
    connection_width = create_gridcovering_array(idomain, lake_cells, 0.6, np.float32)
    result = lake_api.LakeLake(
        starting_stage,
        boundname,
        connectionType,
        bed_leak,
        top_elevation,
        bot_elevation,
        connection_length,
        connection_width,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    return result


def test_lake_api(basic_dis):
    idomain, _, _ = basic_dis

    outlet1 = lake_api.OutletManning(1, "Naardermeer", "Ijsselmeer", 23, 24, 25, 26)
    outlet2 = lake_api.OutletManning(2, "Ijsselmeer", "Naardermeer", 27, 28, 29, 30)

    lake1 = create_lakelake(
        idomain, 11, "Naardermeer", [(1, 2, 2), (1, 2, 3), (1, 3, 3)]
    )
    lake2 = create_lakelake(
        idomain, 15, "Ijsselmeer", [(1, 5, 5), (1, 5, 6), (1, 6, 6)]
    )
    lake_package = lake_api.from_lakes_and_outlets([lake1, lake2], [outlet1, outlet2])
    actual = lake_package.render(None, None, None, False)
    expected = textwrap.dedent(
        """\
        begin options
        end options

        begin dimensions
          nlakes 2
          noutlets 2
          ntables 0
        end dimensions

        begin packagedata
          1  11  Naardermeer
          2  15  Ijsselmeer
        end packagedata

        begin connectiondata
          0 1 5 0.0  0.2 0.4    0.6 0.5
          0 1 6 0.0  0.2 0.4    0.6 0.5
          0 1 6 0.0  0.2 0.4    0.6 0.5
          1 1 5 0.0  0.2 0.4    0.6 0.5
          1 1 6 0.0  0.2 0.4    0.6 0.5
          1 1 6 0.0  0.2 0.4    0.6 0.5
        end connectiondata

        begin outlets
          1 2 MANNING 23 25 24 26
          2 1 MANNING 27 29 28 30
        end outlets
        """
    )

    assert actual == expected
