import itertools

import numpy as np
import xarray as xr

"""
This source file contains an interface to the lake package
"""

connection_types = {"HORIZONTAL": 0, "VERTICAL": 1, "EMBEDDEDH": 2, "EMBEDDEDV": 3}
missing_values = {
    "float32": np.nan,
    "int32": -6789012,
}


class LakeLake:
    """ """

    def __init__(
        self,
        starting_stage: float,
        boundname: str,
        connectionType,
        bed_leak,
        top_elevation,
        bot_elevation,
        connection_length,
        connection_width,
        laketable,
        status,
        stage,
        rainfall,
        evaporation,
        runoff,
        inflow,
        withdrawal,
        auxiliary,
    ):
        self.starting_stage = starting_stage
        self.boundname = boundname
        self.connectionType = connectionType
        self.bed_leak = bed_leak
        self.top_elevation = top_elevation
        self.bot_elevation = bot_elevation
        self.connection_length = connection_length
        self.connection_width = connection_width

        # table for this lake
        self.laketable = (laketable,)

        # timeseries
        self.status = (status,)
        self.stage = (stage,)
        self.rainfall = (rainfall,)
        self.evaporation = (evaporation,)
        self.runoff = (runoff,)
        self.inflow = (inflow,)
        self.withdrawal = (withdrawal,)
        self.auxiliar = auxiliary

    @classmethod
    def get_subdomain_indices(cls, whole_domain_coords, subdomain_coords):
        result = []
        list_whole_domain_coords = list(whole_domain_coords)
        for i in range(0, len(subdomain_coords)):
            result.append(list_whole_domain_coords.index(subdomain_coords[i]))
        return result

    @classmethod
    def get_1d_array(cls, grid_array):
        dummy = grid_array.sel()
        dummy = dummy.where(dummy != missing_values[grid_array.dtype.name], drop=True)
        x_indexes = cls.get_subdomain_indices(grid_array.x, dummy.x)
        y_indexes = cls.get_subdomain_indices(grid_array.y, dummy.y)
        layer_indexes = cls.get_subdomain_indices(grid_array.layer, dummy.layer)
        dummy = dummy.assign_coords(
            {"x": x_indexes, "y": y_indexes, "layer": layer_indexes}
        )

        dummy = dummy.stack(z=("x", "y", "layer"))
        dummy = dummy.dropna("z")

        x_values = list(dummy.z.x.values)
        y_values = list(dummy.z.y.values)
        layer_values = list(dummy.z.layer.values)
        array_values = list(dummy.values)
        return x_values, y_values, layer_values, array_values

        # todo: check input


class OutletBase:
    def __init__(self, outletNumber: int, lakeIn: str, lakeOut: str):
        self.outletNumber = outletNumber
        self.lakeIn = lakeIn
        self.lakeOut = lakeOut


class OutletManning(OutletBase):
    def __init__(
        self,
        outletNumber: int,
        lakeIn: str,
        lakeOut: str,
        invert,
        width,
        roughness,
        slope,
    ):
        super().__init__(outletNumber, lakeIn, lakeOut)
        self.invertt = invert
        self.width = width
        self.roughness = roughness
        self.slope = slope


class OutletWeir(OutletBase):
    def __init__(
        self,
        outletNumber: int,
        lakeIn: str,
        lakeOut: str,
        invert,
        width,
    ):
        super().__init__(outletNumber, lakeIn, lakeOut)
        self.invertt = invert
        self.width = width


class OutletSpecified(OutletBase):
    def __init__(self, outletNumber: int, lakeIn: str, lakeOut: str, rate):
        super().__init__(outletNumber, lakeIn, lakeOut)
        self.rate = rate


class LakeTable:
    def __init__(self, stage, volume, surface, exchange_surface=None):
        self.stage = stage
        self.volume = volume
        self.surface = surface
        self.exchange_surface = exchange_surface


def from_lakes_and_outlets(list_of_lakes, list_of_outlets=[]):

    nrlakes = len(list_of_lakes)

    dimensions = ["lake_nr"]
    coordinates = {"lake_nr": np.arange(0, nrlakes)}
    starting_stage = xr.DataArray(
        np.ones(nrlakes, dtype=np.float32), coords=coordinates, dims=dimensions
    )
    dimensions = ["lake_nr"]
    coordinates = {"lake_nr": np.arange(0, nrlakes)}
    boundname = xr.DataArray(
        np.ones(nrlakes, dtype=str), coords=coordinates, dims=dimensions
    )
    lakenumber = []
    c_cell_id_row_or_index = []
    c_cell_id_col = []
    c_cell_id_layer = []
    c_type = []
    c_bed_leak = []
    c_bottom_elevation = []
    c_top_elevation = []
    c_width = []
    c_length = []
    c_cell_id_row_or_index = []
    for i in range(0, nrlakes):
        starting_stage.values[i] = list_of_lakes[i].starting_stage
        boundname.values[i] = list_of_lakes[i].boundname
        layer, y, x, ctype = list_of_lakes[i].get_1d_array(
            list_of_lakes[i].connectionType
        )
        c_type += ctype
        c_cell_id_row_or_index += x
        c_cell_id_col += y
        c_cell_id_layer += layer
        _, _, _, bed_leak = list_of_lakes[i].get_1d_array(list_of_lakes[i].bed_leak)
        c_bed_leak += bed_leak
        _, _, _, top_elevation = list_of_lakes[i].get_1d_array(
            list_of_lakes[i].top_elevation
        )
        c_top_elevation += top_elevation
        _, _, _, bot_elevation = list_of_lakes[i].get_1d_array(
            list_of_lakes[i].bot_elevation
        )
        c_bottom_elevation += bot_elevation
        _, _, _, connection_length = list_of_lakes[i].get_1d_array(
            list_of_lakes[i].connection_length
        )
        c_length += connection_length
        _, _, _, connection_width = list_of_lakes[i].get_1d_array(
            list_of_lakes[i].connection_width
        )
        c_width += connection_width

        lakenumber.append(itertools.repeat(i, len(ctype)))
