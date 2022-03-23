import numpy as np
import pandas as pd
import xarray as xr

from imod.fixed_format import VariableMetaData
from imod.msw.pkgbase import Package
from imod.msw.timeutil import to_metaswap_timeformat
from imod.util import spatial_reference


class OutputControl(Package):
    # TODO: Get list from Joachim which files we want to support, as there
    # are a lot options, many of which we never use.

    def __init__(self):
        super().__init__()

    def get_settings(self):
        """
        Return relevant settings for the PARA_SIM.INP file
        """
        return self._settings


# I did not use long variable names here (e.g. "precipitation"), as MetaSWAP
# uses these 2 to 4 character names to print its output to. This also has the
# benefit that the user is able to set additional variable names via kwargs
# (there are more than 130 possible variable names to choose from in MetaSWAP)
class VariableOutputControl(OutputControl):
    """
    Control which variables will be created as output. The variable names used
    in this class provide a condensed water balance. You can use additional
    keyword arguments to set more variables. For all possibilities see the
    SIMGRO Input and Output description.

    All budgets will be written in m unit to in `.idf` files and to mm unit in
    `.csv` files.

    Parameters
    ----------
    Pm: bool
        Write measured precipitation
    Psgw: bool
        Write sprinkling precipitation, from groundwater
    Pssw: bool
        Write sprinkling precipitation, from surface water
    qrun: bool
        Write runon
    qdr: bool
        Write net infiltration of surface water
    qspgw: bool
        Groundwater extraction for sprinkling from layer
    qmodf: bool
        Sum of all MODFLOW stresses on groundwater
    ETact: bool
        Write total actual evapotranspiration, which is the sum of the
        sprinkling evaporation (Esp), interception evaporation (Eic), ponding
        evaporation (Epd) bare soil evaporation (Ebs), and actual transpiration
        (Tact).
    **kwargs: bool
        Additional variables to let MetaSWAP write
    """

    _file_name = "sel_key_svat_per.inp"
    _settings = {}
    _metadata_dict = {
        "variable": VariableMetaData(10, None, None, str),
        "option": VariableMetaData(10, 0, 3, int),
    }

    def __init__(
        self,
        Pm=True,
        Psgw=True,
        Pssw=True,
        qrun=True,
        qdr=True,
        qspgw=True,
        qmodf=True,
        ETact=True,
        **kwargs,
    ):
        super().__init__()

        # Convert to integer, as MetaSWAP expects its values as integers.
        self.dataset["Pm"] = int(Pm)
        self.dataset["Psgw"] = int(Psgw)
        self.dataset["Pssw"] = int(Pssw)
        self.dataset["qrun"] = int(qrun)
        self.dataset["qdr"] = int(qdr)
        self.dataset["qspgw"] = int(qspgw)
        self.dataset["qmodf"] = int(qmodf)
        self.dataset["ETact"] = int(ETact)

        # Set additional settings
        for key, value in kwargs.items():
            self.dataset[key] = int(value)

    def _render(self, file, *args):
        variable, option = zip(
            *[(var, self.dataset[var].values) for var in self.dataset.data_vars]
        )

        dataframe = pd.DataFrame(data=dict(variable=variable, option=option))

        self._check_range(dataframe)

        return self.write_dataframe_fixed_width(file, dataframe)


class TimeOutputControl(OutputControl):
    """
    Specify the accumulation periods which will be used to write output.

    Parameters
    ----------
    time: xr.DataArray
        Timesteps at which to write output.
    """

    _file_name = "tiop_sim.inp"
    _settings = {}
    _metadata_dict = {
        "time_since_start_year": VariableMetaData(15, 0.0, 366.0, float),
        "year": VariableMetaData(6, 1, 9999, int),
        "option": VariableMetaData(6, 1, 7, int),
    }

    def __init__(self, time):
        super().__init__()

        self.dataset["times"] = time

    def _render(self, file, *args):

        year, time_since_start_year = to_metaswap_timeformat(self.dataset["times"])

        dataframe = pd.DataFrame(
            data=dict(time_since_start_year=time_since_start_year, year=year)
        )

        dataframe["time_since_start_year"] += 1
        dataframe["option"] = 7

        self._check_range(dataframe)

        return self.write_dataframe_fixed_width(file, dataframe)


class IdfOutputControl(OutputControl):
    """
    Output control to generate IDFs.

    Note that MetaSWAP can only write equidistant grids.
    """

    _file_name = "idf_svat.inp"
    _metadata_dict = {
        "svat": VariableMetaData(10, 1, 9999999, int),
        "rows": VariableMetaData(10, 1, 9999999, int),
        "columns": VariableMetaData(10, 1, 9999999, int),
        # TODO: Check if x and y limits are properly set.
        "y_grid": VariableMetaData(15, 0.0, 9999999.0, float),
        "x_grid": VariableMetaData(15, 0.0, 9999999.0, float),
    }

    _with_subunit = []
    _without_subunit = ["rows", "columns", "y_grid", "x_grid"]
    _to_fill = []

    # TODO: Quote from IO manual: The x- and y-coordinates should increase with increasing col, row.
    # But example works with decreasing y-coordinates?

    def __init__(self, area, nodata):
        super().__init__()

        self.dataset["area"] = area
        self.dataset["nodata"] = nodata

        nrow = self.dataset.coords["y"].size
        ncol = self.dataset.coords["x"].size

        y_index = xr.DataArray(
            np.arange(1, nrow + 1), coords={"y": self.dataset.coords["y"]}, dims=("y",)
        )
        x_index = xr.DataArray(
            np.arange(1, ncol + 1), coords={"x": self.dataset.coords["x"]}, dims=("x",)
        )
        rows, columns = xr.broadcast(y_index, x_index)

        self.dataset["rows"] = rows
        self.dataset["columns"] = columns

        y_grid, x_grid = xr.broadcast(self.dataset["y"], self.dataset["x"])

        self.dataset["x_grid"] = x_grid
        self.dataset["y_grid"] = y_grid

    def get_settings(self):
        grid = self.dataset["area"]
        dx, xmin, _, dy, ymin, _ = spatial_reference(grid)
        ncol = grid["x"].size
        nrow = grid["y"].size

        # If non-equidistant, spatial_reference returned a 1d array instead of
        # float
        if (type(dx) is np.ndarray) or (type(dy) is np.ndarray):
            raise ValueError("MetaSWAP can only write equidistant IDF grids")

        nodata = self.dataset["nodata"].values

        # TODO: Check if netcdf_per is also required, as manual seems a bit vague
        # about this. "To activate the idf-option the netcdf_per parameter in
        # PARA_SIM.INP must be set to 1." Could be a typo.
        return dict(
            simgro_opt=-1,
            idf_per=1,
            idf_dx=dx,
            idf_dy=np.abs(dy),
            idf_ncol=ncol,
            idf_nrow=nrow,
            idf_xmin=xmin,
            idf_ymin=ymin,
            idf_nodata=nodata,
        )
