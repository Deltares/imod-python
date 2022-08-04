import xarray as xr
import numpy as np

connection_types = { "HORIZONTAL": 0, "VERTICAL": 1, "EMBEDDEDH": 2, "EMBEDDEDV": 3}
missing_values = { 'float32': np.NaN, 'int32': -6789012, }

class LakeLake:
    def __init__(self, starting_stage: float, boundname: str, connectionType, bed_leak,top_elevation, bot_elevation,
        connection_length, connection_width, laketable, status, stage, rainfall, evaporation, runoff, inflow, withdrawal,
        auxiliary):
        self.starting_stage = starting_stage
        self.boundname = boundname
        self.connectionType = connectionType
        self.bed_leak = bed_leak
        self.top_elevation = top_elevation
        self.bot_elevation = bot_elevation
        self.connection_length = connection_length
        self.connection_width = connection_width

        #table for this lake
        self.laketable = laketable,

        # timeseries
        self.status = status,
        self.stage = stage,
        self.rainfall = rainfall,
        self.evaporation= evaporation,
        self.runoff = runoff,
        self.inflow = inflow,
        self.withdrawal = withdrawal,
        self.auxiliar = auxiliary

    def get_1d_array(self, name):
        for varname, value in vars(self).items():
            if varname == name:
                dummy = value.sel()
                dummy=dummy.where(dummy!=missing_values[value.dtype.name], drop=True).squeeze()

        #todo: check input

class OutletBase:
    def __init__(self,outletNumber: int, lakeIn: str, lakeOut: str ):
        self.outletNumber = outletNumber
        self.lakeIn= lakeIn
        self.lakeOut = lakeOut


class OutletManning (OutletBase):
    def __init__(self, outletNumber: int, lakeIn: str, lakeOut: str, invert, width, roughness, slope):
        super().__init__(outletNumber,lakeIn,lakeOut )
        self.invertt = invert
        self.width = width
        self.roughness = roughness
        self.slope = slope

class OutletWeir(OutletBase):
    def __init__(self, outletNumber: int, lakeIn: str, lakeOut: str,invert, width,  ):
        super().__init__(outletNumber,lakeIn,lakeOut )
        self.invertt = invert
        self.width = width


class OutletSpecified(OutletBase):
    def __init__(self, outletNumber: int, lakeIn: str, lakeOut: str, rate):
        super().__init__(outletNumber,lakeIn,lakeOut )
        self.rate = rate

def from_lakes_and_outlets( list_of_lakes, list_of_outlets = []):

    nrlakes = len(list_of_lakes)
    nroutlets = len(list_of_outlets)
    nrtables = sum(1 for lake in list_of_lakes if lake.laketable is not None)


    dimensions = ["lake_nr"]
    coordinates = {"lake_nr": np.arange(0,nrlakes)}
    starting_stage = xr.DataArray(np.ones(nrlakes, dtype=np.float32), coords=coordinates, dims=dimensions)
    dimensions = ["lake_nr"]
    coordinates = {"lake_nr": np.arange(0,nrlakes)}
    boundname = xr.DataArray(np.ones(nrlakes, dtype=str), coords=coordinates, dims=dimensions)
    for i in range(0, nrlakes):
        starting_stage.values[i]=list_of_lakes[i].starting_stage
        boundname.values[i]=list_of_lakes[i].boundname

        connectionType = list_of_lakes[i].get_1d_array("connectionType")
'''
        self.bed_leak = bed_leak
        self.top_elevation = top_elevation
        self.bot_elevation = bot_elevation
        self.connection_length = connection_length
        self.connection_width = connection_width

'''


