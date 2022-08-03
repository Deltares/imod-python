from mercantile import parent


class LakeLake:
    def __init__(self, starting_stage: float, boundname: str, connectionType,bed_leak,top_elevation, bot_elevation,
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

        #todo: check input

class OutletBase:
    def __init__(self,outletNumber: int, lakeIn: str, lakeOut: str ):
        self.outletNumber = outletNumber
        self.lakeIn= lakeIn
        self.lakeOut = lakeOut


class OutletManning (OutletBase):
    def __init__(self, outletNumber: int, lakeIn: str, lakeOut: str, invert, width, roughness, slope):
        parent.__init__(outletNumber,lakeIn,lakeOut )
        self.invertt = invert
        self.width = width
        self.roughness = roughness
        self.slope = slope



class OutletWeir(OutletBase):
    def __init__(self, outletNumber: int, lakeIn: str, lakeOut: str,invert, width,  ):
        parent.__init__(outletNumber,lakeIn,lakeOut )
        self.invertt = invert
        self.width = width


class OutletSpecified(OutletBase):
    def __init__(self, outletNumber: int, lakeIn: str, lakeOut: str, rate):
        parent.__init__(outletNumber,lakeIn,lakeOut )
        self.rate = rate