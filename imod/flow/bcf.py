from imod.flow.pkgbase import Package

class Transmissivity(Package):
    """Transmissivity of the aquifer [L^2/T], 
    defined as the thickness multiplied by the horizontal hydraulic 
    conductivity. Using this package means you follow MODFLOW 2005's
    Quasi-3D "Block Centred Flow (BCF)" schematization, 
    meaning you cannot assign a vertical hydraulic conductivity as well. 
    Instead use the vertical resistance.

    Parameters
    ----------
    transmissivity : xr.DataArray
        Transmissivity of the aquifer [L^2/T]
    """

    _pkg_id = "kdw"
    _variable_order = ["transmissivity"]

    def __init__(self, transmissivity=None):
        super(__class__, self).__init__()
        self.dataset["transmissivity"] = transmissivity

class VerticalResistance(Package):
    """Vertical Resistance of the aquitard [T],
    defined as the thickness of the aquitard divided by the 
    vertical horizontal conductivity. 
    
    Using this package means you follow MODFLOW 2005's
    Quasi-3D "Block Centred Flow (BCF)" schematization, 
    meaning you cannot assign a horizontal hydraulic conductivity as well. 
    Instead use the transmissivity.

    Parameters
    ----------
    vertical_resistance : xr.DataArray
        Vertical resitance over the aquitard [T]
    """

    _pkg_id = "vcw"
    _variable_order = ["vertical_resistance"]

    def __init__(self, vertical_resistance=None):
        super(__class__, self).__init__()
        self.dataset["vertical_resistance"] = vertical_resistance