from imod.flow.pkgbase import TopBoundaryCondition


class EvapoTranspiration(TopBoundaryCondition):
    """
    Recharge provides a fixed flux boundary condition to the top layer of the
    groundwater system. Note that unlike in iMOD-WQ, there is only the option
    in iMODFLOW to apply the recharge package to the top layer.

    Parameters
    ----------
    rate: float or xr.DataArray of floats
        evaporation rate in mm/day (NOTA BENE!), dims ``("time", "y", "x")``.
    top_elevation: floats or xr.DataArray of floats
        Top elevation in m+MSL for maximal evapotranspiration strength.
    extinction_depth: float or xr.Datarray of floats
        Depth [m] in which evapotranspiration strength reduced to zero.
    """

    _pkg_id = "evt"
    _variable_order = ["rate", "top_elevation", "extinction_depth"]

    def __init__(self, rate, top_elevation, extinction_depth):
        super(__class__, self).__init__()
        self.dataset["rate"] = rate
        self.dataset["top_elevation"] = top_elevation
        self.dataset["extinction_depth"] = extinction_depth
