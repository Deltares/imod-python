from imod.wq.pkgbase import BoundaryCondition


class MassLoading(BoundaryCondition):
    """
    Mass loading package. Has no direct effect on groundwater flow, is only
    included via MT3DMS source and sinks. (SSM ITYPE 15)

    Parameters
    ----------
    concentration: array of floats (xr.DataArray)
    """

    __slots__ = ("concentration",)
    _pkg_id = "mal"

    def __init__(self, concentration):
        self["concentration"] = concentration

    def add_timemap(self, concentration, use_cftime=False):
        self._add_timemap("concentration", concentration, use_cftime)

    def _pkgcheck(self, ibound=None):
        self._check_positive("concentration")
