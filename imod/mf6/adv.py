"""
When simulating transport, MODFLOW6 needs to compute the concentration at a
cellface between 2 adjacent cells. It supports 3 ways of doing this. Each of
those has its own wrapper class. These numerical schemes differ in how much
numerical dispersion they cause, how much oscillations, and how timestep and
grid size affect stability. Central-in-space weighting is not often used
because it can result in spurious oscillations in the simulated concentrations.
Upstream weighting is a fast alternative, and TVD is a more expensive and more
robust alternative.
"""
from imod.mf6.pkgbase import Package


class AdvectionUpstream(Package):
    """
    The upstream weighting (first order upwind) scheme sets the concentration
    at the cellface between two adjacent cells equal to the concentration in
    the cell where the flow comes from. It surpresses oscillations.
    """

    _pkg_id = "adv"
    _template = Package._initialize_template(_pkg_id)

    def __init__(self):
        pass

    def render(self, directory, pkgname, globaltimes, binary):
        return self._template.render({"scheme": "upstream"})


class AdvectionCentral(Package):
    """
    The central-in-space weighting scheme is based on a simple
    distance-weighted linear interpolation between the center of cell n and the
    center of cell m to calculate solute concentration at the shared face
    between cell n and cell m. Although central-in-space is a misnomer for
    grids without equal spacing between connected cells, it is retained here
    for consistency with nomenclature used by other MODFLOW-based transport
    programs, such as MT3D.
    """

    _pkg_id = "adv"
    _template = Package._initialize_template(_pkg_id)

    def __init__(self):
        pass

    def render(self, directory, pkgname, globaltimes, binary):
        return self._template.render({"scheme": "central"})


class AdvectionTVD(Package):
    """
    An implicit second order TVD scheme. More expensive than upstream
    weighting but more robust.
    """

    _pkg_id = "adv"
    _template = Package._initialize_template(_pkg_id)

    def __init__(self):
        pass

    def render(self, directory, pkgname, globaltimes, binary):
        return self._template.render({"scheme": "TVD"})
