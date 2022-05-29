from imod.mf6.pkgbase import Package

"""
    When simulating transport, modflow needs to compute the concentration at a cellface between 2 adjacent cells.
    It supports 3 ways of doing this. Each of those has its own wrapper class. These numerical schemes differ in how much
    numerical dispersion they cause, how much oscillations, and how timestep and grid size affect stability.
    Central-in-space weighting is not often used because it can result in spurious oscillations in the simulated concentrations.
    Upstream weigthing is a fast alternative, and TVD is a more expensive and more robust alternative.
"""


"""
    The upstream weighting (first order upwind) scheme sets the concentration at the cellface between 2 adjacent cells equal to the concentration
    in the cell where the flow comes from. surpresses oscilations.
"""


class AdvectionUpstream(Package):
    _pkg_id = "adv"
    _template = Package._initialize_template(_pkg_id)

    def __init__(self):
        pass

    def render(self, directory, pkgname, globaltimes, binary):
        return self._template.render({"scheme": "upstream"})


"""
    The central-in-space weighting scheme is based on a simple distance-weighted linear interpolation
    between the center of cell n and the center of cell m to calculate solute concentration at the shared face
    between cell n and cell m. Although central-in-space is a misnomer for grids without equal spacing between
    connected cells, it is retained here for consistency with nomenclature used by other MODFLOW-based transport programs, such as MT3D.
"""


class AdvectionCentral(Package):
    _pkg_id = "adv"
    _template = Package._initialize_template(_pkg_id)

    def __init__(self):
        pass

    def render(self, directory, pkgname, globaltimes, binary):
        return self._template.render({"scheme": "central"})


"""
   An implicit second order TVD scheme. More exensive than upstream weighting but more robust.
"""


class AdvectionTVD(Package):

    _pkg_id = "adv"
    _template = Package._initialize_template(_pkg_id)

    def __init__(self):
        pass

    def render(self, directory, pkgname, globaltimes, binary):
        return self._template.render({"scheme": "TVD"})
