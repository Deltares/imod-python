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


class Advection(Package):
    _pkg_id = "adv"
    _template = Package._initialize_template(_pkg_id)

    def __init__(self, scheme):
        super().__init__()
        self.dataset["scheme"] = scheme

    def render(self, directory, pkgname, globaltimes, binary):
        scheme = self.dataset["scheme"].item()
        return self._template.render({"scheme": scheme})

    @classmethod
    def _from_file(cls, path, **kwargs):
        classes = {
            "upstream": AdvectionUpstream,
            "central": AdvectionCentral,
            "tvd": AdvectionTVD,
        }
        pkg_kwargs = cls._open_dataset(path, **kwargs)
        scheme = pkg_kwargs["scheme"].item().lower()
        adv_cls = classes[scheme]
        return adv_cls()


class AdvectionUpstream(Advection):
    """
    The upstream weighting (first order upwind) scheme sets the concentration
    at the cellface between two adjacent cells equal to the concentration in
    the cell where the flow comes from. It surpresses oscillations.
    """

    def __init__(self):
        super().__init__(scheme="upstream")


class AdvectionCentral(Advection):
    """
    The central-in-space weighting scheme is based on a simple
    distance-weighted linear interpolation between the center of cell n and the
    center of cell m to calculate solute concentration at the shared face
    between cell n and cell m. Although central-in-space is a misnomer for
    grids without equal spacing between connected cells, it is retained here
    for consistency with nomenclature used by other MODFLOW-based transport
    programs, such as MT3D.
    """

    def __init__(self):
        super().__init__(scheme="central")


class AdvectionTVD(Advection):
    """
    An implicit second order TVD scheme. More expensive than upstream
    weighting but more robust.
    """

    def __init__(self):
        super().__init__(scheme="TVD")
