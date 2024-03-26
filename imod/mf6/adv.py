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

from copy import deepcopy
from typing import Optional, Tuple

from imod.mf6.interfaces.iregridpackage import IRegridPackage
from imod.mf6.package import Package
from imod.mf6.utilities.regrid import RegridderType


class Advection(Package, IRegridPackage):
    _pkg_id = "adv"
    _template = Package._initialize_template(_pkg_id)
    _regrid_method: dict[str, tuple[RegridderType, str]] = {}

    def __init__(self, scheme: str):
        dict_dataset = {"scheme": scheme}
        super().__init__(dict_dataset)

    def render(self, directory, pkgname, globaltimes, binary):
        scheme = self.dataset["scheme"].item()
        return self._template.render({"scheme": scheme})

    def mask(self, _) -> Package:
        """
        The mask method is irrelevant for this package , instead this method
        retuns a copy of itself.
        """
        return deepcopy(self)

    def get_regrid_methods(self) -> Optional[dict[str, Tuple[RegridderType, str]]]:
        return self._regrid_method


class AdvectionUpstream(Advection):
    """
    The upstream weighting (first order upwind) scheme sets the concentration
    at the cellface between two adjacent cells equal to the concentration in
    the cell where the flow comes from. It surpresses oscillations.
    Note: all constructor arguments will be ignored
    """

    def __init__(self, scheme: str = "upstream"):
        if not scheme == "upstream":
            raise ValueError(
                "error in scheme parameter. Should be 'upstream' if present."
            )
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
    Note: all constructor arguments will be ignored
    """

    def __init__(self, scheme: str = "central"):
        if not scheme == "central":
            raise ValueError(
                "error in scheme parameter. Should be 'central' if present."
            )
        super().__init__(scheme="central")


class AdvectionTVD(Advection):
    """
    An implicit second order TVD scheme. More expensive than upstream
    weighting but more robust.
    Note: all constructor arguments will be ignored
    """

    def __init__(self, scheme: str = "TVD"):
        if not scheme == "TVD":
            raise ValueError("error in scheme parameter. Should be 'TVD' if present.")
        super().__init__(scheme="TVD")
