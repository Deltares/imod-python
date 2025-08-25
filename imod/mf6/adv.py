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
from typing import Optional

import numpy as np

from imod.common.interfaces.iregridpackage import IRegridPackage
from imod.mf6.package import Package
from imod.schemata import AllValueSchema, DimsSchema, DTypeSchema


class Advection(Package, IRegridPackage):
    _pkg_id = "adv"
    _template = Package._initialize_template(_pkg_id)

    _init_schemata = {
        "ats_percel": [
            DimsSchema(),
            DTypeSchema(np.floating),
            AllValueSchema(">", 0.0),
        ],
    }

    def _render(self, directory, pkgname, globaltimes, binary):
        render_dict = {}
        render_dict["scheme"] = self.dataset["scheme"].item()
        if "ats_percel" in self.dataset and self._valid(
            self.dataset["ats_percel"].item()
        ):
            render_dict["ats_percel"] = self.dataset["ats_percel"].item()
        return self._template.render(render_dict)

    def mask(self, _) -> Package:
        """
        The mask method is irrelevant for this package , instead this method
        retuns a copy of itself.
        """
        return deepcopy(self)


class AdvectionUpstream(Advection):
    """
    The upstream weighting (first order upwind) scheme sets the concentration
    at the cellface between two adjacent cells equal to the concentration in
    the cell where the flow comes from. It surpresses oscillations.

    Parameters
    ----------
    ats_percel: float, optional
        Fractional cell distance submitted by the ADV Package to the
        :class:`imod.mf6.AdaptiveTimeStepping` (ATS) package. If ``ats_percel``
        is specified and the ATS Package is active, a time step calculation will
        be made for each cell based on flow through the cell and cell
        properties. The largest time step will be calculated such that the
        advective fractional cell distance (``ats_percel``) is not exceeded for
        any active cell in the grid. This time-step constraint will be submitted
        to the ATS Package, perhaps with constraints submitted by other
        packages, in the calculation of the time step. ``ats_percel`` must be
        greater than zero. If a value of zero is specified for ``ats_percel``
        the program will automatically reset it to an internal no data value to
        indicate that time steps should not be subject to this constraint.
        Requires MODFLOW 6.6.0 or higher.
    validate: bool, optional
        Validate the package upon initialization. Defaults to True.
    """

    def __init__(self, ats_percel: Optional[float] = None, validate: bool = True):
        dict_dataset = {"scheme": "upstream", "ats_percel": ats_percel}
        super().__init__(dict_dataset)
        self._validate_init_schemata(validate)


class AdvectionCentral(Advection):
    """
    The central-in-space weighting scheme is based on a simple
    distance-weighted linear interpolation between the center of cell n and the
    center of cell m to calculate solute concentration at the shared face
    between cell n and cell m. Although central-in-space is a misnomer for
    grids without equal spacing between connected cells, it is retained here
    for consistency with nomenclature used by other MODFLOW-based transport
    programs, such as MT3D.

    Parameters
    ----------
    ats_percel: float, optional
        Fractional cell distance submitted by the ADV Package to the
        :class:`imod.mf6.AdaptiveTimeStepping` (ATS) package. If ``ats_percel``
        is specified and the ATS Package is active, a time step calculation will
        be made for each cell based on flow through the cell and cell
        properties. The largest time step will be calculated such that the
        advective fractional cell distance (``ats_percel``) is not exceeded for
        any active cell in the grid. This time-step constraint will be submitted
        to the ATS Package, perhaps with constraints submitted by other
        packages, in the calculation of the time step. ``ats_percel`` must be
        greater than zero. If a value of zero is specified for ``ats_percel``
        the program will automatically reset it to an internal no data value to
        indicate that time steps should not be subject to this constraint.
        Requires MODFLOW 6.6.0 or higher.
    validate: bool, optional
        Validate the package upon initialization. Defaults to True.
    """

    def __init__(self, ats_percel: Optional[float] = None, validate: bool = True):
        dict_dataset = {"scheme": "central", "ats_percel": ats_percel}
        super().__init__(dict_dataset)
        self._validate_init_schemata(validate)


class AdvectionTVD(Advection):
    """
    An implicit second order TVD scheme. More expensive than upstream
    weighting but more robust.

    Parameters
    ----------
    ats_percel: float, optional
        Fractional cell distance submitted by the ADV Package to the
        :class:`imod.mf6.AdaptiveTimeStepping` (ATS) package. If ``ats_percel``
        is specified and the ATS Package is active, a time step calculation will
        be made for each cell based on flow through the cell and cell
        properties. The largest time step will be calculated such that the
        advective fractional cell distance (``ats_percel``) is not exceeded for
        any active cell in the grid. This time-step constraint will be submitted
        to the ATS Package, perhaps with constraints submitted by other
        packages, in the calculation of the time step. ``ats_percel`` must be
        greater than zero. If a value of zero is specified for ``ats_percel``
        the program will automatically reset it to an internal no data value to
        indicate that time steps should not be subject to this constraint.
        Requires MODFLOW 6.6.0 or higher.
    validate: bool, optional
        Validate the package upon initialization. Defaults to True.
    """

    def __init__(self, ats_percel: Optional[float] = None, validate: bool = True):
        dict_dataset = {"scheme": "TVD", "ats_percel": ats_percel}
        super().__init__(dict_dataset)
        self._validate_init_schemata(validate)
