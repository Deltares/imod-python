# -*- coding: utf-8 -*-
from imod.flow.pkgbase import Package


class HorizontalAnisotropy(Package):
    """
    Horizontal anisotropy is a phenomenon in which the horizontal hydraulic
    conductivity is not equal along the x and y Cartesian axes. iMODFLOW can
    calculate this anisotropy based on a anisotropy factor and an anisotropy
    angle. iMODFLOW also accounts for the cross-terms in the horizontal
    hydraulic conductivity tensor.

    See also section 12.14 "ANI Horizontal anisotropy module" in the iMOD v5.2
    manual for further explanation.

    Parameters
    ----------
    anisotropy_factor : xr.DataArray
        The anisotropy factor is defined perpendicular to the main principal
        axis. The factor is between 0.0 (full anisotropic) and 1.0 (full isotropic)
    anisotropy_angle : xr.DataArray
        The anisotropy angle denotes the angle along the main principal axis
        (highest permeability k) measured in degrees from
        north (0°), east (90°), south (180°) and west (270°).
    """

    _pkg_id = "ani"
    _variable_order = ["anisotropy_factor", "anisotropy_angle"]

    def __init__(self, anisotropy_factor=None, anisotropy_angle=None):
        super(__class__, self).__init__()
        self.dataset["anisotropy_factor"] = anisotropy_factor
        self.dataset["anisotropy_angle"] = anisotropy_angle
