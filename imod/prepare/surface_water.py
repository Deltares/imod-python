# Import for backwards compatibility
import warnings

from imod.prepare.topsystem import c_leakage as _c_leakage
from imod.prepare.topsystem import c_radial as _c_radial

WARNING_MESSAGE = (
    "function has been moved from imod.prepare.surface_water to"
    "imod.prepare.topsystem, please update your scripts."
    "imod.prepare.surface_water is going to be removed in version 1.0"
)


def c_radial(L, kh, kv, B, D):
    warnings.warn(WARNING_MESSAGE, DeprecationWarning)
    return _c_radial(L, kh, kv, B, D)


def c_leakage(kh, kv, D, c0, c1, B, length, dx, dy):
    warnings.warn(WARNING_MESSAGE, DeprecationWarning)
    return _c_leakage(kh, kv, D, c0, c1, B, length, dx, dy)
