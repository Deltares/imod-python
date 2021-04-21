from imod.flow.pkgbase import Package, Vividict
import imod
import numpy as np
from imod.wq import timeutil


class HorizontalFlowBoundary(Package):
    """
    Horizontal barriers obstructing flow such as semi- or impermeable fault zone or a sheet pile wall are
    defined for each model layer by a *.GEN line file.

    Parameters
    ----------
    id_name: str or list of str
        name of the barrier
    geometry: object array of shapely LineStrings
        geometry of barriers, should be lines
    layer: "None" or int
        layer where barrier is located
    resistance: float or list of floats
        resistance of the barrier (d).
    """

    _pkg_id = ["wel"]
    _variable_order = ["resistance"]

    # TODO 1. Write save function
    # TODO 2. Compose package
    # TODO 3. Create template (Note that the multiplication factor is used as the resistance)
    # TODO 4. Render
