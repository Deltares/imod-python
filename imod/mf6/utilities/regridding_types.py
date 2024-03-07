from enum import Enum

import xugrid as xu


class RegridderType(Enum):
    """
    Enumerator referring to regridder types in ``xugrid``.
    These can be used safely in scripts, remaining backwards compatible for
    when it is decided to rename regridders in ``xugrid``. For an explanation
    what each regridder type does, we refer to the `xugrid documentation <https://deltares.github.io/xugrid/examples/regridder_overview.html>`_
    """

    CENTROIDLOCATOR = xu.CentroidLocatorRegridder
    BARYCENTRIC = xu.BarycentricInterpolator
    OVERLAP = xu.OverlapRegridder
    RELATIVEOVERLAP = xu.RelativeOverlapRegridder