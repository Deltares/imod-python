from imod import io, pre
from imod.model import SeawatModel

# These were previously in `imod`, now under `imod.io`, make them available in
# `imod` to avoid breaking changes. We will likely keep them here since
# `imod.idf.load()` is less typing and does not pollute the namespace too much.
from imod.io import idf, ipf, run, tec, util

# these are defined in __init__
from imod.io import write, seawat_write

# since this is a big dependency that is sometimes hard to install
# and not always required, we made this an optional dependency
try:
    from imod.io import rasterio
except ImportError:
    pass

from imod._version import get_versions

__version__ = get_versions()["version"]
del get_versions
