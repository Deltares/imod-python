# without these this usage wouldn't work:
# import imod
# imod.idf.*
from imod import idf
from imod import ipf
from imod import tec
from imod import util

# since this is a big dependency that is sometimes hard to install
# and not always required, we made this an optional dependency
try:
    from imod import rasterio
except ImportError:
    pass

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
