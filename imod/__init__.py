import pkg_resources

# exports
from imod import (
    couplers,
    data,
    evaluate,
    flow,
    mf6,
    msw,
    prepare,
    select,
    testing,
    util,
    visualize,
    wq,
)
from imod.formats import gen, idf, ipf, prj, rasterio

# version
try:
    __version__ = pkg_resources.get_distribution(__name__).version
except pkg_resources.DistributionNotFound:
    # package is not installed
    pass
