import pkg_resources

# exports
from imod import (
    couplers,
    data,
    evaluate,
    flow,
    gen,
    idf,
    ipf,
    mf6,
    msw,
    prepare,
    rasterio,
    select,
    util,
    visualize,
    wq,
)
from imod.formats import prj

# version
try:
    __version__ = pkg_resources.get_distribution(__name__).version
except pkg_resources.DistributionNotFound:
    # package is not installed
    pass
