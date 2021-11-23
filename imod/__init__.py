import pkg_resources

# exports
from imod import (
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

# version
try:
    __version__ = pkg_resources.get_distribution(__name__).version
except pkg_resources.DistributionNotFound:
    # package is not installed
    pass
