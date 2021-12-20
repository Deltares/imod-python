import pkg_resources

# exports
from imod import (
    data,
    evaluate,
    flow,
    idf,
    ipf,
    mf6,
    prepare,
    rasterio,
    select,
    util,
    visualize,
    wq,
)
from imod.data_formats import gen

# version
try:
    __version__ = pkg_resources.get_distribution(__name__).version
except pkg_resources.DistributionNotFound:
    # package is not installed
    pass
