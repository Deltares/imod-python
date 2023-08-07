from importlib.metadata import distribution, PackageNotFoundError
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
    __version__ = distribution(__name__).version
except PackageNotFoundError:
    # package is not installed
    pass
