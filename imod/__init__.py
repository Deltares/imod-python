import pathlib
import warnings

import numpy as np
import pandas as pd
import pkg_resources

# subpackages
import imod.evaluate
import imod.flow
import imod.mf6
import imod.prepare
import imod.select
import imod.visualize
import imod.wq

# submodules
from imod import idf, ipf, rasterio, tec, util
from imod.data_formats import gen

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except pkg_resources.DistributionNotFound:
    # package is not installed
    pass
