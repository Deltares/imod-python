"""
Functions to load sample data.
"""
import numpy as np
import pandas as pd
import pkg_resources
import pooch
import xarray as xr

import imod

REGISTRY = pooch.create(
    path=pooch.os_cache("imod"),
    base_url="https://github.com/deltares/xugrid/raw/main/data/",
    version=None,
    version_dev="main",
    env="IMOD_DATA_DIR",
)
with pkg_resources.resource_stream("imod.data", "registry.txt") as registry_file:
    REGISTRY.load_registry(registry_file)
