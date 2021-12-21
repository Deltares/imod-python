"""
Functions to load sample data.
"""
from pathlib import Path
from typing import Union
from zipfile import ZipFile

import pkg_resources
import pooch
import xarray as xr
import pandas as pd

REGISTRY = pooch.create(
    path=pooch.os_cache("imod"),
    base_url="https://gitlab.com/deltares/imod/imod-artifacts/-/raw/main/",
    version=None,
    version_dev="main",
    env="IMOD_DATA_DIR",
)
with pkg_resources.resource_stream("imod.data", "registry.txt") as registry_file:
    REGISTRY.load_registry(registry_file)


def twri_output(path: Union[str, Path]) -> None:
    fname_twri = REGISTRY.fetch("ex01-twri-output.zip")
    with ZipFile(fname_twri) as archive:
        archive.extractall(path)


def hondsrug_initial() -> xr.Dataset:
    fname = REGISTRY.fetch("hondsrug-initial.nc")
    return xr.open_dataset(fname)


def hondsrug_layermodel() -> xr.Dataset:
    fname = REGISTRY.fetch("hondsrug-layermodel.nc")
    return xr.open_dataset(fname)


def hondsrug_meteorology() -> xr.Dataset:
    fname = REGISTRY.fetch("hondsrug-meteorology.nc")
    return xr.open_dataset(fname)


def hondsrug_river() -> xr.Dataset:
    fname = REGISTRY.fetch("hondsrug-river.nc")
    return xr.open_dataset(fname)


def hondsrug_drainage() -> xr.Dataset:
    fname = REGISTRY.fetch("hondsrug-drainage.nc")
    return xr.open_dataset(fname)


def head_observations() -> pd.DataFrame:
    fname = REGISTRY.fetch("head-observations.csv")
    return pd.read_csv(fname)
