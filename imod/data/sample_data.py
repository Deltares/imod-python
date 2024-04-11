"""
Functions to load sample data.
"""

import importlib
from pathlib import Path
from typing import Union
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pooch
import xarray as xr
import xugrid as xu

from imod.formats.prj import open_projectfile_data
from imod.mf6 import Modflow6Simulation
from imod.util.imports import MissingOptionalModule

try:
    import geopandas as gpd
except ImportError:
    gpd = MissingOptionalModule("geopandas")


def create_pooch_registry() -> pooch.core.Pooch:
    registry = pooch.create(
        path=pooch.os_cache("imod"),
        base_url="https://github.com/Deltares/imod-artifacts/raw/main/",
        version=None,
        version_dev="main",
        env="IMOD_DATA_DIR",
    )
    return registry


def load_pooch_registry(registry: pooch.core.Pooch) -> pooch.core.Pooch:
    with importlib.resources.files("imod.data") as pkg_dir:
        registry.load_registry(pkg_dir / "registry.txt")
    return registry


registry = create_pooch_registry()
registry = load_pooch_registry(registry)


def twri_output(path: Union[str, Path]) -> None:
    fname_twri = registry.fetch("ex01-twri-output.zip")
    with ZipFile(fname_twri) as archive:
        archive.extractall(path)


def hondsrug_initial() -> xr.Dataset:
    fname = registry.fetch("hondsrug-initial.nc")
    return xr.open_dataset(fname)


def hondsrug_layermodel() -> xr.Dataset:
    fname = registry.fetch("hondsrug-layermodel.nc")
    return xr.open_dataset(fname)


def hondsrug_meteorology() -> xr.Dataset:
    fname = registry.fetch("hondsrug-meteorology.nc")
    return xr.open_dataset(fname)


def hondsrug_river() -> xr.Dataset:
    fname = registry.fetch("hondsrug-river.nc")
    return xr.open_dataset(fname)


def hondsrug_drainage() -> xr.Dataset:
    fname = registry.fetch("hondsrug-drainage.nc")
    return xr.open_dataset(fname)


def head_observations() -> pd.DataFrame:
    fname = registry.fetch("head-observations.csv")
    df = pd.read_csv(fname)
    # Manually convert time column to datetime type because pandas >2.0 doesn't
    # do this automatically anymore upon reading.
    df["time"] = pd.to_datetime(df["time"])
    return df


def fluxes() -> xr.Dataset:
    fname = registry.fetch("fluxes.nc")
    return xr.open_dataset(fname)


def ahn() -> xr.Dataset:
    fname = registry.fetch("ahn.nc")
    return xr.open_dataset(fname)


def lakes_shp(path: Union[str, Path]) -> "geopandas.GeoDataFrame":  # type: ignore # noqa
    fname_lakes_shp = registry.fetch("lakes_shp.zip")
    with ZipFile(fname_lakes_shp) as archive:
        archive.extractall(path)
    return gpd.read_file(Path(path) / "lakes.shp")


def circle() -> xu.Ugrid2d:
    fname_nodes = registry.fetch("circle-nodes.txt")
    fname_triangles = registry.fetch("circle-triangles.txt")

    nodes = np.loadtxt(fname_nodes)
    triangles = np.loadtxt(fname_triangles).astype(np.int32)

    return xu.Ugrid2d(*nodes.T, -1, triangles)


def imod5_projectfile_data(path: Union[str, Path]) -> dict:
    fname_model = registry.fetch("iMOD5_model.zip")

    with ZipFile(fname_model) as archive:
        archive.extractall(path)

    return open_projectfile_data(Path(path) / "iMOD5_model_pooch" / "iMOD5_model.prj")


def hondsrug_simulation(path: Union[str, Path]) -> Modflow6Simulation:
    fname_simulation = registry.fetch("hondsrug-simulation.zip")

    with ZipFile(fname_simulation) as archive:
        archive.extractall(path)

    return Modflow6Simulation.from_file(Path(path) / "mf6-hondsrug-example.toml")


def hondsrug_crosssection(path: Union[str, Path]) -> "geopandas.GeoDataFrame":  # type: ignore # noqa
    fname_simulation = registry.fetch("hondsrug-crosssection.zip")

    with ZipFile(fname_simulation) as archive:
        archive.extractall(path)

    return gpd.read_file(Path(path) / "crosssection.shp")
