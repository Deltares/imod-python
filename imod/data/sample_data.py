"""
Functions to load sample data.
"""

import importlib
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pooch
import xarray as xr
import xugrid as xu
from filelock import FileLock
from pooch import Unzip

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
    with importlib.resources.as_file(importlib.resources.files("imod.data")) as pkg_dir:
        registry.load_registry(pkg_dir / "registry.txt")
    return registry


REGISTRY = create_pooch_registry()
REGISTRY = load_pooch_registry(REGISTRY)


def twri_output(path: Union[str, Path]) -> None:
    # Unzips TWRI output to ``path``. Has a race condition when executed
    # multiple times with the same path.
    lock = FileLock(REGISTRY.path / "ex01-twri-output.zip.lock")
    with lock:
        _ = REGISTRY.fetch("ex01-twri-output.zip", processor=Unzip(extract_dir=path))


def hondsrug_initial() -> xr.Dataset:
    lock = FileLock(REGISTRY.path / "hondsrug-initial.nc.lock")
    with lock:
        fname = REGISTRY.fetch("hondsrug-initial.nc")
        hondsrug_initial = xr.open_dataset(fname)
    return hondsrug_initial


def hondsrug_layermodel() -> xr.Dataset:
    lock = FileLock(REGISTRY.path / "hondsrug-layermodel.nc.lock")
    with lock:
        fname = REGISTRY.fetch("hondsrug-layermodel.nc")
        hondsrug_layermodel = xr.open_dataset(fname)

    return hondsrug_layermodel


def hondsrug_meteorology() -> xr.Dataset:
    lock = FileLock(REGISTRY.path / "hondsrug-meteorology.lock")
    with lock:
        fname = REGISTRY.fetch("hondsrug-meteorology.nc")
        hondsrug_meteorology = xr.open_dataset(fname)
    return hondsrug_meteorology


def hondsrug_river() -> xr.Dataset:
    lock = FileLock(REGISTRY.path / "hondsrug-river.nc.lock")
    with lock:
        fname = REGISTRY.fetch("hondsrug-river.nc")
        hondsrug_river = xr.open_dataset(fname)
    return hondsrug_river


def hondsrug_drainage() -> xr.Dataset:
    lock = FileLock(REGISTRY.path / "hondsrug-drainage.nc.lock")
    with lock:
        fname = REGISTRY.fetch("hondsrug-drainage.nc")
        hondsrug_drainage = xr.open_dataset(fname)
    return hondsrug_drainage


def head_observations() -> pd.DataFrame:
    lock = FileLock(REGISTRY.path / "head-observations.csv.lock")
    with lock:
        fname = REGISTRY.fetch("head-observations.csv")
        head_observations = pd.read_csv(fname)

    # Manually convert time column to datetime type because pandas >2.0 doesn't
    # do this automatically anymore upon reading.
    head_observations["time"] = pd.to_datetime(head_observations["time"])
    return head_observations


def fluxes() -> xr.Dataset:
    lock = FileLock(REGISTRY.path / "fluxes.nc.lock")
    with lock:
        fname = REGISTRY.fetch("fluxes.nc")
        fluxes = xr.open_dataset(fname)
    return fluxes


def ahn() -> xr.Dataset:
    lock = FileLock(REGISTRY.path / "ahn.nc.lock")
    with lock:
        fname = REGISTRY.fetch("ahn.nc")
        ahn = xr.open_dataset(fname)
    return ahn


def lakes_shp(path: Union[str, Path]) -> "geopandas.GeoDataFrame":  # type: ignore # noqa
    lock = FileLock(REGISTRY.path / "lakes_shp.zip.lock")
    with lock:
        fnames = REGISTRY.fetch("lakes_shp.zip", processor=Unzip(extract_dir=path))
        shape_file = next(filter(lambda files: "lakes.shp" in files, fnames))
        lakes = gpd.read_file(shape_file)
    return lakes


def _circle_nodes():
    lock = FileLock(REGISTRY.path / "circle-nodes.txt.lock")
    with lock:
        fname_nodes = REGISTRY.fetch("circle-nodes.txt")
        nodes = np.loadtxt(fname_nodes)

    return nodes


def _circle_triangles():
    lock = FileLock(REGISTRY.path / "circle-triangles.txt.lock")
    with lock:
        fname_triangles = REGISTRY.fetch("circle-triangles.txt")
        triangles = np.loadtxt(fname_triangles).astype(np.int32)

    return triangles


def circle() -> xu.Ugrid2d:
    nodes = _circle_nodes()
    triangles = _circle_triangles()

    return xu.Ugrid2d(*nodes.T, -1, triangles)


def imod5_projectfile_data(path: Union[str, Path]) -> dict:
    lock = FileLock(REGISTRY.path / "iMOD5_model.zip.lock")
    with lock:
        _ = REGISTRY.fetch("iMOD5_model.zip", processor=Unzip(extract_dir=path))
        iMOD5_model = open_projectfile_data(
            Path(path) / "iMOD5_model_pooch" / "iMOD5_model.prj"
        )

    return iMOD5_model


def hondsrug_simulation(path: Union[str, Path]) -> Modflow6Simulation:
    lock = FileLock(REGISTRY.path / "hondsrug-simulation.zip.lock")
    with lock:
        _ = REGISTRY.fetch("hondsrug-simulation.zip", processor=Unzip(extract_dir=path))

        simulation = Modflow6Simulation.from_file(
            Path(path) / "mf6-hondsrug-example.toml"
        )
        # The model was written before the xt3d_option and rhs_option arguments were
        # added to iMOD Python. Set missing options to False.
        simulation["GWF"]["npf"].set_xt3d_option(is_xt3d_used=False, is_rhs=False)

    return simulation


def hondsrug_crosssection(path: Union[str, Path]) -> "geopandas.GeoDataFrame":  # type: ignore # noqa
    lock = FileLock(REGISTRY.path / "hondsrug-crosssection.zip.lock")
    with lock:
        fnames = REGISTRY.fetch(
            "hondsrug-crosssection.zip", processor=Unzip(extract_dir=path)
        )
        shape_file = next(filter(lambda files: "crosssection.shp" in files, fnames))
        crosssection = gpd.read_file(shape_file)

    return crosssection


def hondsrug_layermodel_topsystem() -> xr.Dataset:
    """
    This is a modified version of the hondsrug_layermodel, used for the
    topsystem example in the user guide. n_max_old original layers are
    taken and subdivided into n_new layers. This makes for more layers around
    the topsystem.
    """
    layer_model = hondsrug_layermodel()
    # Make layer model more interesting for this example by subdividing layers
    # into n_new layers.
    n_new = 4
    n_max_old = 5

    # Loop over original layers until n_max_old and subdivide each into n_new
    # layers.
    new_ds_ls = []
    for i in range(n_max_old):
        sub_iter = np.arange(n_new) + 1
        layer_coord = sub_iter + i * (n_max_old - 1)
        distribution_factors = 1 / n_new * sub_iter
        da_distribution = xr.DataArray(
            distribution_factors, coords={"layer": layer_coord}, dims=("layer",)
        )
        layer_model_sel = layer_model.sel(layer=i + 1, drop=True)
        # Compute thickness
        D = layer_model_sel["top"] - layer_model_sel["bottom"]

        new_ds = xr.Dataset()
        new_ds["k"] = xr.ones_like(da_distribution) * layer_model_sel["k"]
        new_ds["idomain"] = xr.ones_like(da_distribution) * layer_model_sel["idomain"]
        # Put da_distribution in front of equation to enforce dims as (layer, y, x)
        new_ds["top"] = (da_distribution - 1 / n_new) * -D + layer_model_sel["top"]
        new_ds["bottom"] = da_distribution * -D + layer_model_sel["top"]

        new_ds_ls.append(new_ds)

    return xr.concat(new_ds_ls, dim="layer")


def hondsrug_planar_river() -> xr.Dataset:
    """
    This is the hondsrug river dataset with the following modifications:

    1) Aggregated over layer dimension to create planar grid.
    2) Stages raised towards the top of the model, as the original stages are
       most of the time laying at bottom elevation, making for boring examples.

    """
    river = hondsrug_river()
    planar_river = river.max(dim="layer")

    layer_model = hondsrug_layermodel()
    top = layer_model["top"].sel(layer=1)

    planar_river["stage"] = (top - planar_river["stage"]) / 2 + planar_river["stage"]

    return planar_river


def colleagues_river_data(path: Union[str, Path]):
    """
    River data with some mistakes introduced to showcase cleanup
    utilities.
    """
    gwf_model = hondsrug_simulation(path)["GWF"]
    dis_ds = gwf_model["dis"].dataset
    riv_ds_old = gwf_model["riv"].dataset
    # Existing RIV package has only layer coord with [1, 3, 5, 6]. This causes
    # problems with some methods, at least with cleanup. Therefore align data
    # here for the cleanup example. River packages with limited layer coords are
    # currently not fully supported.
    riv_ds, _ = xr.align(riv_ds_old, dis_ds, join="outer")
    x = riv_ds.coords["x"]
    y = riv_ds.coords["y"]
    riv_bot_da = riv_ds["bottom_elevation"]
    riv_ds["stage"] += 0.05
    riv_ds["stage"] = riv_ds["stage"].where(x > 239500)
    riv_ds["conductance"] = riv_ds["conductance"].fillna(0.0)
    x_preserve = (x < 244200) | (x > 246000)
    y_preserve = (y < 560000) | (y > 561000)
    riv_ds["bottom_elevation"] = riv_bot_da.where(
        x_preserve | y_preserve, riv_bot_da + 0.15
    )
    return riv_ds


def tutorial_03(path: Path | str) -> None:
    """
    Starting dataset for tutorial 3 in the iMOD Documentation.
    """
    # Unzips tutorial content to ``path``. Has a race condition when executed
    # multiple times with the same path.
    path = Path(path)
    filename = "iMOD-Documentation-tutorial_03.zip"
    lock = FileLock(REGISTRY.path / f"{filename}.lock")
    with lock:
        _ = REGISTRY.fetch(filename, processor=Unzip(extract_dir=path))

    return path / "tutorial_03" / "GWF_model_Hondsrug.prj"
