import shutil
from pathlib import Path
from typing import Literal, TypeAlias

import xugrid as xu

from imod.typing import GridDataset

EngineType: TypeAlias = Literal["netcdf4", "zarr", "zarr.zip"]


def engine_to_ext(engine: EngineType) -> str:
    """Get file extension for a given file engine.

    Parameters
    ----------
    engine : str
        The file engine. Options are 'netcdf4', 'zarr', and 'zarr.zip'.

    Returns
    -------
    str
        The corresponding file extension.

    Raises
    ------
    ValueError
        If the provided engine is not recognized.

    """
    engine_ext_map = {
        "netcdf4": "nc",
        "zarr": "zarr",
        "zarr.zip": "zarr.zip",
    }

    try:
        return engine_ext_map[engine.lower()]
    except KeyError:
        raise ValueError(
            f"Unrecognized file engine: {engine}. Should be 'netcdf4', 'zarr', and 'zarr.zip'"
        )


def to_zarr(
    dataset: GridDataset, path: str | Path, engine: EngineType, **kwargs
) -> None:
    import zarr

    path = Path(path)
    if path.exists():
        # Check if directory (ordinary .zarr, directory) or ZipStore (zip file).
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    match engine:
        case "zarr":
            if isinstance(dataset, xu.UgridDataset):
                dataset.ugrid.to_zarr(path, **kwargs)
            else:
                dataset.to_zarr(path, **kwargs)
        case "zarr.zip":
            with zarr.storage.ZipStore(path, mode="w") as store:
                if isinstance(dataset, xu.UgridDataset):
                    dataset.ugrid.to_zarr(store, **kwargs)
                else:
                    dataset.to_zarr(store, **kwargs)
        case _:
            raise ValueError(
                f'Expected engine to be "zarr" or "zarr.zip", got: {engine}'
            )


def to_file(
    dataset: GridDataset, path: str | Path, engine: EngineType, **kwargs
) -> None:
    if engine.lower() == "netcdf4":
        dataset.to_netcdf(path, **kwargs)
    else:
        to_zarr(dataset, path, engine=engine, **kwargs)
