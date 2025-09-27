import shutil
from pathlib import Path

import xugrid as xu


def to_zarr(dataset, path: str | Path, engine: str, **kwargs):
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

    return
