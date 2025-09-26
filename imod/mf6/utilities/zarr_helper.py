import xugrid as xu


def to_zarr(dataset, path, engine, **kwargs):
    import zarr

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
