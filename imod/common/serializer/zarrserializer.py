import shutil
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import xugrid as xu

from imod.common.interfaces.ilinedatapackage import ILineDataPackage
from imod.common.interfaces.ipackagebase import IPackageBase
from imod.common.interfaces.ipackageserializer import IPackageSerializer
from imod.typing import GridDataset
from imod.util.imports import MissingOptionalModule

if TYPE_CHECKING:
    import zarr
else:
    try:
        import zarr
    except ImportError:
        zarr = MissingOptionalModule("zarr")


class ZarrSerializer(IPackageSerializer):
    def __init__(self, use_zip: bool = False):
        self.use_zip = use_zip

    def to_file(
        self, pkg: IPackageBase, directory: Path, file_name: str, **kwargs
    ) -> Path:
        if self.use_zip:
            path = directory / f"{file_name}.zarr.zip"
            store = zarr.storage.ZipStore(path, mode="w")
            write_context = store
        else:
            path = directory / f"{file_name}.zarr"
            store = path
            write_context = nullcontext()

        dataset = self._dataset_encoding(pkg)

        if path.exists():
            # Check if directory (ordinary .zarr, directory) or ZipStore (zip file).
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

        with write_context:
            if isinstance(dataset, xu.UgridDataset):
                dataset.ugrid.to_zarr(store=store, **kwargs)
            else:
                dataset.to_zarr(store=store, **kwargs)

        return path

    def _dataset_encoding(self, pkg: IPackageBase) -> GridDataset:
        if isinstance(pkg, ILineDataPackage):
            new: ILineDataPackage = deepcopy(pkg)
            new.dataset["geometry"] = new.line_data.to_json()
            return new.dataset

        return pkg.dataset
