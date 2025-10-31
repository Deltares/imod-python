from copy import deepcopy
from pathlib import Path

import numpy as np
import xugrid as xu

from imod import util
from imod.common.interfaces.ilinedatapackage import ILineDataPackage
from imod.common.interfaces.ipackagebase import IPackageBase
from imod.common.interfaces.ipackageserializer import IPackageSerializer
from imod.typing import GridDataset
from imod.typing.grid import is_spatial_grid


class NetCDFSerializer(IPackageSerializer):
    def __init__(self, mdal_compliant: bool = False, crs: str | None = None):
        self.mdal_compliant = mdal_compliant
        self.crs = crs

    def to_file(
        self, pkg: IPackageBase, directory: Path, file_name: str, **kwargs
    ) -> Path:
        path = directory / f"{file_name}.nc"

        kwargs.update({"encoding": self._netcdf_encoding(pkg)})
        dataset = self._dataset_encoding(pkg)

        # Create encoding dict for float16 variables
        for var in dataset.data_vars:
            if dataset[var].dtype == np.float16:
                kwargs["encoding"][var] = {"dtype": "float32"}

        # Also check coordinates
        for coord in dataset.coords:
            if dataset[coord].dtype == np.float16:
                kwargs["encoding"][coord] = {"dtype": "float32"}

        if isinstance(dataset, xu.UgridDataset):
            if self.mdal_compliant:
                dataset = dataset.ugrid.to_dataset()
                mdal_dataset = util.spatial.mdal_compliant_ugrid2d(
                    dataset, crs=self.crs
                )
                mdal_dataset.to_netcdf(path, **kwargs)
            else:
                dataset.ugrid.to_netcdf(path, **kwargs)
        else:
            if is_spatial_grid(dataset):
                dataset = util.spatial.gdal_compliant_grid(dataset, crs=self.crs)
            dataset.to_netcdf(path, **kwargs)

        return path

    def _dataset_encoding(self, pkg: IPackageBase) -> GridDataset:
        if isinstance(pkg, ILineDataPackage):
            new: ILineDataPackage = deepcopy(pkg)
            new.dataset["geometry"] = new.line_data.to_json()
            return new.dataset

        return pkg.dataset

    def _netcdf_encoding(self, pkg: IPackageBase) -> dict:
        """

        The encoding used in the to_netcdf method
        Override this to provide custom encoding rules

        """
        if isinstance(pkg, ILineDataPackage):
            return {"geometry": {"dtype": "str"}}

        return {}
