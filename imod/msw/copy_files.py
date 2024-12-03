from pathlib import Path
from shutil import copy2
from typing import cast

import numpy as np
import xarray as xr

from imod.msw.pkgbase import MetaSwapPackage
from imod.typing import Imod5DataDict


class CopyFiles(MetaSwapPackage):
    def __init__(self, paths: list[str]):
        super().__init__()
        paths_da = xr.DataArray(
            paths, coords={"file_nr": np.arange(len(paths))}, dims=("file_nr",)
        )
        self.dataset["paths"] = paths_da

    @classmethod
    def from_imod5_data(cls, imod5_data: Imod5DataDict):
        paths = cast(list[list[str]], imod5_data["extra"]["paths"])
        paths_unpacked = [Path(p[0]) for p in paths]
        files_to_filter = (
            "mete_grid.inp",
            "para_sim.inp",
            "svat2precgrid.inp",
            "svat2etrefgrid.inp",
        )
        paths_filtered = [
            str(p) for p in paths_unpacked if p.name.lower() not in files_to_filter
        ]

        return cls(paths_filtered)

    def write(self, directory: str | Path, *_):
        directory = Path(directory)

        src_paths = [Path(p) for p in self.dataset["paths"].to_numpy()]
        dst_paths = [directory / p.name for p in src_paths]

        for src_path, dst_path in zip(src_paths, dst_paths):
            copy2(src_path, dst_path)
