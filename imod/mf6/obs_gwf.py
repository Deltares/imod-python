from pathlib import Path
from typing import Any
import os

from imod.mf6.package import Package

import numpy as np
import pandas as pd
import xarray as xr
import xugrid as xu
from imod.mf6.write_context import WriteContext


class GroundwaterFlowObservations(Package):
    """
    Observation Utility for GroundwaterFlowModel.

    Supports "head", "drawdown", and "flow-ja-face".

    """
    _pkg_id = "obs"
    _template = Package._initialize_template(_pkg_id)
    _init_schemata = {}
    _write_schemata = {}

    def __init__(
        self,
        obs_name: xr.DataArray,
        obs_type: xr.DataArray,
        obs_id: xr.DataArray,
        obs_id2: xr.DataArray|None=None,
        obs_file=None,
        digits=None,
        print_input=False,
        binary=True,
    ):
        if obs_id2 is None:
            obs_id2 = xr.full_like(obs_id, np.nan, dtype=float)

        dict_dataset: dict[str, Any] = {
            "obs_name": obs_name,
            "obs_type": obs_type,
            "obs_id": obs_id,
            "obs_id2": obs_id2,
            "obs_file": obs_file,
            "digits": digits,
            "print_input": print_input,
            "binary": binary
        }
        super().__init__(dict_dataset)

    def _get_output_filepath(self, directory: Path, pkgname: str):
        binary = self.dataset["binary"].values[()]
        if binary:
            ext = "bsv"
        else:
            ext = "csv"

        filepath = self.dataset["obs_file"].values[()]
        if filepath is None:
            filepath = directory / f"{pkgname}.{ext}"
        else:
            if not isinstance(filepath, str | Path):
                raise ValueError(
                    f"{filepath} should be of type str or Path. However it is of type {type(filepath)}"
                )
            filepath = Path(filepath)

        if filepath.is_absolute():
            path = filepath
        else:
            # Get path relative to the simulation name file.
            sim_directory = directory.parent
            path = Path(os.path.relpath(filepath, sim_directory))
        return path

    def _render(self, directory, pkgname, globaltimes, binary):
        d: dict[str, Any] = {}
        for varname in ("print_input", "digits", "binary"):
            value = self.dataset[varname].values[()]
            if self._valid(value):
                d[varname] = value
        d["obs_file"] = self._get_output_filepath(directory, pkgname)
        return self._template.render(d)

    def _obs_rows_dataframe(self) -> pd.DataFrame:
        columns = [
            self.dataset[["obs_name", "obs_type"]].to_dataframe(),
            self.dataset["obs_id"].to_pandas(),
            # astype("Int64") maps NaN to pd.NA, and to_csv will write pd.NA as an empty field
            self.dataset["obs_id2"].to_pandas().astype("Int64")
        ]
        df = pd.concat(columns, axis=1)
        # Indent the rows by two spaces.
        # We cannot rely on spaces, as they will be quoted.
        df.insert(0, "__indent0", "")
        df.insert(0, "__indent1", "")
        return df

    def _append_obs_rows(self, filename: Path | str) -> None:
        with open(filename, "a") as f:
            self._obs_rows_dataframe().to_csv(
                f, header=False, index=False, sep=" ", lineterminator="\n"
            )
            f.write("end continuous\n\n")
        return

    def _write_blockfile(self, pkgname, globaltimes, write_context: WriteContext):
        super()._write_blockfile(pkgname, globaltimes, write_context)
        filename = write_context.write_directory / f"{pkgname}.{self._pkg_id}"
        self._append_obs_rows(filename)
        return

    @classmethod
    def from_boolean_grid(cls, mask, obs_type: str, obs_file: str | Path = None, binary=True):
        # TODO: ugly method name, mask as name also dubious
        if isinstance(mask, xr.DataArray):
            if not mask.dims == ("layer", "y", "x"):
                raise ValueError()
        elif isinstance(mask, xu.UgridDataArray):
            # TODO:
            if not mask.dims == ("layer", "mesh2d_nFace"):
                raise ValueError()
        else:
            raise TypeError(f"Expected DataArray or UgridDataArray, received: {type(mask)}")

        if not np.issubdtype(mask.dtype, np.bool_):
            raise TypeError()

        # NOTE: modflow indices are 1-based.
        obs_id = xr.DataArray(
            data=np.column_stack(np.nonzero(mask.to_numpy())) + 1,
            dims=("observation", "dimension"),
        )
        n_obs, _ = obs_id.shape
        return cls(
            obs_name=xr.DataArray(data=np.full(n_obs, "0"), dims=("observation",)),
            obs_type=xr.DataArray(data=np.full(n_obs, obs_type), dims=("observation",)),
            obs_id=obs_id,
            obs_id2=None,
            obs_file=obs_file,
            digits=None,
            print_input=False,
            binary=binary,
        )
