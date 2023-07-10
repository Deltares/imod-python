"""
Module to store (prototype) Modflow 6 adapter classes.

These are closer to the Modflow 6 data model, with a cellid,
instead of x, y coordinates.

We plan to split up the present attributes of the classes in pkgbase.py
(Package and BoundaryCondition) into low-level and high-level classes.

The high-level class contains grids with x, y, z coordinates, closely linked to
GIS systems. The low-level classes contain a dataset based on cellid,
consisting of layer, row, and column, closely resembling input for Modflow6.
"""

from pathlib import Path

import numpy as np
import xarray as xr

from imod.mf6.boundary_condition import BoundaryCondition
from imod.schemata import DTypeSchema


class Mf6Wel(BoundaryCondition):
    _pkg_id = "wel"

    _period_data = ("cellid", "rate")
    _keyword_map = {}
    _template = BoundaryCondition._initialize_template(_pkg_id)
    _auxiliary_data = {"concentration": "species"}

    _init_schemata = {
        "cellid": [DTypeSchema(np.integer)],
        "rate": [DTypeSchema(np.floating)],
        "concentration": [DTypeSchema(np.floating)],
    }
    _write_schemata = {}

    def __init__(
        self,
        cellid,
        rate,
        concentration=None,
        concentration_boundary_type="aux",
        validate: bool = True,
    ):
        super().__init__()
        self.dataset["cellid"] = cellid
        self.dataset["rate"] = rate

        if concentration is not None:
            self.dataset["concentration"] = concentration
            self.dataset["concentration_boundary_type"] = concentration_boundary_type
            self.add_periodic_auxiliary_variable()
        self._validate_init_schemata(validate)

    # TODO: Method can be moved outside class.
    # TODO: Rename to `to_struct_array`
    # NOTE: This function should be part of Mf6Pkg, now defined here,
    #   to ensure function is overrided.
    def _to_sparse(self, ds: xr.Dataset):
        data_vars = [var for var in ds.data_vars if var != "cellid"]
        cellid_names = ds.coords["nmax_cellid"].values
        nrow = ds.coords["ncellid"].size

        index_spec = [(index, np.int32) for index in cellid_names]
        field_spec = [(var, np.float64) for var in data_vars]
        sparse_dtype = np.dtype(index_spec + field_spec)

        # Initialize the structured array
        recarr = np.empty(nrow, dtype=sparse_dtype)
        for cellid_name in cellid_names:
            recarr[cellid_name] = ds["cellid"].sel(nmax_cellid=cellid_name).values
        for var in data_vars:
            recarr[var] = ds[var]
        return recarr

    def write_datafile(self, outpath: Path, ds: xr.Dataset, binary: bool):
        """
        Writes a modflow6 binary data file
        """
        sparse_data = self._to_sparse(ds)
        outpath.parent.mkdir(exist_ok=True, parents=True)
        if binary:
            self._write_binaryfile(outpath, sparse_data)
        else:
            self._write_textfile(outpath, sparse_data)

    @classmethod
    def from_file(cls, path, **kwargs) -> "Mf6Wel":
        """
        Instantiate class from modflow 6 file
        """
        # return cls.__new__(cls)
        raise NotImplementedError("from_file not implemented")
