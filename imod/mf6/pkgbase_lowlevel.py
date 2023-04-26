"""
Module to store (prototype) low-level modflow6 package base classes.

These are closer to the Modflow 6 data model, with a cellid,
instead of x, y coordinates.

We plan to split up the present attributes of the classes in pkgbase.py 
(Package and BoundaryCondition) into low-level and high-level classes.
"""

import numpy as np

from imod.mf6.pkgbase import BoundaryCondition


# FUTURE:
# Make Mf6Bc inherit Mf6Pkg, not BoundaryCondition.
# Move functions from BoundaryCondition or Package,
# which are required to work with low-level data to either Mf6Bc or Mf6Pkg
class Mf6Bc(BoundaryCondition):
    """
    Low-level package to share methods for specific modflow 6 packages with
    time component.

    It is not meant to be used directly, only to inherit from, to implement new
    packages.

    This class only supports `list input
    <https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=19>`_,
    not the array input which is used in :class:`Mf6Pkg`.
    """

    # TODO: Rename to `to_struct_array`
    # NOTE: This function should be part of Mf6Pkg, now defined here,
    #   to ensure function is overrided.
    def to_sparse(self):
        ds = self.dataset
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

    def write_datafile(self, outpath, ds, binary):
        """
        Writes a modflow6 binary data file
        """
        sparse_data = self.to_sparse()
        outpath.parent.mkdir(exist_ok=True, parents=True)
        if binary:
            self._write_binaryfile(outpath, sparse_data)
        else:
            self._write_textfile(outpath, sparse_data)

    @classmethod
    def from_file(cls, path) -> "Mf6Bc":
        """
        Instantiate class from modflow 6 file
        """
        # return cls.__new__(cls)
        raise NotImplementedError("from_file not implemented")
