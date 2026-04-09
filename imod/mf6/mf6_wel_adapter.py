"""
Module to store (prototype) Modflow 6 adapter classes.

These are closer to the Modflow 6 data model, with a cellid,
instead of x, y coordinates.

We plan to split up the present attributes of the classes in Package.py and BoundaryCondition.py into low-level and
high-level classes.

The high-level class contains grids with x, y, z coordinates, closely linked to
GIS systems. The low-level classes contain a dataset based on cellid,
consisting of layer, row, and column, closely resembling input for Modflow6.
"""

from typing import Any, Dict, Optional

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from imod.common.interfaces.ipackage import IPackage
from imod.mf6.boundary_condition import BoundaryCondition
from imod.schemata import DTypeSchema

# FUTURE: There was an idea to autogenerate modflow 6 adapters.
# This was relevant:
# https://github.com/Deltares/xugrid/blob/main/xugrid/core/wrap.py#L90


def cellid_from_arrays__structured(
    layer: NDArray[np.int_], row: NDArray[np.int_], column: NDArray[np.int_]
):
    """
    Create DataArray of cellids for structured indices.

    Parameters
    ----------
    layer: numpy.array
        1D array of integers for layer
    row: numpy.array
        1D array of integers for row
    column: numpy.array
        1D array of integers for column

    Returns
    -------
    cellid: xr.DataArray
        2D DataArray with ``ncellid`` rows and 3 columns.
    """
    return _cellid_from_kwargs(layer=layer, row=row, column=column)


def cellid_from_arrays__unstructured(layer: NDArray[np.int_], cell2d: NDArray[np.int_]):
    """
    Create DataArray of cellids for unstructured indices.

    Parameters
    ----------
    layer: numpy.array
        1D array of integers for layer
    cell2d: numpy.array
        1D array of integers for cell2d indices

    Returns
    -------
    cellid: xr.DataArray
        2D DataArray with ``ncellid`` rows and 2 columns.
    """
    return _cellid_from_kwargs(layer=layer, cell2d=cell2d)


def _cellid_from_kwargs(**kwargs):
    dim_cellid_coord = list(kwargs.keys())
    arr_ls = list(kwargs.values())
    index = np.arange(len(arr_ls[0]))
    indices_ls = [
        xr.DataArray(ind_arr, coords={"index": index}, dims=("index",))
        for ind_arr in arr_ls
    ]
    return concat_indices_to_cellid(indices_ls, dim_cellid_coord)


def concat_indices_to_cellid(
    indices_ls: list[xr.DataArray], dim_cellid_coord: list[str]
) -> xr.DataArray:
    """
    Create DataArray of cellids from list of one-dimensional DataArrays with
    indices.

    Parameters
    ----------
    indices_ls: list of xr.DataArrays
        List of one-dimensional DataArrays with indices. For structured grids,
        these are the layer, rows, columns. For unstructured grids, these are
        layer, cell2d.
    dim_cellid_coord: list of strings
        List of coordinate names, which are assigned to the ``"dim_cellid"``
        dimension.

    Returns
    -------
    cellid: xr.DataArray
        2D DataArray with ``ncellid`` rows and 3 to 2 columns, depending
        on whether on a structured or unstructured grid.
    """
    cellid = xr.concat(indices_ls, dim="dim_cellid")
    # Rename generic dimension name "index" to ncellid.
    cellid = cellid.rename(index="ncellid")
    # Put dimensions in right order after concatenation.
    cellid = cellid.transpose("ncellid", "dim_cellid")
    # Assign extra coordinate names.
    coords = {
        "dim_cellid": dim_cellid_coord,
    }
    return cellid.assign_coords(coords=coords)


class Mf6Wel(BoundaryCondition, IPackage):
    """
    Package resembling input for Modflow 6 List Input. This class has
    methods for the modflow 6 wel packages with time component.

    This class only supports `list input
    <https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=19>`_,
    not the array input which is used in :class:`Mf6Package`.
    """

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
        save_flows: Optional[bool] = None,
        print_flows: Optional[bool] = None,
        print_input: Optional[bool] = None,
        validate: bool = True,
    ):
        dict_dataset = {
            "cellid": cellid,
            "rate": rate,
            "concentration": concentration,
            "concentration_boundary_type": concentration_boundary_type,
            "save_flows": save_flows,
            "print_flows": print_flows,
            "print_input": print_input,
        }
        super().__init__(dict_dataset)
        self._validate_init_schemata(validate)

    def _ds_to_arrdict(self, ds):
        """
        Prepares a dictionary with values needed for the _to_sparse method.
        """
        arrdict: Dict[str, Any] = {}

        arrdict["data_vars"] = [
            var_name for var_name in ds.data_vars if var_name != "cellid"
        ]

        dsvar = {}
        for var in arrdict["data_vars"]:
            dsvar[var] = ds[var]
        arrdict["var_values"] = dsvar

        arrdict["cellid_names"] = ds.coords["dim_cellid"].values
        arrdict["nrow"] = ds.coords["ncellid"].size
        arrdict["cellid"] = ds["cellid"]

        return arrdict

    def _to_struct_array(self, arrdict, _):
        index_spec = [(index, np.int32) for index in arrdict["cellid_names"]]
        field_spec = [(var, np.float64) for var in arrdict["data_vars"]]
        sparse_dtype = np.dtype(index_spec + field_spec)

        # Initialize the structured array
        recarr = np.empty(arrdict["nrow"], dtype=sparse_dtype)
        for cellid_name in arrdict["cellid_names"]:
            recarr[cellid_name] = arrdict["cellid"].sel(dim_cellid=cellid_name).values

        for var in arrdict["data_vars"]:
            recarr[var] = arrdict["var_values"][var]

        return recarr

    @classmethod
    def from_file(cls, path, **kwargs) -> "Mf6Wel":
        """
        Instantiate class from modflow 6 file
        """
        # return cls.__new__(cls)
        raise NotImplementedError("from_file not implemented")
