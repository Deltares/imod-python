from typing import Union

import numpy as np
import xarray as xr

from imod.mf6.boundary_condition import BoundaryCondition
from imod.mf6.write_context import WriteContext
from imod.schemata import (
    AnyValueSchema,
    CoordsSchema,
    DimsSchema,
    DTypeSchema,
    IndexesSchema,
)


class Mf6HorizontalFlowBarrier(BoundaryCondition):
    """
    Horizontal Flow Barrier (HFB) package

    Input to the Horizontal Flow Barrier (HFB) Package is read from the file
    that has type "HFB6" in the Name File. Only one HFB Package can be
    specified for a GWF model.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.2.2.pdf

    We recommend using either the ``HorizontalFlowBarrierResistance`` or
    ``HorizontalFlowBarrierMultiplier`` over this class.

    Parameters
    ----------
    cell_id1: xr.DataArray
        the cell id on 1 side of the barrier. This is either in column, row format for structured grids or in cell2d
         format for unstructured grids. The DataArray should contain a coordinate "cell_dims1" which specifies the
          cell_id structure eg:
            cell_dims1  (cell_dims1) <U8 'row_1' 'column_1' or cell_dims1  (cell_dims1) <U8 'cell2d_1' 'cell2d_1'
    cell_id2: xr.DataArray
        the cell id on the other side of the barrier. This is either in column, row format for structured grids or in
        cell2d format for unstructured grids. The DataArray should contain a coordinate "cell_dims2" which specifies the
          cell_id structure eg:
            cell_dims2  (cell_dims2) <U8 'row_2' 'column_2' or cell_dims2  (cell_dims2) <U8 'cell2d_2' 'cell2d_2'
    layer: xr.DataArray
        the layers in which the barrier is active
    hydraulic_characteristic: xr.DataArray
        hydraulic characteristic of the barrier: the inverse of the hydraulic
        resistance. Negative values are interpreted as a multiplier of the cell
        to cell conductance.

    Examples
    --------
    >> # Structured grid:
    >> row_1 = [1, 2]
    >> column_1 = [1, 1]
    >> row_2 = [1, 2]
    >> column_2 = [2, 2]
    >> layer = [1, 2, 3]
    >>
    >> cell_indices = np.arange(len(row_1)) + 1
    >>
    >> barrier = xr.Dataset()
    >> barrier["cell_id1"] = xr.DataArray([row_1, column_1], coords={"cell_idx": cell_indices, "cell_dims1": ["row_1", "column_1"]})
    >> barrier["cell_id2"] = xr.DataArray([row_2, column_2], coords={"cell_idx": cell_indices, "cell_dims2": ["row_2", "column_2"]})
    >> barrier["hydraulic_characteristic"] = xr.DataArray(np.full((len(layer), len(cell_indices)), 1e-3), coords={"layer": layer, "cell_idx": cell_indices})
    >> barrier = (barrier.stack(cell_id=("layer", "cell_idx"), create_index=False).drop_vars("cell_idx").reset_coords())
    >>
    >> Mf6HorizontalFlowBarrier(**ds)
    >>
    >>
    >> # Unstructured grid
    >> cell2d_id1 = [1, 2]
    >> cell2d_id2 = [3, 4]
    >> layer = [1, 2, 3]
    >>
    >> cell_indices = np.arange(len(cell2d_id1)) + 1
    >>
    >> barrier = xr.Dataset()
    >> barrier["cell_id1"] = xr.DataArray(np.array([cell2d_id1]).T, coords={"cell_idx": cell_indices, "cell_dims1": ["cell2d_1"]})
    >> barrier["cell_id2"] = xr.DataArray(np.array([cell2d_id2]).T, coords={"cell_idx": cell_indices, "cell_dims2": ["cell2d_2"]})
    >> barrier["hydraulic_characteristic"] = xr.DataArray(np.full((len(layer), len(cell_indices)), 1e-3), coords={"layer": layer, "cell_idx": cell_indices})>>
    >> barrier = (barrier.stack(cell_id=("layer", "cell_idx"), create_index=False).drop_vars("cell_idx").reset_coords())
    >>
    >> Mf6HorizontalFlowBarrier(**ds)
    >>

    """

    _pkg_id = "hfb"
    _init_schemata = {
        "hydraulic_characteristic": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            CoordsSchema(("layer",)),
            DimsSchema("layer", "{edge_dim}") | DimsSchema("{edge_dim}"),
        ],
        "idomain": [
            DTypeSchema(np.integer),
            DimsSchema("idomain_layer", "y", "x")
            | DimsSchema("idomain_layer", "{face_dim}"),
            IndexesSchema(),
        ],
    }
    _write_schemata = {
        "idomain": (AnyValueSchema(">", 0),),
    }
    _period_data = ("hydraulic_characteristic",)
    _keyword_map = {}
    _template = BoundaryCondition._initialize_template(_pkg_id)

    def __init__(
        self,
        cell_id1: xr.DataArray,
        cell_id2: xr.DataArray,
        layer: xr.DataArray,
        hydraulic_characteristic: xr.DataArray,
        print_input: Union[bool, xr.DataArray] = False,
        validate: Union[bool, xr.DataArray] = True,
    ):
        super().__init__(locals())
        self.dataset["cell_id1"] = cell_id1
        self.dataset["cell_id2"] = cell_id2
        self.dataset["layer"] = layer
        self.dataset["hydraulic_characteristic"] = hydraulic_characteristic
        self.dataset["print_input"] = print_input

    def _get_bin_ds(self):
        bin_ds = self.dataset[
            ["cell_id1", "cell_id2", "layer", "hydraulic_characteristic"]
        ]

        return bin_ds

    def _ds_to_arrdict(self, ds):
        arrdict = super()._ds_to_arrdict(ds)
        arrdict["cell_dims1"] = ds.coords["cell_dims1"].values
        arrdict["cell_dims2"] = ds.coords["cell_dims2"].values

        return arrdict

    def _to_struct_array(self, arrdict, layer):
        field_spec = (
            [("layer_1", np.int32)]
            + [(dim, np.int32) for dim in arrdict["cell_dims1"]]
            + [("layer_2", np.int32)]
            + [(dim, np.int32) for dim in arrdict["cell_dims2"]]
            + [("hydraulic_characteristic", np.float64)]
        )

        sparse_dtype = np.dtype(field_spec)

        recarr = np.empty(len(arrdict["hydraulic_characteristic"]), dtype=sparse_dtype)
        recarr["layer_1"] = arrdict["layer"]
        for dim, value in dict(zip(arrdict["cell_dims1"], arrdict["cell_id1"])).items():
            recarr[dim] = value
        recarr["layer_2"] = arrdict["layer"]
        for dim, value in dict(zip(arrdict["cell_dims2"], arrdict["cell_id2"])).items():
            recarr[dim] = value
        recarr["hydraulic_characteristic"] = arrdict["hydraulic_characteristic"]

        return recarr

    def write(
        self,
        pkgname: str,
        globaltimes: np.ndarray,
        write_context: WriteContext,
    ):
        # MODFLOW6 does not support binary HFB input.
        super().write(pkgname, globaltimes, write_context)
