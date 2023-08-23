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

import numpy as np

from imod.mf6.boundary_condition import BoundaryCondition
from imod.schemata import DTypeSchema


class Mf6Wel(BoundaryCondition):
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

    def _ds_to_arrdict(self, ds):
        """
        Prepares a dictionary with values needed for the _to_sparse method.
        """
        arrdict = {}

        arrdict["data_vars"] = [
            var_name for var_name in ds.data_vars if var_name != "cellid"
        ]

        dsvar = {}
        for var in arrdict["data_vars"]:
            dsvar[var] = ds[var]
        arrdict["var_values"] = dsvar

        arrdict["cellid_names"] = ds.coords["nmax_cellid"].values
        arrdict["nrow"] = ds.coords["ncellid"].size
        arrdict["cellid"] = ds["cellid"]

        return arrdict

    def _to_sparse(self, arrdict, _):
        index_spec = [(index, np.int32) for index in arrdict["cellid_names"]]
        field_spec = [(var, np.float64) for var in arrdict["data_vars"]]
        sparse_dtype = np.dtype(index_spec + field_spec)

        # Initialize the structured array
        recarr = np.empty(arrdict["nrow"], dtype=sparse_dtype)
        for cellid_name in arrdict["cellid_names"]:
            recarr[cellid_name] = arrdict["cellid"].sel(nmax_cellid=cellid_name).values

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
