import warnings

import numpy as np
import xugrid as xu

from imod.mf6.pkgbase_lowlevel import Mf6Pkg
from imod.mf6.validation import PKG_DIMS_SCHEMA
from imod.schemata import (
    AllValueSchema,
    DTypeSchema,
    IdentityNoDataSchema,
    IndexesSchema,
)


class NodePropertyFlow_HighLevel:
    _init_schemata = {
        "icelltype": [
            DTypeSchema(np.integer),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
        ],
        "k": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
        ],
        "rewet_layer": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
        ],
        "k22": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
        ],
        "k33": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
        ],
        "angle1": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
        ],
        "angle2": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
        ],
        "angle3": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
        ],
        "alternative_cell_averaging": [DTypeSchema(str)],
        "save_flows": [DTypeSchema(np.bool_)],
        "starting_head_as_confined_thickness": [DTypeSchema(np.bool_)],
        "variable_vertical_conductance": [DTypeSchema(np.bool_)],
        "dewatered": [DTypeSchema(np.bool_)],
        "perched": [DTypeSchema(np.bool_)],
        "save_specific_discharge": [DTypeSchema(np.bool_)],
    }

    _default_regrid_methods = {
        "k": ("harmonic", xu.OverlapRegridder),
        "rewet_layer": ("harmonic", xu.OverlapRegridder),
        "k22": ("harmonic", xu.OverlapRegridder),
        "k33": ("harmonic", xu.OverlapRegridder),
        "angle1": ("harmonic", xu.OverlapRegridder),
        "angle2": ("harmonic", xu.OverlapRegridder),
        "angle3": ("harmonic", xu.OverlapRegridder),
    }

    _write_schemata = {
        "k": (
            AllValueSchema(">", 0.0),
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ),
        "rewet_layer": (
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ),
        "k22": (
            AllValueSchema(">", 0.0),
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
            # No need to check coords: dataset ensures they align with idomain.
        ),
        "k33": (
            AllValueSchema(">", 0.0),
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
            # No need to check coords: dataset ensures they align with idomain.
        ),
        "angle1": (IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),),
        "angle2": (IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),),
        "angle3": (IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),),
    }

    def __init__(
        self,
        icelltype,
        k,
        rewet=False,
        rewet_layer=None,
        rewet_factor=None,
        rewet_iterations=None,
        rewet_method=None,
        k22=None,
        k33=None,
        angle1=None,
        angle2=None,
        angle3=None,
        cell_averaging=None,
        alternative_cell_averaging=None,
        save_flows=False,
        starting_head_as_confined_thickness=False,
        variable_vertical_conductance=False,
        dewatered=False,
        perched=False,
        save_specific_discharge=False,
        validate: bool = True,
    ):
        super().__init__(locals())
        # check rewetting
        if not rewet and any(
            [rewet_layer, rewet_factor, rewet_iterations, rewet_method]
        ):
            raise ValueError(
                "rewet_layer, rewet_factor, rewet_iterations, and rewet_method should"
                " all be left at a default value of None if rewet is False."
            )
        self.dataset["icelltype"] = icelltype
        self.dataset["k"] = k
        self.dataset["rewet"] = rewet
        self.dataset["rewet_layer"] = rewet_layer
        self.dataset["rewet_factor"] = rewet_factor
        self.dataset["rewet_iterations"] = rewet_iterations
        self.dataset["rewet_method"] = rewet_method
        self.dataset["k22"] = k22
        self.dataset["k33"] = k33
        self.dataset["angle1"] = angle1
        self.dataset["angle2"] = angle2
        self.dataset["angle3"] = angle3
        if cell_averaging is not None:
            warnings.warn(
                "Use of `cell_averaging` is deprecated, please use `alternative_cell_averaging` instead",
                DeprecationWarning,
            )
            self.dataset["alternative_cell_averaging"] = cell_averaging
        else:
            self.dataset["alternative_cell_averaging"] = alternative_cell_averaging

        self.dataset["save_flows"] = save_flows
        self.dataset[
            "starting_head_as_confined_thickness"
        ] = starting_head_as_confined_thickness
        self.dataset["variable_vertical_conductance"] = variable_vertical_conductance
        self.dataset["dewatered"] = dewatered
        self.dataset["perched"] = perched
        self.dataset["save_specific_discharge"] = save_specific_discharge
        self._validate_init_schemata(validate)

    def regrid_like(self, tgt_grid, regridder):
        d = {}
        for arrayname, regridinfo in self._default_regrid_methods.items():
            method, regridclass = regridinfo
            array = self.dataset[arrayname]

            kwargs = {"source": array, "target": tgt_grid, "method": method}
            regridder = regridclass(**kwargs)

            if isinstance(array, xu.UgridDataArray):
                source_dims = (array.ugrid.grid.face_dim,)
            else:
                source_dims = ("y", "x")

            d[arrayname] = regridder.regrid_dataarray(array, source_dims)

        return type(self)(**d)


    def to_mf6_pkg(self):


        # lopp over arrays
        #  regrid arrays to tgt_grid


class Mf6Npf(Mf6Pkg):
    _pkg_id = "npf"

    _grid_data = {
        "icelltype": np.int32,
        "k": np.float64,
        "rewet_layer": np.float64,
        "k22": np.float64,
        "k33": np.float64,
        "angle1": np.float64,
        "angle2": np.float64,
        "angle3": np.float64,
    }
    _keyword_map = {
        "rewet": "rewet_record",
        "rewet_factor": "wetfct",
        "rewet_method": "ihdwet",
        "rewet_layer": "wetdry",
        "variable_vertical_conductance": "variablecv",
        "starting_head_as_confined_thickness": "thickstrt",
        "rewet_iterations": "iwetit",
    }
    _template = Package._initialize_template(_pkg_id)

    def __init__(self, dataset):    
        self.dataset=dataset
