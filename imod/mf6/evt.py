import numpy as np

from imod.mf6.pkgbase import BoundaryCondition, VariableMetaData
from imod.schemata import (
    AllInsideNoDataSchema,
    AllNoDataSchema,
    DimsSchema,
    DTypeSchema,
    IdentityNoDataSchema,
    IndexesSchema,
    OtherCoordsSchema,
)


class Evapotranspiration(BoundaryCondition):
    """
    Evapotranspiration (EVT) Package.
    Any number of EVT Packages can be specified for a single groundwater flow
    model. All single-valued variables are free format.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=86

    Parameters
    ----------
    surface: array of floats (xr.DataArray)
        is the elevation of the ET surface (L). A time-series name may be
        specified.
    rate: array of floats (xr.DataArray)
        is the maximum ET flux rate (LT −1). A time-series name may be
        specified.
    depth: array of floats (xr.DataArray)
        is the ET extinction depth (L). A time-series name may be specified.
    proportion_rate: array of floats (xr.DataArray)
        is the proportion of the maximum ET flux rate at the bottom of a segment
        (dimensionless). A time-series name may be specified. (petm)
    proportion_depth: array of floats (xr.DataArray)
        is the proportion of the ET extinction depth at the bottom of a segment
        (dimensionless). A timeseries name may be specified. (pxdp)
    fixed_cell: array of floats (xr.DataArray)
        indicates that evapotranspiration will not be reassigned to a cell
        underlying the cell specified in the list if the specified cell is
        inactive.
    print_input: ({True, False}, optional)
        keyword to indicate that the list of evapotranspiration information will
        be written to the listing file immediately after it is read.
        Default is False.
    print_flows: ({True, False}, optional)
        Indicates that the list of evapotranspiration flow rates will be printed
        to the listing file for every stress period time step in which "BUDGET
        PRINT" is specified in Output Control. If there is no Output Control
        option and PRINT FLOWS is specified, then flow rates are printed for the
        last time step of each stress period.
        Default is False.
    save_flows: ({True, False}, optional)
        Indicates that evapotranspiration flow terms will be written to the file
        specified with "BUDGET FILEOUT" in Output Control.
        Default is False.
    observations: [Not yet supported.]
        Default is None.
    """

    _pkg_id = "evt"
    _init_schemata = {
        "surface": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            DimsSchema("time", "y", "x") | DimsSchema("y", "x"),
        ],
        "rate": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            DimsSchema("time", "y", "x") | DimsSchema("y", "x"),
        ],
        "depth": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            DimsSchema("time", "y", "x") | DimsSchema("y", "x"),
        ],
        "proportion_rate": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            DimsSchema("time", "y", "x") | DimsSchema("y", "x"),
        ],
        "proportion_depth": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            DimsSchema("time", "y", "x") | DimsSchema("y", "x"),
        ],
        "print_flows": [DTypeSchema(np.bool_), DimsSchema()],
        "save_flows": [DTypeSchema(np.bool_), DimsSchema()],
    }
    _write_schemata = {
        "surface": [
            OtherCoordsSchema("idomain"),
            AllNoDataSchema(),  # Check for all nan, can occur while clipping
            AllInsideNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ],
        "rate": [IdentityNoDataSchema("surface")],
        "depth": [IdentityNoDataSchema("surface")],
        "proportion_rate": [IdentityNoDataSchema("surface")],
        "proportion_depth": [IdentityNoDataSchema("surface")],
    }

    _metadata_dict = {
        "surface": VariableMetaData(np.floating),
        "rate": VariableMetaData(np.floating),
        "depth": VariableMetaData(np.floating),
        "proportion_depth": VariableMetaData(np.floating),
        "proportion_rate": VariableMetaData(np.floating),
    }
    _period_data = ("surface", "rate", "depth", "proportion_depth", "proportion_rate")
    _keyword_map = {}
    _template = BoundaryCondition._initialize_template(_pkg_id)

    def __init__(
        self,
        surface,
        rate,
        depth,
        proportion_rate,
        proportion_depth,
        fixed_cell=False,
        print_input=False,
        print_flows=False,
        save_flows=False,
        observations=None,
    ):
        super().__init__(locals())
        self.dataset["surface"] = surface
        self.dataset["rate"] = rate
        self.dataset["depth"] = depth
        if ("segment" in proportion_rate.dims) ^ ("segment" in proportion_depth.dims):
            raise ValueError(
                "Segment must be provided for both proportion_rate and"
                " proportion_depth, or for none at all."
            )
        self.dataset["proportion_rate"] = proportion_rate
        self.dataset["proportion_depth"] = proportion_depth
        self.dataset["fixed_cell"] = fixed_cell
        self.dataset["print_input"] = print_input
        self.dataset["print_flows"] = print_flows
        self.dataset["save_flows"] = save_flows
        self.dataset["observations"] = observations

        self._validate_at_init()
        self._pkgcheck_at_init()

        # TODO: add write logic for transforming proportion rate and depth to
        # the right shape in the binary file.

    def _validate(self, schemata, **kwargs):
        # Insert additional kwargs
        kwargs["surface"] = self["surface"]
        errors = super()._validate(schemata, **kwargs)

        return errors
