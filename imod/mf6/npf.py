import warnings

import numpy as np

from imod.mf6.pkgbase import Package
from imod.mf6.regridding_utils import RegridderType
from imod.mf6.validation import PKG_DIMS_SCHEMA
from imod.schemata import (
    AllValueSchema,
    DTypeSchema,
    IdentityNoDataSchema,
    IndexesSchema,
)


class NodePropertyFlow(Package):
    """
    Node Property Flow package.

    In this package the hydraulic conductivity and rewetting in the model is
    specified. A single NPF Package is required for each GWF model.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=51

    Parameters
    ----------
    icelltype: array of int (xr.DataArray)
        flag for each cell that specifies how saturated thickness is treated. 0
        means saturated thickness is held constant; >0 means saturated thickness
        varies with computed head when head is below the cell top; <0 means
        saturated thickness varies with computed head unless the
        starting_head_as_confined_thickness option is in effect. When
        starting_head_as_confined_thickness is in effect, a negative value of
        icelltype indicates that saturated thickness will be computed as
        strt-bot and held constant.
    k: array of floats (xr.DataArray)
        is the hydraulic conductivity. For the common case in which the user
        would like to specify the horizontal hydraulic conductivity and the
        vertical hydraulic conductivity, then K should be assigned as the
        horizontal hydraulic conductivity, K33 should be assigned as the
        vertical hydraulic conductivity, and K22 and the three rotation
        angles should not be specified. When more sophisticated anisotropy is
        required, then K corresponds to the K11 hydraulic conductivity axis. All
        included cells (idomain > 0) must have a K value greater than zero
    rewet: ({True, False}, optional)
        activates model rewetting.
        Default is False.
    rewet_layer: float
        is a combination of the wetting threshold and a flag to indicate which
        neighboring cells can cause a cell to become wet. If rewet_layer < 0,
        only a cell below a dry cell can cause the cell to become wet. If
        rewet_layer > 0, the cell below a dry cell and horizontally adjacent
        cells can cause a cell to become wet. If rewet_layer is 0, the cell
        cannot be wetted. The absolute value of rewet_layer is the wetting
        threshold. When the sum of BOT and the absolute value of rewet_layer at
        a dry cell is equaled or exceeded by the head at an adjacent cell, the
        cell is wetted. rewet_layer must be specified if "rewet" is specified in
        the OPTIONS block. If "rewet" is not specified in the options block,
        then rewet_layer can be entered, and memory will be allocated for it,
        even though it is not used. (WETDRY)
        Default is None.
    rewet_factor:
        is a keyword and factor that is included in the calculation of the head
        that is initially established at a cell when that cell is converted from
        dry to wet. (WETFCT)
        Default is None.
    rewet_iterations:
        is a keyword and iteration interval for attempting to wet cells. Wetting
        is attempted every rewet_iterations iteration. This applies to outer
        iterations and not inner iterations. If rewet_iterations is specified as
        zero or less, then the value is changed to 1. (IWETIT)
        Default is None.
    rewet_method:
        is a keyword and integer flag that determines which equation is used to
        define the initial head at cells that become wet. If rewet_method is 0,
        h = BOT + rewet_factor (hm - BOT). If rewet_method is not 0, h = BOT +
        rewet_factor (THRESH). (IHDWET)
        Default is None.
    k22: array of floats (xr.DataArray)
        is the hydraulic conductivity of the second ellipsoid axis; for an
        unrotated case this is the hydraulic conductivity in the y direction. If
        K22 is not included, then K22 is set equal to K.
        For a regular MODFLOW grid (DIS Package is used) in which no rotation
        angles are specified, K22 is the hydraulic conductivity along columns in
        the y direction. For an unstructured DISU grid, the user must assign
        principal x and y axes and provide the angle for each cell face relative
        to the assigned x direction. All included cells (idomain > 0) must have
        a K22 value greater than zero.
        Default is None.
    k33: array of floats (xr.DataArray)
        is the hydraulic conductivity of the third ellipsoid axis; for an
        unrotated case, this is the vertical hydraulic conductivity. When
        anisotropy is applied, K33 corresponds to the K33 tensor component. All
        included cells (idomain > 0) must have a K33 value greater than zero.
        Default is None.
    angle1: float
        is a rotation angle of the hydraulic conductivity tensor in degrees. The
        angle represents the first of three sequential rotations of the
        hydraulic conductivity ellipsoid. With the K11, K22, and K33 axes of the
        ellipsoid initially aligned with the x, y, and z coordinate axes,
        respectively, angle1 rotates the ellipsoid about its K33 axis (within
        the x - y plane). A positive value represents counter-clockwise rotation
        when viewed from any point on the positive K33 axis, looking toward the
        center of the ellipsoid. A value of zero indicates that the K11 axis
        lies within the x - z plane. If angle1 is not specified, default values
        of zero are assigned to angle1, angle2, and angle3, in which case the
        K11, K22, and K33 axes are aligned with the x, y, and z axes,
        respectively.
        Default is None.
    angle2: float
        is a rotation angle of the hydraulic conductivity tensor in degrees. The
        angle represents the second of three sequential rotations of the
        hydraulic conductivity ellipsoid. Following the rotation by angle1
        described above, angle2 rotates the ellipsoid about its K22 axis (out of
        the x - y plane). An array can be specified for angle2 only if angle1 is
        also specified. A positive value of angle2 represents clockwise rotation
        when viewed from any point on the positive K22 axis, looking toward the
        center of the ellipsoid. A value of zero indicates that the K11 axis
        lies within the x - y plane. If angle2 is not specified, default values
        of zero are assigned to angle2 and angle3; connections that are not
        user-designated as vertical are assumed to be strictly horizontal (that
        is, to have no z component to their orientation); and connection lengths
        are based on horizontal distances.
        Default is None.
    angle3: float
        is a rotation angle of the hydraulic conductivity tensor in degrees. The
        angle represents the third of three sequential rotations of the
        hydraulic conductivity ellipsoid. Following the rotations by angle1 and
        angle2 described above, angle3 rotates the ellipsoid about its K11 axis.
        An array can be specified for angle3 only if angle1 and angle2 are also
        specified. An array must be specified for angle3 if angle2 is specified.
        A positive value of angle3 represents clockwise rotation when viewed
        from any point on the positive K11 axis, looking toward the center of
        the ellipsoid. A value of zero indicates that the K22 axis lies within
        the x - y plane.
        Default is None.
    alternative_cell_averaging : str
        Method calculating horizontal cell connection conductance.
        Options: {"LOGARITHMIC", "AMT-LMK", or "AMT-HMK"}
        Default: uses harmonic mean for averaging
    save_flows: ({True, False}, optional)
        keyword to indicate that cell-by-cell flow terms will be written to the
        file specified with "budget save file" in Output Control.
        Default is False.
    starting_head_as_confined_thickness: ({True, False}, optional)
        indicates that cells having a negative icelltype are confined, and their
        cell thickness for conductance calculations will be computed as strt-bot
        rather than top-bot.
        (THICKSTRT)
        Default is False.
    variable_vertical_conductance: ({True, False}, optional)
        keyword to indicate that the vertical conductance will be calculated
        using the saturated thickness and properties of the overlying cell and
        the thickness and properties of the underlying cell. if the dewatered
        keyword is also specified, then the vertical conductance is calculated
        using only the saturated thickness and properties of the overlying cell
        if the head in the underlying cell is below its top. if these keywords
        are not specified, then the default condition is to calculate the
        vertical conductance at the start of the simulation using the initial
        head and the cell properties. the vertical conductance remains constant
        for the entire simulation.
        (VARIABLECV)
        Default is False.
    dewatered: ({True, False}, optional)
        If the dewatered keyword is specified, then the vertical conductance is
        calculated using only the saturated thickness and properties of the
        overlying cell if the head in the underlying cell is below its top.
        Default is False.
    perched: ({True, False}, optional)
        keyword to indicate that when a cell is overlying a dewatered
        convertible cell, the head difference used in Darcy’s Law is equal to
        the head in the overlying cell minus the bottom elevation of the
        overlying cell. If not specified, then the default is to use the head
        difference between the two cells.
        Default is False.
    save_specific_discharge: ({True, False}, optional)
        keyword to indicate that x, y, and z components of specific discharge
        will be calculated at cell centers and written to the cell-by-cell flow
        file, which is specified with"budget save file" in Output Control. If
        this option is activated, then additional information may be required in
        the discretization packages and the GWF Exchange package (if GWF models
        are coupled). Specifically, angldegx must be specified in the
        connectiondata block of the disu package; angldegx must also be
        specified for the GWF Exchange as an auxiliary variable. disu package
        has not been implemented yet.
        Default is False.
    validate: {True, False}
        Flag to indicate whether the package should be validated upon
        initialization. This raises a ValidationError if package input is
        provided in the wrong manner. Defaults to True.
    """

    _pkg_id = "npf"

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

    _write_schemata = {
        "k": (AllValueSchema(">", 0.0),),
        "rewet_layer": (
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ),
        "k22": (
            AllValueSchema(">", 0.0),
            # No need to check coords: dataset ensures they align with idomain.
        ),
        "k33": (
            AllValueSchema(">", 0.0),
            # No need to check coords: dataset ensures they align with idomain.
        ),
    }

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

    _regrid_method = {
        "icelltype": (RegridderType.OVERLAP, "mean"),
        "k": (RegridderType.OVERLAP, "geometric_mean"),  # horizontal if angle2 = 0
        "k22": (
            RegridderType.OVERLAP,
            "geometric_mean",
        ),  # horizontal if angle2 = 0 & angle3 = 0
        "k33": (
            RegridderType.OVERLAP,
            "harmonic_mean",
        ),  # vertical if angle2 = 0 & angle3 = 0
        "angle1": (RegridderType.OVERLAP, "mean"),
        "angle2": (RegridderType.OVERLAP, "mean"),
        "angle3": (RegridderType.OVERLAP, "mean"),
        "rewet_layer": (RegridderType.BARYCENTRIC, None),
    }
    _skip_mask_arrays = ["k", "k22", "k33", "angle1", "angle2", "angle3"]

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

    @classmethod
    def get_pkg_id(cls) -> str:
        """
        Returns the preferred package id for this class.
        """
        return cls._pkg_id
