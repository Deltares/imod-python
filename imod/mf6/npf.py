from copy import deepcopy
from typing import Optional, cast

import numpy as np
import xarray as xr

from imod.common.interfaces.iregridpackage import IRegridPackage
from imod.common.utilities.regrid import (
    _regrid_package_data,
)
from imod.logging import init_log_decorator
from imod.mf6.package import Package
from imod.mf6.regrid.regrid_schemes import (
    NodePropertyFlowRegridMethod,
)
from imod.mf6.utilities.imod5_converter import fill_missing_layers
from imod.mf6.validation import PKG_DIMS_SCHEMA
from imod.schemata import (
    AllCoordsValueSchema,
    AllValueSchema,
    CompatibleSettingsSchema,
    DimsSchema,
    DTypeSchema,
    IdentityNoDataSchema,
    IndexesSchema,
)
from imod.typing import GridDataArray
from imod.typing.grid import zeros_like
from imod.util.regrid import RegridderWeightsCache


def _dataarray_to_bool(griddataarray: GridDataArray) -> bool:
    if griddataarray is None or griddataarray.values is None:
        return False

    if griddataarray.values.size != 1:
        raise ValueError("DataArray is not a single value")

    if griddataarray.values.dtype != bool:
        raise ValueError("DataArray is not a boolean")

    bool_value = cast(bool, griddataarray.values.item())
    return bool_value


class NodePropertyFlow(Package, IRegridPackage):
    """
    Node Property Flow package.

    In this package the hydraulic conductivity and rewetting in the model is
    specified. A single NPF Package is required for each GWF model.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=51

    A note about regridding: the fields k, k22, k33 define the principal
    components of an anisotropic conductivity tensor. By default, k and k22 are
    in the horizontal plane and k33 is vertical. Angle1, angle2 and angle3
    define the rotation of this tensor. The regridding methods associated by
    default are chosen based on the assumption that k and k22 are horizontal and
    k33 is vertical. If this is not the case, it is up to the user to regrid the
    npf package using other regridding methods. This may be recommended if for
    example the rotation is such that k has become vertical and k33 horizontal.

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
    save_saturation: ({True, False}, optional)
        keyword to indicate that cell saturation will be written to the budget
        file, which is specified with "BUDGET SAVE FILE" in Output Control.
        Saturation will be saved to the budget file as an auxiliary variable
        saved with the DATA-SAT text label. Saturation is a cell variable that
        ranges from zero to one and can be used by post processing programs to
        determine how much of a cell volume is saturated. If ICELLTYPE is 0,
        then saturation is always one.
    xt3d_option:  ({True, False}, optional)
        If True, the XT3D formulation will be used. By default False.
    rhs_option: ({True, False}, optional)
        If True, then the XT3D additional terms will be added to the right-hand
        side. If False, then the XT3D terms will be put into the coefficient
        matrix. By default False.
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
            AllCoordsValueSchema("layer", ">", 0),
        ],
        "k": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
            AllCoordsValueSchema("layer", ">", 0),
        ],
        "rewet_layer": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
            AllCoordsValueSchema("layer", ">", 0),
        ],
        "k22": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
            AllCoordsValueSchema("layer", ">", 0),
        ],
        "k33": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
            AllCoordsValueSchema("layer", ">", 0),
        ],
        "angle1": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
            AllCoordsValueSchema("layer", ">", 0),
        ],
        "angle2": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
            AllCoordsValueSchema("layer", ">", 0),
        ],
        "angle3": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
            AllCoordsValueSchema("layer", ">", 0),
        ],
        "alternative_cell_averaging": [
            DTypeSchema(str),
            DimsSchema(),
            CompatibleSettingsSchema("xt3d_option", False),
        ],
        "rhs_option": [
            DTypeSchema(np.bool_),
            DimsSchema(),
            CompatibleSettingsSchema("xt3d_option", True),
        ],
        "save_flows": [DTypeSchema(np.bool_), DimsSchema()],
        "starting_head_as_confined_thickness": [DTypeSchema(np.bool_)],
        "variable_vertical_conductance": [DTypeSchema(np.bool_)],
        "dewatered": [DTypeSchema(np.bool_)],
        "perched": [DTypeSchema(np.bool_)],
        "save_specific_discharge": [DTypeSchema(np.bool_), DimsSchema()],
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
        "xt3d_option": "xt3doptions",
        "rhs_option": "rhs",
    }
    _template = Package._initialize_template(_pkg_id)
    _regrid_method = NodePropertyFlowRegridMethod()

    @init_log_decorator()
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
        alternative_cell_averaging=None,
        save_flows=False,
        starting_head_as_confined_thickness=False,
        variable_vertical_conductance=False,
        dewatered=False,
        perched=False,
        save_specific_discharge=False,
        save_saturation=False,
        xt3d_option=False,
        rhs_option=False,
        validate: bool = True,
    ):
        # check rewetting
        if not rewet and any(
            [rewet_layer, rewet_factor, rewet_iterations, rewet_method]
        ):
            raise ValueError(
                "rewet_layer, rewet_factor, rewet_iterations, and rewet_method should"
                " all be left at a default value of None if rewet is False."
            )

        dict_dataset = {
            "icelltype": icelltype,
            "k": k,
            "rewet": rewet,
            "rewet_layer": rewet_layer,
            "rewet_factor": rewet_factor,
            "rewet_iterations": rewet_iterations,
            "rewet_method": rewet_method,
            "k22": k22,
            "k33": k33,
            "angle1": angle1,
            "angle2": angle2,
            "angle3": angle3,
            "alternative_cell_averaging": alternative_cell_averaging,
            "save_flows": save_flows,
            "starting_head_as_confined_thickness": starting_head_as_confined_thickness,
            "variable_vertical_conductance": variable_vertical_conductance,
            "dewatered": dewatered,
            "perched": perched,
            "save_specific_discharge": save_specific_discharge,
            "save_saturation": save_saturation,
            "xt3d_option": xt3d_option,
            "rhs_option": rhs_option,
        }
        super().__init__(dict_dataset)
        self._validate_init_schemata(validate)

    def get_xt3d_option(self) -> bool:
        """
        Returns the xt3d option value for this object.
        """
        return _dataarray_to_bool(self.dataset["xt3d_option"])

    def set_xt3d_option(self, is_xt3d_used: bool, is_rhs: bool) -> None:
        """
        Returns the xt3d option value for this object.
        """
        self.dataset["rhs_option"] = is_rhs
        self.dataset["xt3d_option"] = is_xt3d_used

    @property
    def is_variable_vertical_conductance(self) -> bool:
        """
        Returns the VariableCV option value for this object.
        """
        return _dataarray_to_bool(self.dataset["variable_vertical_conductance"])

    @property
    def is_dewatered(self) -> bool:
        """
        Returns the "dewatered" option value for this object. Used only when variable_vertical_conductance is true
        """
        return _dataarray_to_bool(self.dataset["dewatered"])

    def _validate(self, schemata, **kwargs):
        # Insert additional kwargs
        kwargs["xt3d_option"] = self["xt3d_option"]
        errors = super()._validate(schemata, **kwargs)

        return errors

    @classmethod
    def from_imod5_data(
        cls,
        imod5_data: dict[str, dict[str, GridDataArray]],
        target_grid: GridDataArray,
        regridder_types: Optional[NodePropertyFlowRegridMethod] = None,
        regrid_cache: RegridderWeightsCache = RegridderWeightsCache(),
    ) -> "NodePropertyFlow":
        """
        Construct an npf-package from iMOD5 data, loaded with the
        :func:`imod.formats.prj.open_projectfile_data` function.

        .. note::

            The method expects the iMOD5 model to be fully 3D, not quasi-3D.

        Parameters
        ----------
        imod5_data: dict
            Dictionary with iMOD5 data. This can be constructed from the
            :func:`imod.formats.prj.open_projectfile_data` method.
        target_grid: GridDataArray
            The grid that should be used for the new package. Does not
            need to be identical to one of the input grids.
        regridder_types: RegridMethodType, optional
            Optional dataclass with regridder types for a specific variable.
            Use this to override default regridding methods.
        regrid_cache: RegridderWeightsCache, optional
            stores regridder weights for different regridders. Can be used to speed up regridding,
            if the same regridders are used several times for regridding different arrays.

        Returns
        -------
        Modflow 6 npf package.

        """

        data = {
            "k": imod5_data["khv"]["kh"],
        }
        has_vertical_anisotropy = (
            "kva" in imod5_data.keys()
            and "vertical_anisotropy" in imod5_data["kva"].keys()
        )
        has_horizontal_anisotropy = "ani" in imod5_data.keys()

        if has_vertical_anisotropy:
            data["k33"] = data["k"] * imod5_data["kva"]["vertical_anisotropy"]
        if has_horizontal_anisotropy:
            if not np.all(np.isnan(imod5_data["ani"]["factor"].values)):
                factor = imod5_data["ani"]["factor"]
                factor = fill_missing_layers(factor, target_grid, 1.0)
                data["k22"] = data["k"] * factor
            if not np.all(np.isnan(imod5_data["ani"]["angle"].values)):
                angle1 = imod5_data["ani"]["angle"]
                angle1 = 90.0 - angle1
                angle1 = xr.where(angle1 < 0, 360.0 + angle1, angle1)
                angle1 = fill_missing_layers(angle1, target_grid, 90.0)
                data["angle1"] = angle1

        icelltype = zeros_like(target_grid, dtype=int)

        if regridder_types is None:
            regridder_types = NodePropertyFlow.get_regrid_methods()

        new_package_data = _regrid_package_data(
            data, target_grid, regridder_types, regrid_cache, {}
        )
        new_package_data["icelltype"] = icelltype

        pkg = cls(**new_package_data, validate=True)
        pkg.dataset.load()  # Force dask dataset into memory
        return pkg

    @classmethod
    def get_regrid_methods(cls) -> NodePropertyFlowRegridMethod:
        return deepcopy(cls._regrid_method)
