import pathlib
from copy import deepcopy
from typing import Any, List, Optional, Union

import numpy as np

import imod
from imod.common.interfaces.imaskingsettings import IMaskingSettings
from imod.common.interfaces.iregridpackage import IRegridPackage
from imod.common.utilities.dataclass_type import DataclassType
from imod.common.utilities.grid import create_smallest_target_grid
from imod.common.utilities.regrid import (
    _regrid_like,
    _regrid_package_data,
)
from imod.logging import init_log_decorator, standard_log_decorator
from imod.mf6.disv import VerticesDiscretization
from imod.mf6.package import Package
from imod.mf6.regrid.regrid_schemes import DiscretizationRegridMethod
from imod.mf6.utilities.imod5_converter import convert_ibound_to_idomain
from imod.mf6.validation import DisBottomSchema
from imod.schemata import (
    ActiveCellsConnectedSchema,
    AllCoordsValueSchema,
    AllValueSchema,
    AnyValueSchema,
    DimsSchema,
    DTypeSchema,
    IdentityNoDataSchema,
    IndexesSchema,
    ValidationError,
)
from imod.typing.grid import GridDataArray, is_unstructured
from imod.util.regrid import RegridderWeightsCache


class StructuredDiscretization(Package, IRegridPackage, IMaskingSettings):
    """
    Discretization information for structered grids is specified using the file.
    (DIS6) Only one discretization input file (DISU6, DISV6 or DIS6) can be
    specified for a model.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=35

    Parameters
    ----------
    top: array of floats (xr.DataArray)
        is the top elevation for each cell in the top model layer.
    bottom: array of floats (xr.DataArray)
        is the bottom elevation for each cell.
    idomain: array of integers (xr.DataArray)
        Indicates the existence status of a cell as follows:

        * If 0, the cell does not exist in the simulation. Input and output
          values will be read and written for the cell, but internal to the
          program, the cell is excluded from the solution.
        * If >0, the cell exists in the simulation.
        * If <0, the cell does not exist in the simulation. Furthermore, the
          first existing cell above will be connected to the first existing cell
          below. This type of cell is referred to as a "vertical pass through"
          cell.

        This DataArray needs contain a ``"layer"``, ``"y"`` and ``"x"``
        coordinate. Horizontal discretization information will be derived from
        its x and y coordinates.
    validate: {True, False}
        Flag to indicate whether the package should be validated upon
        initialization. This raises a ValidationError if package input is
        provided in the wrong manner. Defaults to True.
    """

    _pkg_id = "dis"
    _init_schemata = {
        "top": [
            DTypeSchema(np.floating),
            DimsSchema("y", "x") | DimsSchema(),
            IndexesSchema(),
        ],
        "bottom": [
            DTypeSchema(np.floating),
            DimsSchema("layer", "y", "x") | DimsSchema("layer"),
            IndexesSchema(),
            AllCoordsValueSchema("layer", ">", 0),
        ],
        "idomain": [
            DTypeSchema(np.integer),
            DimsSchema("layer", "y", "x"),
            IndexesSchema(),
            AllCoordsValueSchema("layer", ">", 0),
        ],
    }
    _write_schemata = {
        "idomain": (
            ActiveCellsConnectedSchema(is_notnull=("!=", 0)),
            AnyValueSchema(">", 0),
        ),
        "top": (
            AllValueSchema(">", "bottom", ignore=("idomain", "<=", 0)),
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
            # No need to check coords: dataset ensures they align with idomain.
        ),
        "bottom": (DisBottomSchema(other="idomain", is_other_notnull=(">", 0)),),
    }

    _grid_data = {"top": np.float64, "bottom": np.float64, "idomain": np.int32}
    _keyword_map = {"bottom": "botm"}
    _template = Package._initialize_template(_pkg_id)
    _regrid_method = DiscretizationRegridMethod()

    @property
    def skip_variables(self) -> List[str]:
        return ["bottom"]

    @init_log_decorator()
    def __init__(
        self,
        top: GridDataArray,
        bottom: GridDataArray,
        idomain: GridDataArray,
        validate: bool = True,
    ):
        dict_dataset = {
            "idomain": idomain,
            "top": top,
            "bottom": bottom,
        }
        super().__init__(dict_dataset)
        self._validate_init_schemata(validate)

    def _delrc(self, dx):
        """
        dx means dx or dy
        """
        if isinstance(dx, (int, float)):
            return f"constant {dx}"
        elif isinstance(dx, np.ndarray):
            arrstr = str(dx)[1:-1]
            return f"internal\n    {arrstr}"
        else:
            raise ValueError(f"Unhandled type of {dx}")

    def _get_render_dictionary(self, directory, pkgname, globaltimes, binary):
        disdirectory = pathlib.Path(directory) / pkgname
        d: dict[str, Any] = {}
        x = self.dataset["idomain"].coords["x"]
        y = self.dataset["idomain"].coords["y"]
        dx, xmin, _ = imod.util.spatial.coord_reference(x)
        dy, ymin, _ = imod.util.spatial.coord_reference(y)

        d["xorigin"] = xmin
        d["yorigin"] = ymin
        d["nlay"] = self.dataset["idomain"].coords["layer"].size
        d["nrow"] = y.size
        d["ncol"] = x.size
        d["delr"] = self._delrc(np.abs(dx))
        d["delc"] = self._delrc(np.abs(dy))
        _, d["top"] = self._compose_values(
            self["top"], disdirectory, "top", binary=binary
        )
        d["botm_layered"], d["botm"] = self._compose_values(
            self["bottom"], disdirectory, "botm", binary=binary
        )
        d["idomain_layered"], d["idomain"] = self._compose_values(
            self["idomain"], disdirectory, "idomain", binary=binary
        )
        return d

    def _validate(self, schemata, **kwargs):
        # Insert additional kwargs
        kwargs["bottom"] = self["bottom"]
        errors = super()._validate(schemata, **kwargs)

        return errors

    @classmethod
    @standard_log_decorator()
    def from_imod5_data(
        cls,
        imod5_data: dict[str, dict[str, GridDataArray]],
        regridder_types: Optional[DataclassType] = None,
        regrid_cache: RegridderWeightsCache = RegridderWeightsCache(),
        validate: bool = True,
    ) -> "StructuredDiscretization":
        """
        Construct package from iMOD5 data, loaded with the
        :func:`imod.formats.prj.open_projectfile_data` function.

        Method regrids all variables to a target grid with the smallest extent
        and smallest cellsize available in all the grids. Consequently it
        converts iMODFLOW data to MODFLOW 6 data.

        .. note::

            The method expects the iMOD5 model to be fully 3D, not quasi-3D.

        Parameters
        ----------
        imod5_data: dict
            Dictionary with iMOD5 data. This can be constructed from the
            :func:`imod.formats.prj.open_projectfile_data` method.
        regridder_types: DiscretizationRegridMethod, optional
            Optional dataclass with regridder types for a specific variable.
            Use this to override default regridding methods.
        regrid_cache: RegridderWeightsCache, optional
            stores regridder weights for different regridders. Can be used to speed up regridding,
            if the same regridders are used several times for regridding different arrays.

        Returns
        -------
        Modflow 6 StructuredDiscretization package.

        """
        data = {
            "idomain": imod5_data["bnd"]["ibound"].astype(np.int32),
            "top": imod5_data["top"]["top"],
            "bottom": imod5_data["bot"]["bottom"],
        }

        target_grid = create_smallest_target_grid(*data.values())

        if regridder_types is None:
            regridder_types = StructuredDiscretization.get_regrid_methods()

        new_package_data = _regrid_package_data(
            data, target_grid, regridder_types, regrid_cache
        )

        # Validate iMOD5 data
        if validate:
            if not np.all(
                new_package_data["top"][1:].data == new_package_data["bottom"][:-1].data
            ):
                raise ValidationError(
                    "Model discretization not fully 3D. Make sure TOP[n+1] matches BOT[n]"
                )

        thickness = new_package_data["top"] - new_package_data["bottom"]
        new_package_data["idomain"] = convert_ibound_to_idomain(
            new_package_data["idomain"], thickness
        )

        # TOP 3D -> TOP 2D
        # Assume iMOD5 data provided as fully 3D and not Quasi-3D
        new_package_data["top"] = new_package_data["top"].sel(layer=1, drop=True)

        pkg = cls(**new_package_data, validate=True)
        pkg.dataset.load()  # Force dask dataset into memory
        return pkg

    @classmethod
    def get_regrid_methods(cls) -> DiscretizationRegridMethod:
        return deepcopy(cls._regrid_method)

    def regrid_like(
        self,
        target_grid: GridDataArray,
        regrid_cache: RegridderWeightsCache,
        regridder_types: Optional[DataclassType] = None,
    ) -> Union["StructuredDiscretization", VerticesDiscretization]:
        """
        Regrid discretization package. Creates a StructuredDiscretization, or
        VerticesDiscretization package, based on type of target_grid. It regrids
        all the arrays in this package to the desired discretization, and leaves
        the options unmodified.

        The default regridding methods are specified in the ``_regrid_method``
        attribute of the package. These defaults can be overridden using the
        input parameters of this function.

        Parameters
        ----------
        target_grid: xr.DataArray or xu.UgridDataArray
            a grid defined using the same discretization as the one we want to
            regrid the package to.
        regrid_cache: RegridderWeightsCache, optional
            stores regridder weights for different regridders. Can be used to
            speed up regridding, if the same regridders are used several times
            for regridding different arrays.
        regridder_types: RegridMethodType, optional
            dictionary mapping arraynames (str) to a tuple of regrid type (a
            specialization class of BaseRegridder) and function name (str) this
            dictionary can be used to override the default mapping method.

        Returns
        -------
        StructuredDiscretization or VerticesDiscretization
            Depending on type of target_grid. A package with the same options as
            this package, and with all the data-arrays regridded to another
            discretization, similar to the one used in input argument "target_grid"
        """
        as_pkg_type = None
        if is_unstructured(target_grid):
            as_pkg_type = VerticesDiscretization

        try:
            result = _regrid_like(
                self, target_grid, regrid_cache, regridder_types, as_pkg_type
            )
        except ValueError as e:
            raise e
        except Exception:
            raise ValueError("package could not be regridded.")
        return result
