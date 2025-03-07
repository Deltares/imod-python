from __future__ import annotations

import abc
import pathlib
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union, cast

import cftime
import jinja2
import numpy as np
import xarray as xr
import xugrid as xu

from imod.common.interfaces.ipackage import IPackage
from imod.common.utilities.clip import (
    clip_layer_slice,
    clip_spatial_box,
    clip_time_slice,
)
from imod.common.utilities.mask import mask_package
from imod.common.utilities.regrid import (
    _regrid_like,
)
from imod.common.utilities.regrid_method_type import EmptyRegridMethod, RegridMethodType
from imod.common.utilities.schemata import filter_schemata_dict
from imod.common.utilities.value_filters import is_valid
from imod.logging import standard_log_decorator
from imod.mf6.auxiliary_variables import (
    expand_transient_auxiliary_variables,
    get_variable_names,
    remove_expanded_auxiliary_variables_from_dataset,
)
from imod.mf6.pkgbase import (
    EXCHANGE_PACKAGES,
    TRANSPORT_PACKAGES,
    PackageBase,
)
from imod.mf6.validation import validation_pkg_error_message
from imod.mf6.write_context import WriteContext
from imod.schemata import (
    AllNoDataSchema,
    EmptyIndexesSchema,
    SchemaType,
    ValidationError,
)
from imod.typing import GridDataArray
from imod.util.regrid import RegridderWeightsCache


class Package(PackageBase, IPackage, abc.ABC):
    """
    Package is used to share methods for specific packages with no time
    component.

    It is not meant to be used directly, only to inherit from, to implement new
    packages.

    This class only supports `array input
    <https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=16>`_,
    not the list input which is used in :class:`BoundaryCondition`.
    """

    _pkg_id = ""
    _init_schemata: dict[str, list[SchemaType] | Tuple[SchemaType, ...]] = {}
    _write_schemata: dict[str, list[SchemaType] | Tuple[SchemaType, ...]] = {}
    _keyword_map: dict[str, str] = {}
    _regrid_method: RegridMethodType = EmptyRegridMethod()
    _template: jinja2.Template

    def __init__(self, allargs: Mapping[str, GridDataArray | float | int | bool | str]):
        super().__init__(allargs)

    def isel(self):
        raise NotImplementedError(
            "Selection on packages not yet supported. To make a selection on "
            f"the xr.Dataset, call {self._pkg_id}.dataset.isel instead."
            "You can create a new package with a selection by calling "
            f"{type(self).__name__}(**{self._pkg_id}.dataset.isel(**selection))"
        )

    def sel(self):
        raise NotImplementedError(
            "Selection on packages not yet supported. To make a selection on "
            f"the xr.Dataset, call {self._pkg_id}.dataset.sel instead. "
            "You can create a new package with a selection by calling "
            f"{type(self).__name__}(**{self._pkg_id}.dataset.sel(**selection))"
        )

    def cleanup(self, dis: Any):
        raise NotImplementedError("Method not implemented for this package.")

    @staticmethod
    def _valid(value: Any) -> bool:
        return is_valid(value)

    @staticmethod
    def _number_format(dtype: type):
        if np.issubdtype(dtype, np.integer):
            return "%i"
        elif np.issubdtype(dtype, np.floating):
            return "%.18G"
        else:
            raise TypeError("dtype should be either integer or float")

    @staticmethod
    def _initialize_template(pkg_id):
        loader = jinja2.PackageLoader("imod", "templates/mf6")
        env = jinja2.Environment(loader=loader, keep_trailing_newline=True)
        if pkg_id == "ims":
            fname = "sln-ims.j2"
        elif pkg_id == "tdis":
            fname = "sim-tdis.j2"
        elif pkg_id in TRANSPORT_PACKAGES:
            fname = f"gwt-{pkg_id}.j2"
        elif pkg_id in EXCHANGE_PACKAGES:
            fname = f"exg-{pkg_id}.j2"
        elif pkg_id == "api":
            fname = f"{pkg_id}.j2"
        else:
            fname = f"gwf-{pkg_id}.j2"
        return env.get_template(fname)

    def write_blockfile(self, pkgname, globaltimes, write_context: WriteContext):
        directory = write_context.get_formatted_write_directory()

        content = self.render(
            directory=directory,
            pkgname=pkgname,
            globaltimes=globaltimes,
            binary=write_context.use_binary,
        )
        filename = write_context.write_directory / f"{pkgname}.{self._pkg_id}"
        with open(filename, "w") as f:
            f.write(content)

    def write_binary_griddata(self, outpath, da, dtype):
        # From the modflow6 source, the header is defined as:
        # integer(I4B) :: kstp --> np.int32 : 1
        # integer(I4B) :: kper --> np.int32 : 2
        # real(DP) :: pertim --> 2 * np.int32 : 4
        # real(DP) :: totim --> 2 * np.int32 : 6
        # character(len=16) :: text --> 4 * np.int32 : 10
        # integer(I4B) :: m1, m2, m3 --> 3 * np.int32 : 13
        # so writing 13 bytes suffices to create a header.

        # The following code is commented out due to modflow issue 189
        # https://github.com/MODFLOW-USGS/modflow6/issues/189
        # We never write LAYERED data.
        # The (structured) dis array reader results in an error if you try to
        # read a 3D botm array. By storing nlayer * nrow * ncol in the first
        # header entry, the array is read properly.

        # haslayer = "layer" in da.dims
        # if haslayer:
        #    nlayer, nrow, ncol = da.shape
        # else:
        #    nrow, ncol = da.shape
        #    nlayer = 1

        # This is a work around for the abovementioned issue.
        nval = np.prod(da.shape)
        header = np.zeros(13, np.int32)
        header[-3] = np.int32(nval)  # ncol
        header[-2] = np.int32(1)  # nrow
        header[-1] = np.int32(1)  # nlayer

        with open(outpath, "w") as f:
            header.tofile(f)
            da.values.flatten().astype(dtype).tofile(f)

    def write_text_griddata(self, outpath, da, dtype):
        with open(outpath, "w") as f:
            # Note: reshaping here avoids writing newlines after every number.
            # This dumps all the values in a single row rather than a single
            # column. This is to be preferred, since editors can easily
            # "reshape" a long row with "word wrap"; they cannot as easily
            # ignore newlines.
            fmt = self._number_format(dtype)
            data = da.values
            if data.ndim > 2:
                np.savetxt(fname=f, X=da.values.reshape((1, -1)), fmt=fmt)
            else:
                np.savetxt(fname=f, X=da.values, fmt=fmt)

    def _get_render_dictionary(
        self,
        directory: pathlib.Path,
        pkgname: str,
        globaltimes: Union[list[np.datetime64], np.ndarray],
        binary: bool,
    ) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if directory is None:
            pkg_directory = pkgname
        else:
            pkg_directory = pathlib.Path(directory) / pkgname

        for varname in self.dataset.data_vars:
            key = self._keyword_map.get(str(varname), str(varname))

            if hasattr(self, "_grid_data") and varname in self._grid_data:
                layered, value = self._compose_values(
                    self.dataset[varname], pkg_directory, key, binary=binary
                )
                if self._valid(value):  # skip False or None
                    d[f"{key}_layered"] = layered
                    d[key] = value
            else:
                value = self[varname].values[()]
                if self._valid(value):  # skip False or None
                    d[key] = value

        if (hasattr(self, "_auxiliary_data")) and (names := get_variable_names(self)):
            d["auxiliary"] = names
        return d

    def render(self, directory, pkgname, globaltimes, binary):
        d = self._get_render_dictionary(directory, pkgname, globaltimes, binary)
        return self._template.render(d)

    @staticmethod
    def _is_xy_data(obj):
        if isinstance(obj, (xr.DataArray, xr.Dataset)):
            xy = "x" in obj.dims and "y" in obj.dims
        elif isinstance(obj, (xu.UgridDataArray, xu.UgridDataset)):
            xy = obj.ugrid.grid.face_dimension in obj.dims
        else:
            raise TypeError(
                "obj should be DataArray or UgridDataArray, "
                f"received {type(obj)} instead"
            )
        return xy

    def _compose_values(
        self, da, directory, name, binary
    ) -> Tuple[bool, Optional[List[str]]]:
        """
        Compose values of dictionary.

        Ignores times. Time dependent boundary conditions use the method from
        BoundaryCondition.

        See documentation of wq
        """
        layered = False
        values = []
        if self._is_xy_data(da):
            if binary:
                path = (directory / f"{name}.bin").as_posix()
                values.append(f"open/close {path} (binary)")
            else:
                path = (directory / f"{name}.dat").as_posix()
                values.append(f"open/close {path}")
        else:
            if "layer" in da.dims:
                layered = True
                for layer in da.coords["layer"]:
                    values.append(f"constant {da.sel(layer=layer).values[()]}")
            else:
                value = da.values[()]
                if self._valid(value):  # skip None or False
                    values.append(f"constant {value}")
                else:
                    return layered, None

        return layered, values

    def write(
        self,
        pkgname: str,
        globaltimes: list[np.datetime64] | np.ndarray,
        directory: str | Path,
        use_binary: bool = False,
        use_absolute_paths: bool = False,
    ):
        """
        Write package to file

        Parameters
        ----------
        pkgname: str
            Package name
        globaltimes: array of np.datetime64
            Times of the simulation's stress periods.
        directory: str or Path
            Directory to write package in
        use_binary: ({True, False}, optional)
            Whether to write time-dependent input for stress packages as binary
            files, which are smaller in size, or more human-readable text files.
        use_absolute_paths: ({True, False}, optional)
            True if all paths written to the mf6 inputfiles should be absolute.
        """
        write_context = WriteContext(
            Path(directory),
            use_binary,
            use_absolute_paths,
        )
        self._write(pkgname, globaltimes, write_context)

    @standard_log_decorator()
    def _write(
        self,
        pkgname: str,
        globaltimes: Union[list[np.datetime64], np.ndarray],
        write_context: WriteContext,
    ):
        directory = write_context.write_directory
        binary = write_context.use_binary
        self.write_blockfile(pkgname, globaltimes, write_context)

        if hasattr(self, "_grid_data"):
            if self._is_xy_data(self.dataset):
                pkgdirectory = directory / pkgname
                pkgdirectory.mkdir(exist_ok=True, parents=True)
                for varname, dtype in self._grid_data.items():
                    key = self._keyword_map.get(varname, varname)
                    da = self.dataset[varname]
                    if self._is_xy_data(da):
                        if binary:
                            path = pkgdirectory / f"{key}.bin"
                            self.write_binary_griddata(path, da, dtype)
                        else:
                            path = pkgdirectory / f"{key}.dat"
                            self.write_text_griddata(path, da, dtype)

    @standard_log_decorator()
    def _validate(self, schemata: dict, **kwargs) -> dict[str, list[ValidationError]]:
        errors = defaultdict(list)
        for variable, var_schemata in schemata.items():
            for schema in var_schemata:
                if (
                    variable in self.dataset.keys()
                ):  # concentration only added to dataset if specified
                    try:
                        schema.validate(self.dataset[variable], **kwargs)
                    except ValidationError as e:
                        errors[variable].append(e)
        return errors

    def is_empty(self) -> bool:
        """
        Returns True if the package is empty- for example if it contains only no-data values.
        """

        # Create schemata dict only containing the
        # variables with a AllNoDataSchema and EmptyIndexesSchema (in case of
        # HFB) in the write schemata.
        allnodata_schemata = filter_schemata_dict(
            self._write_schemata, (AllNoDataSchema, EmptyIndexesSchema)
        )

        # Find if packages throws ValidationError for AllNoDataSchema or
        # EmptyIndexesSchema.
        allnodata_errors = self._validate(allnodata_schemata)
        return len(allnodata_errors) > 0

    def _validate_init_schemata(self, validate: bool):
        """
        Run the "cheap" schema validations.

        The expensive validations are run during writing. Some are only
        available then: e.g. idomain to determine active part of domain.
        """
        if not validate:
            return
        errors = self._validate(self._init_schemata)
        if len(errors) > 0:
            message = validation_pkg_error_message(errors)
            raise ValidationError(message)
        return

    def copy(self) -> Any:
        # All state should be contained in the dataset.
        return type(self)(**self.dataset.copy().to_dict())

    def clip_box(
        self,
        time_min: Optional[cftime.datetime | np.datetime64 | str] = None,
        time_max: Optional[cftime.datetime | np.datetime64 | str] = None,
        layer_min: Optional[int] = None,
        layer_max: Optional[int] = None,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        top: Optional[GridDataArray] = None,
        bottom: Optional[GridDataArray] = None,
    ) -> Package:
        """
        Clip a package by a bounding box (time, layer, y, x).

        Slicing intervals may be half-bounded, by providing None:

        * To select 500.0 <= x <= 1000.0:
          ``clip_box(x_min=500.0, x_max=1000.0)``.
        * To select x <= 1000.0: ``clip_box(x_min=None, x_max=1000.0)``
          or ``clip_box(x_max=1000.0)``.
        * To select x >= 500.0: ``clip_box(x_min = 500.0, x_max=None.0)``
          or ``clip_box(x_min=1000.0)``.

        Parameters
        ----------
        time_min: optional
        time_max: optional
        layer_min: optional, int
        layer_max: optional, int
        x_min: optional, float
        x_max: optional, float
        y_min: optional, float
        y_max: optional, float
        top: optional, GridDataArray
        bottom: optional, GridDataArray
        state_for_boundary: optional, GridDataArray


        Returns
        -------
        clipped: Package
        """
        if not self.is_clipping_supported():
            raise ValueError("this package does not support clipping.")

        selection = self.dataset
        selection = clip_time_slice(selection, time_min=time_min, time_max=time_max)
        selection = clip_layer_slice(
            selection, layer_min=layer_min, layer_max=layer_max
        )
        selection = clip_spatial_box(
            selection,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

        cls = type(self)
        return cls._from_dataset(selection)

    def mask(self, mask: GridDataArray) -> Any:
        """
        Mask values outside of domain.

        Floating values outside of the condition are set to NaN (nodata).
        Integer values outside of the condition are set to 0 (inactive in
        MODFLOW terms).

        Parameters
        ----------
        mask: xr.DataArray, xu.UgridDataArray of ints
            idomain-like integer array. >0 sets cells to active, 0 sets cells to inactive,
            <0 sets cells to vertical passthrough

        Returns
        -------
        masked: Package
            The package with part masked.
        """
        result = cast(IPackage, deepcopy(self))
        remove_expanded_auxiliary_variables_from_dataset(result)
        result = mask_package(result, mask)
        expand_transient_auxiliary_variables(result)
        return result

    def regrid_like(
        self,
        target_grid: GridDataArray,
        regrid_cache: RegridderWeightsCache,
        regridder_types: Optional[RegridMethodType] = None,
    ) -> "Package":
        """
        Creates a package of the same type as this package, based on another
        discretization. It regrids all the arrays in this package to the desired
        discretization, and leaves the options unmodified. At the moment only
        regridding to a different planar grid is supported, meaning
        ``target_grid`` has different ``"x"`` and ``"y"`` or different
        ``cell2d`` coords.

        The default regridding methods are specified in the ``_regrid_method``
        attribute of the package. These defaults can be overridden using the
        input parameters of this function.

        Parameters
        ----------
        target_grid: xr.DataArray or xu.UgridDataArray
            a grid defined using the same discretization as the one we want to
            regrid the package to.
        regrid_cache: RegridderWeightsCache
            stores regridder weights for different regridders. Can be used to
            speed up regridding, if the same regridders are used several times
            for regridding different arrays.
        regridder_types: RegridMethodType, optional
            dictionary mapping arraynames (str) to a tuple of regrid type (a
            specialization class of BaseRegridder) and function name (str) this
            dictionary can be used to override the default mapping method.

        Examples
        --------
        To regrid the npf package with a non-default method for the k-field,
        call ``regrid_like`` with these arguments:

        >>> regridder_types = imod.mf6.regrid.NodePropertyFlowRegridMethod(k=(imod.RegridderType.OVERLAP, "mean"))
        >>> regrid_cache = imod.util.regrid.RegridderWeightsCache()
        >>> new_npf = npf.regrid_like(like, regrid_cache, regridder_types)

        Returns
        -------
        A package with the same options as this package, and with all the
        data-arrays regridded to another discretization, similar to the one used
        in input argument "target_grid"
        """
        try:
            result = deepcopy(self)
            remove_expanded_auxiliary_variables_from_dataset(result)
            result = _regrid_like(result, target_grid, regrid_cache, regridder_types)
            expand_transient_auxiliary_variables(result)
        except ValueError as e:
            raise e
        except Exception:
            raise ValueError("package could not be regridded.")
        return result

    @classmethod
    def is_grid_agnostic_package(cls) -> bool:
        return False

    def __repr__(self) -> str:
        typename = type(self).__name__
        return f"{typename}\n{self.dataset.__repr__()}"

    def _repr_html_(self) -> str:
        typename = type(self).__name__
        return f"<div>{typename}</div>{self.dataset._repr_html_()}"

    @property
    def auxiliary_data_fields(self) -> dict[str, str]:
        if hasattr(self, "_auxiliary_data"):
            return self._auxiliary_data
        return {}

    def get_non_grid_data(self, grid_names: list[str]) -> dict[str, Any]:
        """
        This function copies the attributes of a dataset that are scalars, such as options.

        parameters
        ----------
        grid_names: list of str
            the names of the attribbutes of a dataset that are grids.
        """
        result = {}
        all_non_grid_data = list(self.dataset.keys())
        for name in (
            gridname for gridname in grid_names if gridname in all_non_grid_data
        ):
            all_non_grid_data.remove(name)
        for name in all_non_grid_data:
            if "time" in self.dataset[name].coords:
                result[name] = self.dataset[name]
            else:
                result[name] = self.dataset[name].values[()]
        return result

    def _call_func_on_grids(
        self, func: Callable, dis: dict
    ) -> dict[str, GridDataArray]:
        """
        Call function on dictionary of grids and merge settings back into
        dictionary.

        Parameters
        ----------
        func: Callable
            Function to call on all grids
        """
        grid_varnames = list(self._write_schemata.keys())
        grids = {
            varname: self.dataset[varname]
            for varname in grid_varnames
            if varname in self.dataset.keys()
        }
        cleaned_grids = func(**dis, **grids)
        settings = self.get_non_grid_data(grid_varnames)
        return cleaned_grids | settings

    def is_splitting_supported(self) -> bool:
        return True

    def is_regridding_supported(self) -> bool:
        return True

    def is_clipping_supported(self) -> bool:
        return True

    @classmethod
    def get_regrid_methods(cls) -> RegridMethodType:
        return deepcopy(cls._regrid_method)
