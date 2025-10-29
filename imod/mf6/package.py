from __future__ import annotations

import abc
import pathlib
from copy import deepcopy
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Self,
    Tuple,
    Union,
    cast,
)

import cftime
import jinja2
import numpy as np
import xarray as xr
import xugrid as xu

from imod.common.interfaces.ipackage import IPackage
from imod.common.utilities.clip import (
    clip_box_dataset,
)
from imod.common.utilities.dataclass_type import DataclassType, EmptyRegridMethod
from imod.common.utilities.mask import mask_package
from imod.common.utilities.regrid import (
    _regrid_like,
)
from imod.common.utilities.schemata import (
    filter_schemata_dict,
    validate_schemata_dict,
    validate_with_error_message,
)
from imod.common.utilities.value_filters import is_valid
from imod.common.utilities.version import prepend_content_with_version_info
from imod.logging import standard_log_decorator
from imod.mf6.auxiliary_variables import (
    expand_transient_auxiliary_variables,
    get_variable_names,
    remove_expanded_auxiliary_variables_from_dataset,
)
from imod.mf6.pkgbase import (
    EXCHANGE_PACKAGES,
    TRANSPORT_PACKAGES,
    UTIL_PACKAGES,
    PackageBase,
)
from imod.mf6.validation_settings import ValidationSettings, trim_time_dimension
from imod.mf6.write_context import WriteContext
from imod.schemata import (
    AllNoDataSchema,
    EmptyIndexesSchema,
    SchemataDict,
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
    _init_schemata: SchemataDict = {}
    _write_schemata: SchemataDict = {}
    _keyword_map: dict[str, str] = {}
    _regrid_method: DataclassType = EmptyRegridMethod()
    _template: jinja2.Template

    def __init__(self, allargs: Mapping[str, GridDataArray | float | int | bool | str]):
        super().__init__(allargs)

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
        elif pkg_id in UTIL_PACKAGES:
            fname = f"utl-{pkg_id}.j2"
        elif pkg_id == "api":
            fname = f"{pkg_id}.j2"
        else:
            fname = f"gwf-{pkg_id}.j2"
        return env.get_template(fname)

    def _write_blockfile(self, pkgname, globaltimes, write_context: WriteContext):
        directory = write_context.get_formatted_write_directory()

        content = self._render(
            directory=directory,
            pkgname=pkgname,
            globaltimes=globaltimes,
            binary=write_context.use_binary,
        )
        content = prepend_content_with_version_info(content)
        filename = write_context.write_directory / f"{pkgname}.{self._pkg_id}"
        with open(filename, "w") as f:
            f.write(content)

    def _write_binary_griddata(self, outpath, da, dtype):
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

    def _write_text_griddata(self, outpath, da, dtype):
        # Note: reshaping here avoids writing newlines after every number.
        # This dumps all the values in a single row rather than a single
        # column. This is to be preferred, since editors can easily
        # "reshape" a long row with "word wrap"; they cannot as easily
        # ignore newlines.
        fmt = self._number_format(dtype)
        data = da.values
        if data.ndim > 2:
            np.savetxt(fname=outpath, X=data.reshape((1, -1)), fmt=fmt)
        else:
            np.savetxt(fname=outpath, X=data, fmt=fmt)

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

    def _render(self, directory, pkgname, globaltimes, binary):
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
    ) -> None:
        directory = write_context.write_directory
        binary = write_context.use_binary
        self._write_blockfile(pkgname, globaltimes, write_context)

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
                            self._write_binary_griddata(path, da, dtype)
                        else:
                            path = pkgdirectory / f"{key}.dat"
                            self._write_text_griddata(path, da, dtype)

    @standard_log_decorator()
    def _validate(self, schemata: dict, **kwargs) -> dict[str, list[ValidationError]]:
        ds = trim_time_dimension(self.dataset, **kwargs)
        return validate_schemata_dict(schemata, ds, **kwargs)

    def is_empty(self, ignore_time: bool = False) -> bool:
        """
        Returns True if the package is empty, that is if it contains only
        no-data values.

        Parameters
        ----------
        ignore_time: bool, optional
            If True, the first timestep is selected to validate. This increases
            performance for packages with a time dimensions over which changes
            of cell activity are not expected. Default is False, which means the
            time dimension is not dropped.
        """

        # Create schemata dict only containing the
        # variables with a AllNoDataSchema and EmptyIndexesSchema (in case of
        # HFB) in the write schemata.
        allnodata_schemata = filter_schemata_dict(
            self._write_schemata, (AllNoDataSchema, EmptyIndexesSchema)
        )
        validation_context = ValidationSettings(ignore_time=ignore_time)
        ds = trim_time_dimension(self.dataset, validation_context=validation_context)
        # Find if packages throws ValidationError for AllNoDataSchema or
        # EmptyIndexesSchema.
        allnodata_errors = validate_schemata_dict(allnodata_schemata, ds)
        return len(allnodata_errors) > 0

    def _validate_init_schemata(self, validate: bool, **kwargs) -> None:
        """
        Run the "cheap" schema validations.

        The expensive validations are run during writing. Some are only
        available then: e.g. idomain to determine active part of domain.
        """
        validate_with_error_message(
            self._validate, validate, self._init_schemata, **kwargs
        )

    def copy(self) -> Any:
        """
        Copy package into a new package of the same type.

        Returns
        -------
        Package
            A copy of the package, with the same type as this package.
        """
        # All state should be contained in the dataset.
        dataset_copy = cast(Mapping[str, Any], self.dataset.copy())
        return type(self)(**dataset_copy)

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
    ) -> Self:
        """
        Clip a package by a bounding box (time, layer, y, x).

        Parameters
        ----------
        time_min: optional, np.datetime64
            Start time to select. Data will be forward filled to this date. If
            time_min is before the start time of the dataset, data is
            backfilled.
        time_max: optional
            End time to select.
        layer_min: optional, int
            Minimum layer to select.
        layer_max: optional, int
            Maximum layer to select.
        x_min: optional, float
            Minimum x-coordinate to select.
        x_max: optional, float
            Maximum x-coordinate to select.
        y_min: optional, float
            Minimum y-coordinate to select.
        y_max: optional, float
            Maximum y-coordinate to select.
        top: optional, GridDataArray
            Ignored.
        bottom: optional, GridDataArray
            Ignored.

        Returns
        -------
        clipped : Package
            A new package that is clipped to the specified bounding box.

        Examples
        --------
        Slicing intervals may be half-bounded, by providing None:

        To select 500.0 <= x <= 1000.0:

        >>> pkg.clip_box(x_min=500.0, x_max=1000.0)

        To select x <= 1000.0:

        >>> pkg.clip_box(x_max=1000.0)``

        To select x >= 500.0:

        >>> pkg.clip_box(x_min=500.0)

        To select a time interval, you can use datetime64:

        >>> pkg.clip_box(time_min=np.datetime64("2020-01-01"), time_max=np.datetime64("2020-12-31"))

        """
        if not self._is_clipping_supported():
            raise ValueError("this package does not support clipping.")

        selection = clip_box_dataset(
            self.dataset,
            time_min,
            time_max,
            layer_min,
            layer_max,
            x_min,
            x_max,
            y_min,
            y_max,
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
            idomain-like integer array. >0 sets cells to active, 0 sets cells to
            inactive, <0 sets cells to vertical passthrough

        Returns
        -------
        masked: Package
            The package with part masked.

        Examples
        --------
        To mask a package with an idomain-like array. For example, to create a
        package with the first 10 rows and columns masked, create the mask first:

        >>> mask = xr.ones_like(idomain, dtype=int)
        >>> mask[0:10, 0:10] = 0

        Then call mask:

        >>> masked_pkg = pkg.mask(mask)
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
        regridder_types: Optional[DataclassType] = None,
    ) -> "Package":
        """
        Creates a package of the same type as this package, based on another
        discretization. It regrids all the arrays in this package to the desired
        discretization, and leaves the options unmodified. At the moment only
        regridding to a different planar grid is supported, meaning
        ``target_grid`` has different ``"x"`` and ``"y"`` or different
        ``cell2d`` coords.

        The default regridding methods are obtained by calling
        ``.get_regrid_methods()`` on the package, which returns a dataclass with
        the default regridding methods for each variable in the package.

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

        Returns
        -------
        Package
            A package with the same options as this package, and with all the
            data-arrays regridded to another discretization, similar to the one used
            in input argument "target_grid"

        Examples
        --------
        To regrid the npf package with a non-default method for the k-field,
        call ``regrid_like`` with these arguments:

        >>> regridder_types = imod.mf6.regrid.NodePropertyFlowRegridMethod(k=(imod.RegridderType.OVERLAP, "mean"))
        >>> regrid_cache = imod.util.regrid.RegridderWeightsCache()
        >>> new_npf = npf.regrid_like(like, regrid_cache, regridder_types)
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
    def _is_grid_agnostic_package(cls) -> bool:
        """
        Returns True if this package does not depend on a grid, e.g. the
        :class:`imod.mf6.wel.Wel` package.
        """
        return False

    @property
    def pkg_id(self) -> str:
        return self._pkg_id

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

    def _get_non_grid_data(self, grid_names: list[str]) -> dict[str, Any]:
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

        name = "repeat_stress"
        if name in all_non_grid_data:
            if "repeat" in self.dataset[name].dims:
                result[name] = self.dataset[name]
            else:
                result[name] = self.dataset[name].values[()]
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
        settings = self._get_non_grid_data(grid_varnames)
        return cleaned_grids | settings

    def _is_splitting_supported(self) -> bool:
        """
        Return True if this package supports splitting.

        Returns
        -------
        bool
            True if this package supports splitting, False otherwise.
        """
        return True

    def _is_regridding_supported(self) -> bool:
        """
        Return True if this package supports regridding.

        Returns
        -------
        bool
            True if this package supports regridding, False otherwise.
        """
        return True

    def _is_clipping_supported(self) -> bool:
        """
        Return True if this package supports clipping.

        Returns
        -------
        bool
            True if this package supports clipping, False otherwise.
        """
        return True

    @classmethod
    def get_regrid_methods(cls) -> DataclassType:
        """
        Returns the default regrid methods for this package. You can use modify
        to customize the regridding of the package.

        Returns
        -------
        DataclassType
            The regrid methods for this package, which is a dataclass with
            attributes that are tuples of (regridder type, method name). If no
            regrid methods are defined, returns an instance of
            EmptyRegridMethod.

        Examples
        --------
        Get the regrid methods for the Drainage package:

        >>> regrid_settings = Drainage.get_regrid_methods()

        You can modify the regrid methods by changing the attributes of the
        returned dataclass instance. For example, to set the regridding method
        for ``elevation`` to minimum.

        >>> regrid_settings.elevation = (imod.RegridderType.OVERLAP, "min")

        These settings can then be used to regrid the package:

        >>> drain.regrid_like(like, regridder_types=regrid_settings)

        """
        return deepcopy(cls._regrid_method)
