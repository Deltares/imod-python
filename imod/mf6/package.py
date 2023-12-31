import abc
import copy
import numbers
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import cftime
import jinja2
import numpy as np
import xarray as xr
import xugrid as xu
from xarray.core.utils import is_scalar

import imod
from imod.mf6.auxiliary_variables import get_variable_names
from imod.mf6.pkgbase import EXCHANGE_PACKAGES, TRANSPORT_PACKAGES, PackageBase
from imod.mf6.regridding_utils import (
    RegridderInstancesCollection,
    RegridderType,
    get_non_grid_data,
)
from imod.mf6.validation import validation_pkg_error_message
from imod.mf6.write_context import WriteContext
from imod.schemata import ValidationError
from imod.typing import GridDataArray


class Package(PackageBase, abc.ABC):
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
    _init_schemata = {}
    _write_schemata = {}

    def __init__(self, allargs=None):
        super().__init__(allargs)

    def isel(self):
        raise NotImplementedError(
            "Selection on packages not yet supported. To make a selection on "
            f"the xr.Dataset, call {self._pkg_id}.dataset.isel instead."
            "You can create a new package with a selection by calling "
            f"{__class__.__name__}(**{self._pkg_id}.dataset.isel(**selection))"
        )

    def sel(self):
        raise NotImplementedError(
            "Selection on packages not yet supported. To make a selection on "
            f"the xr.Dataset, call {self._pkg_id}.dataset.sel instead. "
            "You can create a new package with a selection by calling "
            f"{__class__.__name__}(**{self._pkg_id}.dataset.sel(**selection))"
        )

    def _valid(self, value):
        """
        Filters values that are None, False, or a numpy.bool_ False.
        Needs to be this specific, since 0.0 and 0 are valid values, but are
        equal to a boolean False.
        """
        # Test singletons
        if value is False or value is None:
            return False
        # Test numpy bool (not singleton)
        elif isinstance(value, np.bool_) and not value:
            return False
        # When dumping to netCDF and reading back, None will have been
        # converted into a NaN. Only check NaN if it's a floating type to avoid
        # TypeErrors.
        elif np.issubdtype(type(value), np.floating) and np.isnan(value):
            return False
        else:
            return True

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
        nval = np.product(da.shape)
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

    def render(self, directory, pkgname, globaltimes, binary):
        d = {}
        if directory is None:
            pkg_directory = pkgname
        else:
            pkg_directory = pathlib.Path(directory) / pkgname

        for varname in self.dataset.data_vars:
            key = self._keyword_map.get(varname, varname)

            if hasattr(self, "_grid_data") and varname in self._grid_data:
                layered, value = self._compose_values(
                    self.dataset[varname], pkg_directory, key, binary=binary
                )
                if self._valid(value):  # skip False or None
                    d[f"{key}_layered"], d[key] = layered, value
            else:
                value = self[varname].values[()]
                if self._valid(value):  # skip False or None
                    d[key] = value

        if (hasattr(self, "_auxiliary_data")) and (names := get_variable_names(self)):
            d["auxiliary"] = names

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

    def _compose_values(self, da, directory, name, binary):
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
                    values = None

        return layered, values

    def write(
        self,
        pkgname: str,
        globaltimes: Union[List, np.ndarray],
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

    def _validate(self, schemata: Dict, **kwargs) -> Dict[str, List[ValidationError]]:
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

    def _get_vars_to_check(self):
        """
        Helper function to get all variables which were not set to None
        """
        variables = []
        for var in self._metadata_dict.keys():
            if (  # Filter optional variables not filled in
                self.dataset[var].size != 1
            ) or (
                self.dataset[var] != None  # noqa: E711
            ):
                variables.append(var)

        return variables

    def copy(self) -> Any:
        # All state should be contained in the dataset.
        return type(self)(**self.dataset.copy())

    @staticmethod
    def _clip_repeat_stress(
        repeat_stress: xr.DataArray,
        time,
        time_start,
        time_end,
    ):
        """
        Selection may remove the original data which are repeated.
        These should be re-inserted at the first occuring "key".
        Next, remove these keys as they've been "promoted" to regular
        timestamps with data.
        """
        # First, "pop" and filter.
        keys, values = repeat_stress.values.T
        keep = (keys >= time_start) & (keys <= time_end)
        new_keys = keys[keep]
        new_values = values[keep]
        # Now detect which "value" entries have gone missing
        insert_values, index = np.unique(new_values, return_index=True)
        insert_keys = new_keys[index]
        # Setup indexer
        indexer = xr.DataArray(
            data=np.arange(time.size),
            coords={"time": time},
            dims=("time",),
        ).sel(time=insert_values)
        indexer["time"] = insert_keys

        # Update the key-value pairs. Discard keys that have been "promoted".
        keep = np.in1d(new_keys, insert_keys, assume_unique=True, invert=True)
        new_keys = new_keys[keep]
        new_values = new_values[keep]
        # Set the values to their new source.
        new_values = insert_keys[np.searchsorted(insert_values, new_values)]
        repeat_stress = xr.DataArray(
            data=np.column_stack((new_keys, new_values)),
            dims=("repeat", "repeat_items"),
        )
        return indexer, repeat_stress

    @staticmethod
    def _clip_time_indexer(
        time,
        time_start,
        time_end,
    ):
        original = xr.DataArray(
            data=np.arange(time.size),
            coords={"time": time},
            dims=("time",),
        )
        indexer = original.sel(time=slice(time_start, time_end))

        # The selection might return a 0-sized dimension.
        if indexer.size > 0:
            first_time = indexer["time"].values[0]
        else:
            first_time = None

        # If the first time matches exactly, xarray will have done thing we
        # wanted and our work with the time dimension is finished.
        if (time_start is not None) and (time_start != first_time):
            # If the first time is before the original time, we need to
            # backfill; otherwise, we need to ffill the first timestamp.
            if time_start < time[0]:
                method = "bfill"
            else:
                method = "ffill"
            # Index with a list rather than a scalar to preserve the time
            # dimension.
            first = original.sel(time=[time_start], method=method)
            first["time"] = [time_start]
            indexer = xr.concat([first, indexer], dim="time")

        return indexer

    def __to_datetime(self, time, use_cftime):
        """
        Helper function that converts to datetime, except when None.
        """
        if time is None:
            return time
        else:
            return imod.wq.timeutil.to_datetime(time, use_cftime)

    def clip_box(
        self,
        time_min=None,
        time_max=None,
        layer_min=None,
        layer_max=None,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        state_for_boundary=None,
    ) -> "Package":
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

        Returns
        -------
        clipped: Package
        """
        selection = self.dataset
        if "time" in selection:
            time = selection["time"].values
            use_cftime = isinstance(time[0], cftime.datetime)
            time_start = self.__to_datetime(time_min, use_cftime)
            time_end = self.__to_datetime(time_max, use_cftime)

            indexer = self._clip_time_indexer(
                time=time,
                time_start=time_start,
                time_end=time_end,
            )

            if "repeat_stress" in selection.data_vars and self._valid(
                selection["repeat_stress"].values[()]
            ):
                repeat_indexer, repeat_stress = self._clip_repeat_stress(
                    repeat_stress=selection["repeat_stress"],
                    time=time,
                    time_start=time_start,
                    time_end=time_end,
                )
                selection = selection.drop_vars("repeat_stress")
                selection["repeat_stress"] = repeat_stress
                indexer = repeat_indexer.combine_first(indexer).astype(int)

            selection = selection.drop_vars("time").isel(time=indexer)

        if "layer" in selection.coords:
            layer_slice = slice(layer_min, layer_max)
            # Cannot select if it's not a dimension!
            if "layer" not in selection.dims:
                selection = (
                    selection.expand_dims("layer")
                    .sel(layer=layer_slice)
                    .squeeze("layer")
                )
            else:
                selection = selection.sel(layer=layer_slice)

        x_slice = slice(x_min, x_max)
        y_slice = slice(y_min, y_max)
        if isinstance(selection, xu.UgridDataset):
            selection = selection.ugrid.sel(x=x_slice, y=y_slice)
        elif ("x" in selection.coords) and ("y" in selection.coords):
            if selection.indexes["y"].is_monotonic_decreasing:
                y_slice = slice(y_max, y_min)
            selection = selection.sel(x=x_slice, y=y_slice)

        cls = type(self)
        new = cls.__new__(cls)
        new.dataset = selection
        return new

    def mask(self, domain: GridDataArray) -> Any:
        """
        Mask values outside of domain.

        Floating values outside of the condition are set to NaN (nodata).
        Integer values outside of the condition are set to 0 (inactive in
        MODFLOW terms).

        Parameters
        ----------
        domain: xr.DataArray of integers. Preservers values where domain is larger than 0.

        Returns
        -------
        masked: Package
            The package with part masked.
        """
        masked = {}
        for var in self.dataset.data_vars.keys():
            da = self.dataset[var]
            if self.skip_masking_dataarray(var):
                masked[var] = da
                continue
            if set(domain.dims).issubset(da.dims):
                if issubclass(da.dtype.type, numbers.Integral):
                    masked[var] = da.where(domain > 0, other=0)
                elif issubclass(da.dtype.type, numbers.Real):
                    masked[var] = da.where(domain > 0)
                else:
                    raise TypeError(
                        f"Expected dtype float or integer. Received instead: {da.dtype}"
                    )
            else:
                if da.values[()] is not None:
                    if is_scalar(da.values[()]):
                        masked[var] = da.values[()]  # For scalars, such as options
                    else:
                        masked[
                            var
                        ] = da  # For example for arrays with only a layer dimension
                else:
                    masked[var] = None

        return type(self)(**masked)

    def is_regridding_supported(self) -> bool:
        """
        returns true if package supports regridding.
        """
        return hasattr(self, "_regrid_method")

    def get_regrid_methods(self) -> Optional[Dict[str, Tuple[RegridderType, str]]]:
        if self.is_regridding_supported():
            return self._regrid_method
        return None

    def _regrid_array(
        self,
        varname: str,
        regridder_collection: RegridderInstancesCollection,
        regridder_name: str,
        regridder_function: str,
        target_grid: GridDataArray,
    ) -> Optional[GridDataArray]:
        """
        Regrids a data_array. The array is specified by its key in the dataset.
        Each data-array can represent:
        -a scalar value, valid for the whole grid
        -an array of a different scalar per layer
        -an array with a value per grid block
        -None
        """

        # skip regridding for arrays with no valid values (such as "None")
        if not self._valid(self.dataset[varname].values[()]):
            return None

        # the dataarray might be a scalar. If it is, then it does not need regridding.
        if is_scalar(self.dataset[varname]):
            return self.dataset[varname].values[()]

        if isinstance(self.dataset[varname], xr.DataArray):
            coords = self.dataset[varname].coords
            # if it is an xr.DataArray it may be layer-based; then no regridding is needed
            if not ("x" in coords and "y" in coords):
                return self.dataset[varname]

            # if it is an xr.DataArray it needs the dx, dy coordinates for regridding, which are otherwise not mandatory
            if not ("dx" in coords and "dy" in coords):
                raise ValueError(
                    f"DataArray {varname} does not have both a dx and dy coordinates"
                )

        # obtain an instance of a regridder for the chosen method
        regridder = regridder_collection.get_regridder(
            regridder_name,
            regridder_function,
        )

        # store original dtype of data
        original_dtype = self.dataset[varname].dtype

        # regrid data array
        regridded_array = regridder.regrid(self.dataset[varname])

        # reconvert the result to the same dtype as the original
        return regridded_array.astype(original_dtype)

    def regrid_like(
        self,
        target_grid: GridDataArray,
        regridder_types: Dict[str, Tuple[RegridderType, str]] = None,
    ) -> "Package":
        """
        Creates a package of the same type as this package, based on another discretization.
        It regrids all the arrays in this package to the desired discretization, and leaves the options
        unmodified. At the moment only regridding to a different planar grid is supported, meaning
        ``target_grid`` has different ``"x"`` and ``"y"`` or different ``cell2d`` coords.

        The regridding methods can be specified in the _regrid_method attribute of the package. These are the defaults
        that specify how each array should be regridded. These defaults can be overridden using the input
        parameters of this function.

        Examples
        --------
        To regrid the npf package with a non-default method for the k-field, call regrid_like with these arguments:

        >>> new_npf = npf.regrid_like(like, {"k": (imod.RegridderType.OVERLAP, "mean")})


        Parameters
        ----------
        target_grid: xr.DataArray or xu.UgridDataArray
            a grid defined over the same discretization as the one we want to regrid the package to
        regridder_types: dict(str->(regridder type,str))
           dictionary mapping arraynames (str) to a tuple of regrid type (a specialization class of BaseRegridder) and function name (str)
            this dictionary can be used to override the default mapping method.

        Returns
        -------
        a package with the same options as this package, and with all the data-arrays regridded to another discretization,
        similar to the one used in input argument "target_grid"
        """
        if not self.is_regridding_supported():
            raise NotImplementedError(
                f"Package {type(self).__name__} does not support regridding"
            )

        regridder_collection = RegridderInstancesCollection(
            self.dataset, target_grid=target_grid
        )

        regridder_settings = copy.deepcopy(self._regrid_method)
        if regridder_types is not None:
            regridder_settings.update(regridder_types)

        new_package_data = get_non_grid_data(self, list(regridder_settings.keys()))

        for (
            varname,
            regridder_type_and_function,
        ) in regridder_settings.items():
            regridder_name, regridder_function = regridder_type_and_function

            # skip variables that are not in this dataset
            if varname not in self.dataset.keys():
                continue

            # regrid the variable
            new_package_data[varname] = self._regrid_array(
                varname,
                regridder_collection,
                regridder_name,
                regridder_function,
                target_grid,
            )

        new_package = self.__class__(**new_package_data)

        return new_package

    def skip_masking_dataarray(self, array_name: str) -> bool:
        if hasattr(self, "_skip_mask_arrays"):
            return array_name in self._skip_mask_arrays
        return False

    @classmethod
    def is_grid_agnostic_package(cls) -> bool:
        return False

    def __repr__(self) -> str:
        typename = type(self).__name__
        return f"{typename}\n{self.dataset.__repr__()}"

    def _repr_html_(self) -> str:
        typename = type(self).__name__
        return f"<div>{typename}</div>{self.dataset._repr_html_()}"
