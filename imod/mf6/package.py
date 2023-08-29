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
from imod.mf6.pkgbase import TRANSPORT_PACKAGES, PackageBase
from imod.mf6.validation import validation_pkg_error_message
from imod.mf6.write_context import WriteContext
from imod.schemata import ValidationError


def dis_recarr(arrdict, layer, notnull):
    # Define the numpy structured array dtype
    index_spec = [("layer", np.int32), ("row", np.int32), ("column", np.int32)]
    field_spec = [(key, np.float64) for key in arrdict]
    sparse_dtype = np.dtype(index_spec + field_spec)
    # Initialize the structured array
    nrow = notnull.sum()
    recarr = np.empty(nrow, dtype=sparse_dtype)
    # Fill in the indices
    if notnull.ndim == 2:
        recarr["row"], recarr["column"] = (np.argwhere(notnull) + 1).transpose()
        recarr["layer"] = layer
    else:
        ilayer, irow, icolumn = np.argwhere(notnull).transpose()
        recarr["row"] = irow + 1
        recarr["column"] = icolumn + 1
        recarr["layer"] = layer[ilayer]
    return recarr


def disv_recarr(arrdict, layer, notnull):
    # Define the numpy structured array dtype
    index_spec = [("layer", np.int32), ("cell2d", np.int32)]
    field_spec = [(key, np.float64) for key in arrdict]
    sparse_dtype = np.dtype(index_spec + field_spec)
    # Initialize the structured array
    nrow = notnull.sum()
    recarr = np.empty(nrow, dtype=sparse_dtype)
    # Fill in the indices
    if notnull.ndim == 1 and layer.size == 1:
        recarr["cell2d"] = (np.argwhere(notnull) + 1).transpose()
        recarr["layer"] = layer
    else:
        ilayer, icell2d = np.argwhere(notnull).transpose()
        recarr["cell2d"] = icell2d + 1
        recarr["layer"] = layer[ilayer]
    return recarr


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

    def to_sparse(self, arrdict, layer):
        """Convert from dense arrays to list based input"""
        # TODO stream the data per stress period
        # TODO add pkgcheck that period table aligns
        # Get the number of valid values
        data = next(iter(arrdict.values()))
        notnull = ~np.isnan(data)

        if isinstance(self.dataset, xr.Dataset):
            recarr = dis_recarr(arrdict, layer, notnull)
        elif isinstance(self.dataset, xu.UgridDataset):
            recarr = disv_recarr(arrdict, layer, notnull)
        else:
            raise TypeError(
                "self.dataset should be xarray.Dataset or xugrid.UgridDataset,"
                f" is {type(self.dataset)} instead"
            )
        # Fill in the data
        for key, arr in arrdict.items():
            values = arr[notnull].astype(np.float64)
            recarr[key] = values

        return recarr

    def _ds_to_arrdict(self, ds):
        arrdict = {}
        for datavar in ds.data_vars:
            if ds[datavar].shape == ():
                raise ValueError(
                    f"{datavar} in {self._pkg_id} package cannot be a scalar"
                )
            auxiliary_vars = (
                self.get_auxiliary_variable_names()
            )  # returns something like {"concentration": "species"}
            if datavar in auxiliary_vars.keys():  # if datavar is concentration
                if (
                    auxiliary_vars[datavar] in ds[datavar].dims
                ):  # if this concentration array has the species dimension
                    for s in ds[datavar].values:  # loop over species
                        arrdict[s] = (
                            ds[datavar]
                            .sel({auxiliary_vars[datavar]: s})
                            .values  # store species array under its species name
                        )
            else:
                arrdict[datavar] = ds[datavar].values
        return arrdict

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
            directory = pkgname
        else:
            directory = pathlib.Path(directory) / pkgname

        for varname in self.dataset.data_vars:
            key = self._keyword_map.get(varname, varname)

            if hasattr(self, "_grid_data") and varname in self._grid_data:
                layered, value = self._compose_values(
                    self.dataset[varname], directory, key, binary=binary
                )
                if self._valid(value):  # skip False or None
                    d[f"{key}_layered"], d[key] = layered, value
            else:
                value = self[varname].values[()]
                if self._valid(value):  # skip False or None
                    d[key] = value
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

    def _netcdf_path(self, directory, pkgname):
        """create path for netcdf, this function is also used to create paths to use inside the qgis projectfiles"""
        return directory / pkgname / f"{self._pkg_id}.nc"

    def write_netcdf(self, directory, pkgname, aggregate_layers=False):
        """Write to netcdf. Useful for generating .qgs projectfiles to view model input.
        These files cannot be used to run a modflow model.

        Parameters
        ----------
        directory : Path
            directory of qgis project

        pkgname : str
            package name

        aggregate_layers : bool
            If True, aggregate layers by taking the mean, i.e. ds.mean(dim="layer")

        Returns
        -------
        has_dims : list of str
            list of variables that have an x and y dimension.

        """

        has_dims = []
        for varname in self.dataset.data_vars.keys():  # pylint:disable=no-member
            if all(i in self.dataset[varname].dims for i in ["x", "y"]):
                has_dims.append(varname)

        spatial_ds = self.dataset[has_dims]

        if aggregate_layers and ("layer" in spatial_ds.dims):
            spatial_ds = spatial_ds.mean(dim="layer")

        if "time" not in spatial_ds:
            # Hack to circumvent this issue:
            # https://github.com/lutraconsulting/MDAL/issues/300
            spatial_ds = spatial_ds.assign_coords(
                time=np.array("1970-01-01", dtype=np.datetime64)
            ).expand_dims(dim="time")

        path = self._netcdf_path(directory, pkgname)
        path.parent.mkdir(exist_ok=True, parents=True)

        spatial_ds.to_netcdf(path)
        return has_dims

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

    def period_data(self):
        result = []
        if hasattr(self, "_period_data"):
            result += self._period_data
        if hasattr(self, "_auxiliary_data"):
            for aux_var_name, aux_var_dimensions in self._auxiliary_data.items():
                if aux_var_name in self.dataset.keys():
                    for s in (
                        self.dataset[aux_var_name].coords[aux_var_dimensions].values
                    ):
                        result.append(s)
        return result

    def add_periodic_auxiliary_variable(self):
        if hasattr(self, "_auxiliary_data"):
            for aux_var_name, aux_var_dimensions in self._auxiliary_data.items():
                aux_coords = (
                    self.dataset[aux_var_name].coords[aux_var_dimensions].values
                )
                for s in aux_coords:
                    self.dataset[s] = self.dataset[aux_var_name].sel(
                        {aux_var_dimensions: s}
                    )

    def get_auxiliary_variable_names(self):
        result = {}
        if hasattr(self, "_auxiliary_data"):
            result.update(self._auxiliary_data)
        return result

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
        if time_start != first_time:
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
        x_min: optional, float
        y_max: optional, float
        y_max: optional, float

        Returns
        -------
        clipped: Package
        """
        selection = self.dataset
        if "time" in selection:
            time = selection["time"].values
            use_cftime = isinstance(time[0], cftime.datetime)
            time_start = imod.wq.timeutil.to_datetime(time_min, use_cftime)
            time_end = imod.wq.timeutil.to_datetime(time_max, use_cftime)

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

    def mask(self, domain: xr.DataArray) -> Any:
        """
        Mask values outside of domain.

        Floating values outside of the condition are set to NaN (nodata).
        Integer values outside of the condition are set to 0 (inactive in
        MODFLOW terms).

        Parameters
        ----------
        domain: xr.DataArray of bools
            The condition. Preserve values where True, discard where False.

        Returns
        -------
        masked: Package
            The package with part masked.
        """
        masked = {}
        for var, da in self.dataset.data_vars.items():
            if set(domain.dims).issubset(da.dims):
                # Check if this should be: np.issubdtype(da.dtype, np.floating)
                if issubclass(da.dtype, numbers.Real):
                    masked[var] = da.where(domain, other=np.nan)
                elif issubclass(da.dtype, numbers.Integral):
                    masked[var] = da.where(domain, other=0)
                else:
                    raise TypeError(
                        f"Expected dtype float or integer. Received instead: {da.dtype}"
                    )
            else:
                masked[var] = da

        return type(self)(**masked)
