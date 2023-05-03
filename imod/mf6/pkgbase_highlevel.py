"""
Module to store (prototype) low-level modflow6 package base classes.

These are close to the NetCDF data model, with x, y coordinates.

We plan to split up the present attributes of the classes in pkgbase.py
(Package and BoundaryCondition) into low-level and high-level classes.
"""

import abc
import numbers
import pathlib
from collections import defaultdict
from typing import Any, Dict, List

import cftime
import numpy as np
import xarray as xr
import xugrid as xu

import imod
from imod.mf6.pkgbase_shared import NewPackageBase
from imod.mf6.validation import validation_pkg_error_message
from imod.schemata import ValidationError


class HighLevelPackageBase(NewPackageBase, abc.ABC):
    def __init__(self, allargs=None):
        super().__init__(allargs)

    @classmethod
    def from_file(cls, path, **kwargs):
        """
        Loads an imod mf6 package from a file (currently only netcdf and zarr are supported).
        Note that it is expected that this file was saved with imod.mf6.Package.dataset.to_netcdf(),
        as the checks upon package initialization are not done again!

        Parameters
        ----------
        path : str, pathlib.Path
            Path to the file.
        **kwargs : keyword arguments
            Arbitrary keyword arguments forwarded to ``xarray.open_dataset()``, or
            ``xarray.open_zarr()``.
        Refer to the examples.

        Returns
        -------
        package : imod.mf6.Package
            Returns a package with data loaded from file.

        Examples
        --------

        To load a package from a file, e.g. a River package:

        >>> river = imod.mf6.River.from_file("river.nc")

        For large datasets, you likely want to process it in chunks. You can
        forward keyword arguments to ``xarray.open_dataset()`` or
        ``xarray.open_zarr()``:

        >>> river = imod.mf6.River.from_file("river.nc", chunks={"time": 1})

        Refer to the xarray documentation for the possible keyword arguments.
        """
        path = pathlib.Path(path)
        if path.suffix in (".zip", ".zarr"):
            # TODO: seems like a bug? Remove str() call if fixed in xarray/zarr
            dataset = xr.open_zarr(str(path), **kwargs)
        else:
            dataset = xr.open_dataset(path, **kwargs)

        if dataset.ugrid_roles.topology:
            dataset = xu.UgridDataset(dataset)
            dataset = imod.util.from_mdal_compliant_ugrid2d(dataset)

        # Replace NaNs by None
        for key, value in dataset.items():
            stripped_value = value.values[()]
            if isinstance(stripped_value, numbers.Real) and np.isnan(stripped_value):
                dataset[key] = None

        instance = cls.__new__(cls)
        instance.dataset = dataset
        return instance


class HighLevelPackage(HighLevelPackageBase, abc.ABC):
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
    ) -> "HighLevelPackage":
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


class TimeDependentPackage(HighLevelPackage):
    def set_repeat_stress(self, times) -> None:
        """
        Set repeat stresses: re-use data of earlier periods.

        Parameters
        ----------
        times: Dict of datetime-like to datetime-like.
            The data of the value datetime is used for the key datetime.
        """
        keys = [
            imod.wq.timeutil.to_datetime(key, use_cftime=False) for key in times.keys()
        ]
        values = [
            imod.wq.timeutil.to_datetime(value, use_cftime=False)
            for value in times.values()
        ]
        self.dataset["repeat_stress"] = xr.DataArray(
            data=np.column_stack((keys, values)),
            dims=("repeat", "repeat_items"),
        )

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
