import pathlib

import jinja2
import numpy as np
import pandas as pd
import xarray as xr
from xarray.core.utils import either_dict_or_kwargs

import imod
from imod import util
from imod.wq import timeutil
from imod.wq.pkgbase import BoundaryCondition


class Well(BoundaryCondition):
    """
    The Well package is used to simulate a specified flux to individual cells
    and specified in units of length3/time.

    Parameters
    ----------
    id_name: str or list of str
        name of the well(s).
    x: float or list of floats
        x coordinate of the well(s).
    y: float or list of floats
        y coordinate of the well(s).
    rate: float or list of floats.
        pumping rate in the well(s).
    layer: "None" or int, optional
        layer from which the pumping takes place.
    time: "None" or listlike of np.datetime64, datetime.datetime, pd.Timestamp,
    cftime.datetime
        time during which the pumping takes place. Only need to specify if model
        is transient.
    save_budget: bool, optional
        is a flag indicating if the budget should be saved (IRIVCB).
        Default is False.
    """

    __slots__ = ("save_budget",)
    _pkg_id = "wel"

    _template = jinja2.Template(
        "    {%- for time, timedict in wels.items() -%}"
        "        {%- for layer, value in timedict.items() %}\n"
        "    wel_p{{time}}_s{{system_index}}_l{{layer}} = {{value}}"
        "        {%- endfor -%}\n"
        "    {%- endfor -%}"
    )

    def __init__(
        self,
        id_name,
        x,
        y,
        rate,
        layer=None,
        time=None,
        concentration=None,
        save_budget=False,
    ):
        super(__class__, self).__init__()

        if (
            isinstance(concentration, xr.DataArray)
            and "species" in concentration.coords
        ):
            self["species"] = concentration["species"].values
        elif isinstance(concentration, np.ndarray) and len(concentration.shape) == 2:
            self["species"] = np.arange(1, concentration.shape[0] + 1)

        variables = {
            "id_name": id_name,
            "x": x,
            "y": y,
            "rate": rate,
            "layer": layer,
            "time": time,
            "concentration": concentration,
        }
        variables = {k: np.atleast_1d(v) for k, v in variables.items() if v is not None}
        length = max([max(v.shape) for v in variables.values()])
        index = np.arange(1, length + 1)
        self["index"] = index

        for k, v in variables.items():
            if (
                k == "concentration"
                and (len(v.shape) == 2)
                and (v.shape[1] == index.size)
            ):
                self[k] = (["species", "index"], v)
            elif v.size == index.size:
                self[k] = ("index", v)
            elif v.size == 1:
                self[k] = ("index", np.full(length, v))
            else:
                raise ValueError(f"Length of {k} does not match other arguments")

        self["save_budget"] = save_budget

    def _max_active_n(self, varname, nlayer, nrow, ncol):
        """
        Determine the maximum active number of cells that are active
        during a stress period.

        Parameters
        ----------
        varname : str
            name of the variable to use to calculate the maximum number of
            active cells. Not used for well, here for polymorphism.
        nlayer, nrow, ncol : int
        """
        nmax = np.unique(self["id_name"]).size
        if not "layer" in self.coords:  # Then it applies to every layer
            nmax *= nlayer
        self._cellcount = nmax
        self._ssm_cellcount = nmax
        return nmax

    def _compose_values_layer(self, directory, nlayer, name, time=None, compress=True):
        values = {}
        d = {"directory": directory, "name": name, "extension": ".ipf"}

        if time is None:
            if "layer" in self:
                for layer in np.unique(self["layer"]):
                    layer = int(layer)
                    d["layer"] = layer
                    values[layer] = self._compose(d)
            else:
                values["?"] = self._compose(d)

        else:
            d["time"] = time
            if "layer" in self:
                # Since the well data is in long table format, it's the only
                # input that has to be inspected.
                select = np.argwhere((self["time"] == time).values)
                for layer in np.unique(self["layer"].values[select]):
                    d["layer"] = layer
                    values[layer] = self._compose(d)
            else:
                values["?"] = self._compose(d)

        if "layer" in self:
            # Compose does not accept non-integers, so use 0, then replace
            d["layer"] = 0
            if np.unique(self["layer"].values).size == nlayer:
                token_path = imod.util.compose(d).as_posix()
                token_path = token_path.replace("_l0", "_l$")
                values = {"$": token_path}
            elif compress:
                range_path = imod.util.compose(d).as_posix()
                range_path = range_path.replace("_l0", "_l:")
                # TODO: temporarily disable until imod-wq is fixed
                values = self._compress_idflayers(values, range_path)

        return values

    def _compose_values_time(self, directory, name, globaltimes, nlayer):
        # TODO: rename to _compose_values_timelayer?
        values = {}
        if "time" in self:
            self_times = np.unique(self["time"].values)
            if "timemap" in self.attrs:
                timemap_keys = np.array(list(self.attrs["timemap"].keys()))
                timemap_values = np.array(list(self.attrs["timemap"].values()))
                package_times, inds = np.unique(
                    np.concatenate([self_times, timemap_keys]), return_index=True
                )
                # Times to write in the runfile
                runfile_times = np.concatenate([self_times, timemap_values])[inds]
            else:
                runfile_times = package_times = self_times

            starts_ends = timeutil.forcing_starts_ends(package_times, globaltimes)

            for time, start_end in zip(runfile_times, starts_ends):
                # Check whether any range occurs in the input.
                # If does does, compress should be False
                compress = not (":" in start_end)
                values[start_end] = self._compose_values_layer(
                    directory, nlayer=nlayer, name=name, time=time, compress=compress
                )
        else:  # for all periods
            values["?"] = self._compose_values_layer(
                directory, nlayer=nlayer, name=name
            )
        return values

    def _render(self, directory, globaltimes, system_index, nlayer):
        d = {"system_index": system_index}
        name = directory.stem
        d["wels"] = self._compose_values_time(directory, name, globaltimes, nlayer)
        return self._template.render(d)

    def _render_ssm(self, directory, globaltimes, nlayer):
        if "concentration" in self.data_vars:
            d = {"pkg_id": self._pkg_id}
            name = f"{directory.stem}-concentration"
            if "species" in self["concentration"].dims:
                concentration = {}
                for species in self["concentration"]["species"].values:
                    concentration[species] = self._compose_values_time(
                        directory, f"{name}_c{species}", globaltimes, nlayer=nlayer
                    )
            else:
                concentration = {
                    1: self._compose_values_time(
                        directory, name, globaltimes, nlayer=nlayer
                    )
                }
            d["concentration"] = concentration
            return self._ssm_template.render(d)
        else:
            return ""

    def _save_layers(self, df, directory, time=None):
        d = {"directory": directory, "name": directory.stem, "extension": ".ipf"}
        d["directory"].mkdir(exist_ok=True, parents=True)

        if time is not None:
            d["time"] = time

        if "layer" in df:
            for layer, layerdf in df.groupby("layer"):
                d["layer"] = layer
                # Ensure right order
                outdf = layerdf[["x", "y", "rate", "id_name"]]
                path = self._compose(d)
                imod.ipf.write(path, outdf)
        else:
            outdf = df[["x", "y", "rate", "id_name"]]
            path = self._compose(d)
            imod.ipf.write(path, outdf)

    @staticmethod
    def _save_layers_concentration(df, directory, name, time=None):
        d = {"directory": directory, "name": name, "extension": ".ipf"}
        d["directory"].mkdir(exist_ok=True, parents=True)

        if time is not None:
            d["time"] = time

        if "layer" in df:
            for layer, layerdf in df.groupby("layer"):
                d["layer"] = layer
                # Ensure right order
                outdf = layerdf[["x", "y", "concentration", "id_name"]]
                path = util.compose(d)
                imod.ipf.write(path, outdf)
        else:
            outdf = df[["x", "y", "concentration", "id_name"]]
            path = util.compose(d)
            imod.ipf.write(path, outdf)

    @staticmethod
    def _save_layers_concentration(df, directory, name, time=None):
        d = {"directory": directory, "name": name, "extension": ".ipf"}
        d["directory"].mkdir(exist_ok=True, parents=True)

        if time is not None:
            d["time"] = time

        if "layer" in df:
            for layer, layerdf in df.groupby("layer"):
                d["layer"] = layer
                # Ensure right order
                outdf = layerdf[["x", "y", "concentration", "id_name"]]
                path = util.compose(d)
                imod.ipf.write(path, outdf)
        else:
            outdf = df[["x", "y", "concentration", "id_name"]]
            path = util.compose(d)
            imod.ipf.write(path, outdf)

    def save(self, directory):
        all_species = [None]
        if "concentration" in self.data_vars:
            if "species" in self["concentration"].dims:
                all_species = self["concentration"]["species"].values

        # Loop over species if applicable
        for species in all_species:
            if species is not None:
                ds = xr.Dataset.sel(self, species=species)
            else:
                ds = self

            if "time" in ds:
                for time, timeda in ds.groupby("time"):
                    timedf = timeda.to_dataframe()
                    ds._save_layers(timedf, directory, time=time)
                    if "concentration" in ds.data_vars:
                        name = f"{directory.stem}-concentration"
                        if species is not None:
                            name = f"{name}_c{species}"
                        ds._save_layers_concentration(
                            timedf, directory, name, time=time
                        )
            else:
                ds._save_layers(ds.to_dataframe(), directory)
                if "concentration" in ds.data_vars:
                    name = f"{directory.stem}-concentration"
                    if species is not None:
                        name = f"{name}_c{species}"
                    ds._save_layers_concentration(timedf, directory, name)

    def _pkgcheck(self, ibound=None):
        # TODO: implement
        pass

    def add_timemap(self, timemap, use_cftime=False):
        # To ensure consistency, it isn't possible to use differing timemaps
        # between rate and concentration: the number of points might change
        # between stress periods, and isn't especially easy to check.
        if "time" not in self:
            raise ValueError(
                f"This Wel package does not have time, cannot add timemap."
            )
        # Replace both key and value by the right datetime type
        d = {
            timeutil.to_datetime(k, use_cftime): timeutil.to_datetime(v, use_cftime)
            for k, v in timemap.items()
        }
        self.attrs["timemap"] = d

    def to_sparse_dataset(self):
        """
        Converts Well to a xr.Dataset with sparse variables with coordinates:

        * ``id_name``
        * ``y``
        * ``x``
        * ``time`` (optional)
        * ``layer`` (optional)
        * ``species`` (optional)

        The optional coordinates will only be added if they have been set in
        the Well.

        Symmetrical with ``.from_sparse_dataset()``.

        Returns
        -------
        sparse_dataset: xr.Dataset
        """

        def to_sparse(ds, index_vars):
            tmp_df = (
                ds.drop_vars(["save_budget"])
                .to_dataframe()
                .reset_index()
                .set_index([i for i in index_vars if i in ds])
            )
            return xr.Dataset.from_dataframe(tmp_df.drop("index", 1), sparse=True)

        index_vars = ["id_name", "y", "x", "layer", "time"]
        if "concentration" in self and "species" in self["concentration"].dims:
            conc_ds = self.drop_vars(["rate"])
            rate_ds = self.drop_vars(["concentration", "species"])
            sparse_ds = to_sparse(conc_ds, ["species"] + index_vars)
            sparse_ds["rate"] = to_sparse(rate_ds, index_vars)["rate"]
        else:
            sparse_ds = to_sparse(self, index_vars)

        sparse_ds["save_budget"] = self["save_budget"]
        return sparse_ds

    @staticmethod
    def _check_sparse_concentration(ds: xr.Dataset):
        has_species = "species" in ds["concentration"].dims
        if has_species:
            if not ds["concentration"].dims[0] == "species":
                raise ValueError("species must be first dimension for concentration")
            sample = ds["concentration"].isel(species=0, drop=True)
        else:
            sample = ds["concentration"]

        # Check that rate and concentration COO coordinates match up
        rate_coords = {k: v for k, v in zip(ds["rate"].indexes, ds["rate"].data.coords)}
        conc_coords = {k: v for k, v in zip(sample.indexes, sample.data.coords)}
        for dim in rate_coords:
            if not np.array_equal(rate_coords[dim], conc_coords[dim]):
                raise ValueError(
                    "Sparse COO coords do not match between rate and concentration"
                    f" for dimension {dim}"
                )

        if has_species:
            return xr.DataArray(
                data=ds["concentration"].data.data.reshape(len(ds["species"]), -1),
                coords={"species": ds["species"]},
                dims=["species", "index"],
            )
        else:
            return ds["concentration"].data.data

    @staticmethod
    def from_sparse_dataset(ds: xr.Dataset) -> "Well":
        """
        Create a Well object from an xarray Dataset with sparse data variables.

        Symmetrical with ``.to_sparse_dataset()``.

        Returns
        -------
        well: imod.wq.Well
        """
        # Rate variable will always have all dimensions, except species
        # Add the scalar coordinates coordinates
        well_kwargs = {}
        # Add the scalar coordinates
        for k, v in ds["rate"].coords.items():
            if k not in ds["rate"].indexes:
                well_kwargs[k] = v.values
        # Add the array coordinates
        for (k, v), idx in zip(ds["rate"].indexes.items(), ds["rate"].data.coords):
            well_kwargs[k] = v[idx]
        # Add the rate data variable
        well_kwargs["rate"] = ds["rate"].data.data
        # Now deal with concentration
        if "concentration" in ds:
            well_kwargs["concentration"] = Well._check_sparse_concentration(ds)

        well_kwargs["save_budget"] = bool(ds["save_budget"])
        well_kwargs.pop("species", None)  # Species might have ended up in the selection
        return Well(**well_kwargs)

    def sel(
        self,
        indexers=None,
        method=None,
        tolerance=None,
        drop=False,
        **indexers_kwargs,
    ) -> "Well":
        """Returns a new Well package with each array indexed by tick labels
        along the specified dimension(s).

        Indexers for this method should use labels instead of integers.

        The Well.sel method is a special implementation of Package.sel method,
        that allows selecting on coords in the Well data, not just on its
        dimensions. Well times are selected within separate wells, so that
        active times are selected per well. E.g., if a well becomes active at
        time a, and still active at a later time b, a is returned when time b is
        selected for. Time a in the returned result is adjusted to b.

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values given by scalars,
            slices or arrays of tick labels. For dimensions with multi-index,
            the indexer may also be a dict-like object with keys matching index
            level names. If DataArrays are passed as indexers, xarray-style
            indexing will be carried out.
            Dimensions not present in Well are ignored.
            One of indexers or indexers_kwargs must be provided.
        method : {None, 'nearest', 'pad'/'ffill', 'backfill'/'bfill'}, optional
            Method to use for inexact matches:

            * None (default): only exact matches
            * pad / ffill: propagate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.
        **indexers_kwarg : {dim: indexer, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        Returns
        -------
        obj : Well
            A new Well with the same contents as this package, except each
            variable and dimension is indexed by the appropriate indexers.
            If indexer DataArrays have coordinates that do not conflict with
            this object, then these coordinates will be attached.

        See Also
        --------
        Package.sel
        xarray.Dataset.sel
        """
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "sel")
        indexers = {k: v for k, v in indexers.items() if k in self}

        if len(indexers) == 0:  # Nothing to be done.
            return self
        else:
            # Create a temporary dataframe and temporary sparse dataset
            # use this sparse dataset to do the selection
            sparse_ds = self.to_sparse_dataset()
            time_indexer = indexers.pop("time", None)
            if len(indexers) == 0:
                selection = sparse_ds
            else:
                selection = xr.Dataset.sel(
                    sparse_ds, indexers, method=method, tolerance=tolerance, drop=False
                )

            if time_indexer:
                selection = self._sel_time(selection, time_indexer)

            return self.from_sparse_dataset(selection)
