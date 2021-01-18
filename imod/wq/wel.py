import pathlib

import jinja2
import numpy as np
import pandas as pd
import xarray as xr

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

    # TODO: implement well to concentration IDF and use ssm_template
    # Ignored for now, since wells are nearly always extracting

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
        length = max(map(len, variables.values()))
        index = np.arange(1, length + 1)
        self["index"] = index

        for k, v in variables.items():
            if v.size == index.size:
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
            if "species" in self["concentration"].coords:
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
            if "species" in self["concentration"].coords:
                all_species = self["concentration"]["species"].values

        # Loop over species if applicable
        for species in all_species:
            if species is not None:
                ds = self.sel(species=species)
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

    def _df_to_well(self, df):
        # to handle concentration with species when casting back to Well
        conc = None
        if "species" in df.index.names:
            conc = df["concentration"]
            df = df.drop(columns="concentration")
            df = df.droplevel("species").drop_duplicates()
        elif "species" in df:
            conc = df.set_index([df.index, df["species"]])["concentration"]
            df = df.drop(columns=["concentration", "species"])
        elif "concentration" in df:
            conc = df["concentration"]
            df = df.drop(columns="concentration")

        w = type(self)(
            save_budget=self["save_budget"], **df
        )  # recast df to Well package
        if conc is not None:
            w["concentration"] = conc.to_xarray().transpose()
        return w

    def _sel_time(self, time_sel):
        # some foo to select last previous time for indexed times
        if "time" in self:
            df = self.to_dataframe().drop(columns="save_budget")
            if "layer" in df:
                grouped = df.groupby(["id_name", "x", "y", "layer"])
            else:
                grouped = df.groupby(["id_name", "x", "y"])

            if isinstance(time_sel, slice):
                if time_sel.start is None:
                    time_sel = slice(df["time"].min(), time_sel.stop, time_sel.step)
                if time_sel.stop is None:
                    time_sel = slice(time_sel.start, df["time"].max(), time_sel.step)

                # set start time per group to last previous time (concurrent stress)
                def _set_times(g):
                    il = (
                        g.searchsorted(
                            pd.Timestamp(time_sel.start) + pd.to_timedelta("1ns")
                        )
                        - 1
                    )
                    if il >= 0:
                        g.iloc[il] = pd.Timestamp(time_sel.start)
                    return g

                df["time"] = grouped["time"].transform(_set_times)

                # and select for time slice
                sel = df.loc[
                    (df["time"] >= time_sel.start) & (df["time"] <= time_sel.stop)
                ]
            else:
                # (list of) individual dates
                time_sel = timeutil.array_to_datetime(
                    np.atleast_1d(time_sel), False
                )  # TODO: Well and cftime?

                # set requested times per group to last previous time (concurrent stress)
                def _set_times(g):
                    g.iloc[
                        g.searchsorted(time_sel + pd.to_timedelta("1ns")) - 1
                    ] = time_sel
                    return g

                df["time"] = grouped["time"].transform(_set_times)

                # and select for times
                sel = df.loc[df["time"].isin(time_sel)]

            # back to Well package
            return self._df_to_well(sel)
        else:
            return self

    def sel(self, **dimensions):
        """Returns a new Well package with each array indexed by tick labels
        along the specified dimension(s).

        Indexers for this method should use labels instead of integers.

        The Well.sel method is a special implementation of Package.sel method, that
        allows selecting on coords in the Well data, not just on its dimensions. Well
        times are selected within separate wells, so that active times are selected
        per well. E.g., if a well becomes active at time a, and still active at a
        later time b, a is returned when time b is selected for. Time a in the returned
        result is adjusted to b.


        Parameters
        ----------
        **dimensions : {dim: indexer, ...}
            Keyword arguments with keys matching dimensions and values given
            by scalars, slices or arrays of tick labels. For dimensions with
            multi-index, the indexer may also be a dict-like object with keys
            matching index level names.
            If DataArrays are passed as indexers, xarray-style indexing will be
            carried out. See :ref:`indexing` for the details.
            Dimensions not present in Package are ignored.

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
        drop : bool, optional
            If ``drop=True``, drop coordinates variables in `indexers` instead
            of making them scalar.

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
        # filter out possible keyword arguments method tolerance and drop
        method = dimensions.pop("method", None)
        tolerance = dimensions.pop("tolerance", None)
        drop = dimensions.pop("drop", False)
        # account for time separately
        time_sel = dimensions.pop("time", None)

        # first: selection on dimensions
        sel_dims = {k: v for k, v in dimensions.items() if k in self.dims}
        if len(sel_dims):
            for k, v in sel_dims.items():
                v = dimensions.pop(k)
                return xr.Dataset.sel(
                    self, {k: v}, method=method, tolerance=tolerance, drop=drop
                ).sel(**dimensions)

        # then: selection on dataframe
        sel_dims = {k: v for k, v in dimensions.items() if k in self}
        if len(sel_dims) == 0:
            sel = self
        else:
            df = self.to_dataframe().drop(columns="save_budget")
            b = np.ones(len(df), dtype=bool)
            for k, v in sel_dims.items():
                try:
                    if isinstance(v, slice):
                        # slice?
                        if v.start is None:
                            v = slice(df[k].min(), v.stop, v.step)
                        if v.stop is None:
                            v = slice(v.start, df[k].min(), v.step)
                        # to account for reversed order of y
                        low, high = min(v.start, v.stop), max(v.start, v.stop)
                        b = b & (df[k] >= low) & (df[k] <= high)
                    else:
                        v = np.atleast_1d(v)
                        # boolean indexer
                        if v.dtype == bool:
                            b = b & v
                        else:
                            # list, labels etc
                            b = b & df[k].isin(v)
                except:
                    raise ValueError(
                        "Invalid indexer for Well package, accepts slice or (list-like of) values"
                    )
            sel = df.loc[b]
            sel = self._df_to_well(sel)
        if time_sel is not None:
            return sel._sel_time(time_sel)
        else:
            return sel
