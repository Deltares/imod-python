from imod.flow.pkgbase import BoundaryCondition, Vividict
import imod
import numpy as np
from imod.wq import timeutil


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
    """

    _pkg_id = "wel"
    _variable_order = ["rate"]

    def __init__(
        self,
        id_name=None,
        x=None,
        y=None,
        rate=None,
        layer=None,
        time=None,
    ):
        super(__class__, self).__init__()
        variables = {
            "id_name": id_name,
            "x": x,
            "y": y,
            "rate": rate,
            "layer": layer,
            "time": time,
        }
        variables = {k: np.atleast_1d(v) for k, v in variables.items() if v is not None}
        length = max(map(len, variables.values()))
        index = np.arange(1, length + 1)
        self.dataset["index"] = index

        for k, v in variables.items():
            if v.size == index.size:
                self.dataset[k] = ("index", v)
            elif v.size == 1:
                self.dataset[k] = ("index", np.full(length, v))
            else:
                raise ValueError(f"Length of {k} does not match other arguments")

    def _compose_values_layer(self, varname, directory, time=None):
        values = {}
        d = {"directory": directory, "name": self._pkg_id, "extension": ".ipf"}

        if time is None:
            if "layer" in self.dataset:
                for layer in np.unique(self.dataset["layer"]):
                    layer = int(layer)
                    d["layer"] = layer
                    values[layer] = self._compose_path(d)
            else:
                values["?"] = self._compose_path(d)

        else:
            d["time"] = time
            if "layer" in self.dataset:
                # Since the well data is in long table format, it's the only
                # input that has to be inspected.
                select = np.argwhere((self["time"] == time).values)
                for layer in np.unique(self["layer"].values[select]):
                    d["layer"] = layer
                    values[layer] = self._compose_path(d)
            else:
                values["?"] = self._compose_path(d)

        return values

    def _get_runfile_times(self, globaltimes):
        self_times = np.unique(self["time"].values)
        if "timemap" in self.dataset.attrs:
            timemap_keys = np.array(list(self.dataset.attrs["timemap"].keys()))
            timemap_values = np.array(list(self.dataset.attrs["timemap"].values()))
            package_times, inds = np.unique(
                np.concatenate([self_times, timemap_keys]), return_index=True
            )
            # Times to write in the runfile
            runfile_times = np.concatenate([self_times, timemap_values])[inds]
        else:
            runfile_times = package_times = self_times

        starts_ends = timeutil.forcing_starts_ends(package_times, globaltimes)

        return runfile_times, starts_ends

    def _compose_values_timelayer(
        self,
        varname,
        globaltimes,
        directory,
        nlayer,
        values=None,
        system_index=1,
        compose_projectfile=True,
    ):
        """
        Composes paths to files, or gets the appropriate scalar value for
        a single variable in a dataset.

        Parameters
        ----------
        varname : str
            variable name of the DataArray
        globaltimes : list, np.array
            Holds the global times, i.e. the combined unique times of
            every boundary condition that are used to define the stress
            periods.
            The times of the BoundaryCondition do not have to match all
            the global times. When a globaltime is not present in the
            BoundaryCondition, the value of the first previous available time is
            filled in. The effective result is a forward fill in time.
        directory : str
            Path to working directory, where files will be written.
            Necessary to generate the paths for the runfile.
        nlayer : int
            Number of layers, unused
        values : Vividict
            Vividict (tree-like dictionary) to which values should be added
        system_index : int
            System number. Defaults as 1, but for package groups it
        compose_projectfile : bool
            Compose values in a hierarchy suitable for the projectfile

        Returns
        -------
        values : Vividict
            A nested dictionary containing following the projectfile hierarchy:
            {_pkg_id : {stress_period : {varname : {system_index : {lay_nr : value}}}}}
            or runfile hierarchy:
            {stress_period : {_pkg_id : {varname : {system_index : {lay_nr : value}}}}}
            where 'value' can be a scalar or a path to a file.
            The stress period number may be the wildcard '?'.
        """

        if values is None:
            values = Vividict()

        args = (varname, directory)
        kwargs = dict(time=None)

        if "time" in self.dataset:
            runfile_times, starts_ends = self._get_runfile_times(globaltimes)

            for time, start_end in zip(runfile_times, starts_ends):
                kwargs["time"] = time
                if compose_projectfile == True:
                    values[self._pkg_id][start_end][varname][
                        system_index
                    ] = self._compose_values_layer(*args, **kwargs)
                else:  # render runfile
                    values[start_end][self._pkg_id][varname][
                        system_index
                    ] = self._compose_values_layer(*args, **kwargs)

        else:
            if compose_projectfile == True:
                values[self._pkg_id]["steady-state"][varname][
                    system_index
                ] = self._compose_values_layer(*args, **kwargs)
            else:
                values["steady-state"][self._pkg_id][varname][
                    system_index
                ] = self._compose_values_layer(*args, **kwargs)

        return values

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
                path = self._compose_path(d)
                imod.ipf.write(path, outdf)
        else:
            outdf = df[["x", "y", "rate", "id_name"]]
            path = self._compose_path(d)
            imod.ipf.write(path, outdf)

    def save(self, directory):
        ds = self.dataset

        if "time" in ds:
            for time, timeda in ds.groupby("time"):
                timedf = timeda.to_dataframe()
                self._save_layers(timedf, directory, time=time)
        else:
            self._save_layers(ds.to_dataframe(), directory)
