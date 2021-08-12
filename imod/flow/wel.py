import imod
import numpy as np
from imod.flow.pkgbase import BoundaryCondition


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
        time during which the pumping takes place. Only need to specify if
        model is transient.
    """

    _pkg_id = "wel"
    _variable_order = ["rate"]

    def __init__(
        self,
        id_name,
        x,
        y,
        rate,
        layer,
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

    def _compose_values_layer(self, varname, directory, nlayer, time=None):
        values = {}
        d = {"directory": directory, "name": directory.stem, "extension": ".ipf"}

        if time is None:
            if "layer" in self.dataset:
                for layer in np.unique(self.dataset["layer"]):
                    layer = int(layer)
                    d["layer"] = layer
                    values[layer] = self._compose_path(d)
            else:
                for layer in range(1, nlayer + 1):  # 1-based indexing
                    values[layer] = self._compose_path(d)

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
                for layer in range(1, nlayer + 1):  # 1-based indexing
                    values[layer] = self._compose_path(d)

        return values

    def _is_periodic(self):
        # Periodic stresses are defined for all variables
        return "stress_periodic" in self.dataset.attrs

    def _get_runfile_times(self, _, globaltimes, ds_times=None):
        if ds_times is None:
            ds_times = np.unique(self["time"].values)

        da = self.dataset

        runfile_times, starts = super(__class__, self)._get_runfile_times(
            da, globaltimes, ds_times=ds_times
        )

        return runfile_times, starts

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
