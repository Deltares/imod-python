import pathlib

import jinja2
import numpy as np

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
    Layer: "None" or int, optional
        layer from which the pumping takes place.
    time: "None" or listlike of np.datetime64, datetime.datetime, pd.Timestamp,
    cftime.datetime
        time during which the pumping takes place. Only need to specify if model
        is transient.
    save_budget: {True, False}, optional
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

    def __init__(self, id_name, x, y, rate, layer=None, time=None, save_budget=False):
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
        self["index"] = index

        for k, v in variables.items():
            if v.size == index.size:
                self[k] = ("index", v)
            elif v.size == 1:
                self[k] = ("index", np.full(length, v))
            else:
                raise ValueError(f"Length of {k} does not match other arguments")

        self.save_budget = save_budget

    def _max_active_n(self, varname, nlayer):
        """
        Determine the maximum active number of cells that are active
        during a stress period.

        Parameters
        ----------
        varname : str
            name of the variable to use to calculate the maximum number of
            active cells. Not used for well, here for polymorphism.
        nlayer : int
            number of layers, taken from ibound.
        """
        nmax = np.unique(self["id_name"]).size
        if not "layer" in self.coords:  # Then it applies to every layer
            nmax *= nlayer
        return nmax

    def _compose_values_layer(self, directory, time=None):
        values = {}
        d = {"directory": directory, "name": directory.stem, "extension": ".ipf"}

        if time is None:
            if "layer" in self:
                for layer in np.unique(self["layer"]):
                    layer = int(layer)
                    d["layer"] = layer
                    values[layer] = util.compose(d).as_posix()
            else:
                values["?"] = util.compose(d).as_posix()

        else:
            d["time"] = time
            if "layer" in self:
                # Since the well data is in long table format, it's the only
                # input that has to be inspected.
                select = np.argwhere((self["time"] == time).values)
                for layer in np.unique(self["layer"].values[select]):
                    d["layer"] = layer
                    values[layer] = util.compose(d).as_posix()
            else:
                values["?"] = util.compose(d).as_posix()

        return values

    def _compose_values_time(self, directory, globaltimes):
        values = {}
        if "time" in self:
            package_times = self["time"].values

            starts_ends = timeutil.forcing_starts_ends(package_times, globaltimes)
            for time, start_end in zip(package_times, starts_ends):
                values[start_end] = self._compose_values_layer(directory, time)
        else:  # for all periods
            values["?"] = self._compose_values_layer(directory)
        return values

    def _render(self, directory, globaltimes, system_index):
        d = {"system_index": system_index}
        d["wels"] = self._compose_values_time(directory, globaltimes)
        return self._template.render(d)

    def _render_ssm(self, directory, globaltimes):
        return ""

    @staticmethod
    def _save_layers(df, directory, time=None):
        d = {"directory": directory, "name": directory.stem, "extension": ".ipf"}
        d["directory"].mkdir(exist_ok=True, parents=True)

        if time is not None:
            d["time"] = time

        if "layer" in df:
            for layer, layerdf in df.groupby("layer"):
                d["layer"] = layer
                # Ensure right order
                outdf = layerdf[["x", "y", "rate", "id_name"]]
                path = util.compose(d)
                imod.ipf.write(path, outdf)
        else:
            outdf = df[["x", "y", "rate", "id_name"]]
            path = util.compose(d)
            imod.ipf.write(path, outdf)

    def save(self, directory):
        if "time" in self:
            for time, timeda in self.groupby("time"):
                timedf = timeda.to_dataframe()
                self._save_layers(timedf, directory, time=time)
        else:
            self._save_layers(self.to_dataframe(), directory)

    def _pkgcheck(self, ibound=None):
        # TODO: implement
        pass
