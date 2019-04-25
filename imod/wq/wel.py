import pathlib

import jinja2
import pandas as pd
import imod
from imod import util
from imod.wq.pkgbase import Package
from imod.wq import timeutil


class Well(pd.DataFrame):
    _pkg_id = "wel"
    save_budget = False

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
        self["x"] = x
        self["y"] = y
        self["rate"] = rate
        self["id_name"] = id_name
        if layer is not None:
            self["layer"] = layer
        if time is not None:
            self["time"] = time
        self.save_budget = save_budget

    @property
    def data_vars(self):
        return self.columns

    @property
    def coords(self):
        return self.columns

    def _compose_values_layer(self, directory, time=None):
        values = {}
        d = {"directory": directory, "name": directory.stem, "extension": ".ipf"}
        if time is not None:
            d["time"] = time
        if "layer" in self:
            for layer in pd.unique(self["layer"]):
                layer = int(layer)
                values[layer] = util.compose(d)
        else:
            values["?"] = util.compose(d)
        return values

    def _compose_values_time(self, directory, globaltimes):
        values = {}
        if "time" in self:
            package_times = timeutil.to_datetime(self["time"].values)
            globaltimes = timeutil.to_datetime(globaltimes)
            starts_ends = timeutil.forcing_starts_ends(package_times, globaltimes)

            for itime, (time, start_end) in enumerate(zip(package_times, starts_ends)):
                values[start_end] = self._compose_values_layer(directory, time)
        else:
            values["?"] = self._compose_values_layer(directory)
        return values

    def _render(self, directory, globaltimes, system_index):
        d = {"system_index": system_index}
        d["wels"] = self._compose_values_time(directory, globaltimes)
        return self._template.render(d)

    def _render_ssm(self, directory, globaltimes):
        return ""

    def _max_active_n(self, varname, nlayer):
        if "time" in self:
            nmax = self.groupby("time").size().max()
        else:
            nmax = self.shape[0]
        if "layer" not in self:
            nmax *= nlayer
        return nmax

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
            for time, timedf in self.groupby("time"):
                self._save_layers(timedf, directory, time=time)
        else:
            self._save_layers(self, directory)
