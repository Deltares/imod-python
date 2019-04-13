import jinja2
import numpy as np
import pandas as pd
import xarray as xr

from imod.io import util


class Package(xr.Dataset):
    def _replace_keyword(self, d, key):
        keyword = d[key][()]  # Get value from 0d np.array
        value = self._keywords[key][keyword]
        d[key] = value
    
    def _render(self):
        d = {k: v.values for k, v in self.data_vars.items()}
        if hasattr(self, "_keywords"):
            for key in self._keywords.keys():
                self._replace_keyword(d, key)
        return self._template.format(**d)

    def _compose_values_layer(self, key, directory, d={}, da=None):
        values = {}
        if da is None:
            da = self[key]
        d.update({
            "directory": directory,
            "name": key,
            "extension": ".idf",
        })

        # Scalar value
        if "y" not in da.coords and "x" not in da.coords:
            idf = False
        else:
            idf = True

        if "layer" not in da.coords:
            if idf:
                values["?"] = util.compose(d)
            else:
                values["?"] = da.values[()]

        else:
            for layer in np.atleast_1d(da.coords["layer"].values):
                if idf:
                    d["layer"] = layer
                    values[layer] = util.compose(d)
                else:
                    values[layer] = da.sel(layer=layer).values[()]

        return values


class BoundaryCondition(Package):
    _template = jinja2.Template(
    "    {%- for name, dictname in mapping -%}"
    "        {%- for time, timedict in dicts[dictname].items() -%}"
    "            {%- for layer, value in timedict.items() %}\n"
    "    {{name}}_p{{time}}_s{{system_index}}_l{{layer}} = {{value}}\n"
    "            {%- endfor -%}\n"
    "        {%- endfor -%}"
    "    {%- endfor -%}"
    )

    def _compose_values_timelayer(self, key, globaltimes, directory, da=None):
        values = {}

        if da is None:
            da = self[key]

        if "time" in da.coords:
            # TODO: get working for cftime
            package_times = [
                pd.to_datetime(t) for t in np.atleast_1d(da.coords["time"].values)
            ]

        d = {}
        for timestep, globaltime in enumerate(globaltimes):
            if "time" in da.coords:
                # forward fill
                # TODO: do smart forward fill using the colon notation
                time = list(filter(lambda t: t <= globaltime, package_times))[-1]
                d["time"] = time
                # Offset 0 counting in Python, add one
                values[timestep + 1] = self._compose_values_layer(key, directory, d)
            else:
                values["?"] = self._compose_values_layer(key, directory)

        return values


    def _render(self, directory, globaltimes, system_index):
        d = {}
        d["mapping"] = self._mapping
        d["system_index"] = system_index
        dicts = {}

        for varname in self.data_vars.keys():
            dicts[varname] = self._compose_values_timelayer(varname, globaltimes, directory)

        d["dicts"] = dicts

        return self._template.render(d)
