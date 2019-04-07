import numpy as np
import pandas as pd
import xarray as xr

from imod import util


class Package(xr.Dataset):
    def _replace_keyword(self, d, key):
        keyword = d[key][()]  # Get value from 0d np.array
        value = self._keywords[key][keyword]
        d[key] = value
    
    def _render(self):
        d = {k, v.values for k, v in self.data_vars.items()}
        return self._template.format(d)

    def _compose_values_layer(self, key, directory, d={}):
        values = {}
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
    def _compose_values_timelayer(self, key, globaltimes, directory):
        values = {}
        da = self[key]

        if "time" in da.coords:
            # TODO: get working for cftime
            package_times = [
                pd.to_datetime(t) for t in np.atleast_1d(da.coords["time"].values)
            ]

        d = {}
        for globaltime in globaltimes:
            if "time" in da.coords:
                # forward fill
                time = list(filter(lambda t: t <= globaltime, package_times))[-1]
                d["time"] = time
                values[time] = self._compose_values_layer(key, directory, d)
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
