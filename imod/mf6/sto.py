import numpy as np

from imod.mf6.pkgbase import Package


class Storage(Package):
    def __init__(self, specific_storage, specific_yield, transient, convertible):
        self["specific_storage"] = specific_storage
        self["specific_yield"] = specific_yield
        self["convertible"] = convertible
        self["transient"] = transient
        self._initialize_template()

    def render(self, directory, pkgname, globaltimes):
        d = {}
        for varname in ["specific_storage", "specific_yield", "convertible"]:
            d[varname] = self._compose_values(varname, directory)

        periods = {}
        if "time" in self["transient"]:
            package_times = self["transient"].coords["time"].values
            starts = np.searchsorted(globaltimes, package_times) + 1
            for i, s in enumerate(starts):
                periods[s] = self["transient"].isel(time=i).values[()]
        else:
            periods[1] = self["transient"].values[()]

        d["periods"] = periods

        return self._template.render(d)
