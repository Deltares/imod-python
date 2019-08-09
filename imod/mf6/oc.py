import collections
import numpy as np

from imod.mf6.pkgbase import Package


class OutputControl(Package):
    """
    Attributes
    ----------

    save_head : bool, or xr.DataArray of bools
        Bool per stress period
    save_budget : bool, or xr.DataArray of bools
        Bool per stress period
    """

    _pkg_id = "oc"

    def __init__(self, save_head, save_budget):
        super(__class__, self).__init__()
        self["save_head"] = save_head
        self["save_budget"] = save_budget
        self._initialize_template()

    def render(self, directory, pkgname, globaltimes):
        d = {}

        periods = collections.defaultdict(dict)
        for datavar in self.data_vars:
            key = datavar.replace("_", " ")
            if "time" in self[datavar]:
                package_times = self[datavar].coords["time"].values
                starts = np.searchsorted(globaltimes, package_times) + 1
                for i, s in enumerate(starts):
                    if self[datavar].isel(time=i).values[()]:
                        periods[s][key] = "all"
            else:
                if self[datavar].values[()]:
                    periods[1][key] = "all"

        d["periods"] = periods

        return self._template.render(d)
