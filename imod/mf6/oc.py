import collections

import numpy as np

from imod.mf6.pkgbase import Package


class OutputControl(Package):
    """
    The Output Control Option determines how and when heads are printed to the
    listing file and/or written to a separate binary output file.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=47

    Parameters
    ----------
    save_head : bool, or xr.DataArray of bools
        Bool per stress period.
    save_budget : bool, or xr.DataArray of bools
        Bool per stress period.
    """

    __slots__ = ("save_head", "save_budget")
    _pkg_id = "oc"
    _template = Package._initialize_template(_pkg_id)

    def __init__(self, save_head, save_budget):
        super(__class__, self).__init__()
        self["save_head"] = save_head
        self["save_budget"] = save_budget

    def render(self, directory, pkgname, globaltimes):
        d = {}
        modelname = directory.stem
        if self["save_head"].values.any():
            d["headfile"] = (directory / f"{modelname}.hds").as_posix()
        if self["save_budget"].any():
            d["budgetfile"] = (directory / f"{modelname}.cbb").as_posix()

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
