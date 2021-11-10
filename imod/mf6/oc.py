import collections

import numpy as np

from imod.mf6.pkgbase import Package


class OutputControl(Package):
    """
    The Output Control Option determines how and when heads are printed to the
    listing file and/or written to a separate binary output file.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=47

    Currently the settings "first", "last", "all", and "frequency"
    are supported, the "steps" setting is not supported, because of
    its ragged nature. Furthermore, only one setting per stress period
    can be specified in imod-python.

    Parameters
    ----------
    save_head : {string, integer}, or xr.DataArray of {string, integer}, optional
        String or integer indicating output control for head file (.hds)
        If string, should be one of ["first", "last", "all"].
        If integer, interpreted as frequency.
    save_budget : {string, integer}, or xr.DataArray of {string, integer}, optional
        String or integer indicating output control for cell budgets (.cbc)
        If string, should be one of ["first", "last", "all"].
        If integer, interpreted as frequency.

    Examples
    --------
    To specify a mix of both 'frequency' and 'first' setting,
    we need to specify an array with both integers and strings.
    For this we need to create a numpy object array first,
    otherwise xarray converts all to strings automatically.

    >>> time = [np.datetime64("2000-01-01"), np.datetime64("2000-01-02")]
    >>> data = np.array(["last", 5], dtype="object")
    >>> save_head = xr.DataArray(data, coords={"time": time}, dims=("time"))
    >>> oc = imod.mf6.OutputControl(save_head=save_head, save_budget=None)

    """

    __slots__ = ("save_head", "save_budget")
    _pkg_id = "oc"
    _keyword_map = {}
    _template = Package._initialize_template(_pkg_id)

    def __init__(self, save_head=None, save_budget=None):
        super().__init__()
        self["save_head"] = save_head
        self["save_budget"] = save_budget

    def _get_ocsetting(self, setting):
        """Get oc setting based on its type. If integers return f'frequency {setting}', if"""
        if isinstance(setting, (int, np.integer)) and not isinstance(setting, bool):
            return f"frequency {setting}"
        elif isinstance(setting, str):
            if setting.lower() in ["first", "last", "all"]:
                return setting.lower()
            else:
                raise ValueError(
                    f"Output Control received wrong string. String should be one of ['first', 'last', 'all'], instead got {setting}"
                )
        else:
            raise TypeError(
                f"Output Control setting should be either integer or string in ['first', 'last', 'all'], instead got {setting}"
            )

    def render(self, directory, pkgname, globaltimes):
        d = {}
        modelname = directory.stem
        if self["save_head"].values[()] is not None:
            d["headfile"] = (directory / f"{modelname}.hds").as_posix()
        if self["save_budget"].values[()] is not None:
            d["budgetfile"] = (directory / f"{modelname}.cbc").as_posix()

        periods = collections.defaultdict(dict)
        for datavar in self.data_vars:
            if self[datavar].values[()] is None:
                continue
            key = datavar.replace("_", " ")
            if "time" in self[datavar].coords:
                package_times = self[datavar].coords["time"].values
                starts = np.searchsorted(globaltimes, package_times) + 1
                for i, s in enumerate(starts):
                    setting = self[datavar].isel(time=i).item()
                    periods[s][key] = self._get_ocsetting(setting)

            else:
                setting = self[datavar].item()
                periods[1][key] = self._get_ocsetting(setting)

        d["periods"] = periods

        return self._template.render(d)
