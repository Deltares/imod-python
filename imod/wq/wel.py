from imod.wq.pkgbase import Package


class Well(Package):
    _pkg_id = "wel"

    def __init(self, rate, concentration=0):
        super(__class__, self).__init__()
        # ds = df.set_index(["time", "y", "x"]).to_xarray() does work
