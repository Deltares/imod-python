from imod.pkg.pkgbase import Package

class Well(Package):
    _pkg_id = "wel"
    def __init(self, rate, conc):
        super(__class__, self).__init__()
        # ds = df.set_index(["time", "y", "x"]).to_xarray() does work