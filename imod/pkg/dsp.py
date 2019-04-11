from imod.pkg.pkgbase import Package

class Dispersion(Package):
    _pkg_id = "dsp"
    def __init__(
        self, longitudinal, ratio_horizontal, ratio_vertical, diffusion_coefficient
    ):
        super(__class__, self).__init__()
