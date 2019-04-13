from imod.pkg.pkgbase import Package


class AdvectionFiniteDifference(Package):
    """
    0
    """
    _pkg_id = "adv"
    def __init__(self, courant, weighting="upstream", weighting_factor=0.5):
        super(__class__, self).__init__()
        self["courant"] = courant
        self["weighting"] = weighting
        self["weighting_factor"] = weighting_factor


class AdvectionMOC(Package):
    """
    Method of Characteristics
    1
    """
    _pkg_id = "adv"
    def __init__(self, courant, max_nparticles, tracking="euler", weighting_factor=0.5, dconcentration_epsilon=1.0e-5, nplane=2, nparticles_no_advection=10, nparticles_advection=40, cell_min_nparticles=5, cell_max_nparticles=80):
        super(__class__, self).__init__()


class AdvectionModifiedMOC(Package):
    """
    Modified Method of Characteristics
    2
    """
    _pkg_id = "adv"
    def __init__(self, courant, tracking, weighting_factor, interp, nlsink, npsink):
        super(__class__, self).__init__()


class AdvectionHybridMOC(Package):
    """
    Hybrid Method of Characteristics and Modified Method of Characteristics
    3
    """
    _pkg_id = "adv"
    def __init__(self, courant, max_particles, tracking, weighting_factor, dceps, nplane, npl, nph, npmin, npmax, dchmoc):
        super(__class__, self).__init__()


class AdvectionTVD(Package):
    """
    Total Variation Diminishing formulation, ULTIMATE
    -1
    """
    _pkg_id = "adv"

    _template = (
    "[adv]\n"
    "    mixelm = -1\n"
    "    percel = {courant}\n"
    )

    def __init__(self, courant):
        super(__class__, self).__init__()
        self["courant"] = courant
    