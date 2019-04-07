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
    1
    """
    _pkg_id = "adv"
    def __init__(self, courant, max_particles, tracking="euler", weighting_factor=0.5, dceps, nplane, npl, nph, npmin, npmax):
        super(__class__, self).__init__()


class AdvectionModifiedMOC(Package):
    """
    2
    """
    _pkg_id = "adv"
    def __init__(self, courant, tracking="euler", weighting_factor=0.5, interp, nlsink, npsink):
        super(__class__, self).__init__()


class AdvectionHybridMOC(Package):
    """
    3
    """
    _pkg_id = "adv"
    def __init__(self, courant, max_particles, tracking="euler", weighting_factor=0.5, dceps, nplane, npl, nph, npmin, npmax, dchmoc):
        super(__class__, self).__init__()


class AdvectionTVD(Package):
    """
    -1
    """
    _pkg_id = "adv"
    def __init__(self, courant, weighting_factor=0.5):
        super(__class__, self).__init__()
        self["courant"] = courant
        self["weighting_factor"] = weighting_factor
    