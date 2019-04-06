import xarray as xr

class AdvectionFiniteDifference(xr.Dataset):
    """
    0
    """
    def __init__(self, courant, weighting="upstream", weighting_factor=0.5):
        super(__class__, self).__init__()


class AdvectionMOC(xr.Dataset):
    """
    1
    """
    def __init__(self, courant, max_particles, tracking="euler", weighting_factor=0.5, dceps, nplane, npl, nph, npmin, npmax):
        super(__class__, self).__init__()


class AdvectionModifiedMOC(xr.Dataset):
    """
    2
    """
    def __init__(self, courant, tracking="euler", weighting_factor=0.5, interp, nlsink, npsink):
        super(__class__, self).__init__()


class AdvectionHybridMOC(xr.Dataset):
    """
    3
    """
    def __init__(self, courant, max_particles, tracking="euler", weighting_factor=0.5, dceps, nplane, npl, nph, npmin, npmax, dchmoc):
        super(__class__, self).__init__()


class AdvectionTVD(xr.Dataset):
    """
    -1
    """
    def __init__(self, courant, weighting_factor=0.5):
        super(__class__, self).__init__()