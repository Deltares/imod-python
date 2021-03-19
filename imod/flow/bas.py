from imod.flow.pkgbase import Package

class BasicFlow(Package):
    """
    The Basic package is used to specify certain data used in all models.
    These include:
    1. the locations of acitve, inactive, and specified head in cells,
    2. the head stored in inactive cells,
    3. the initial head in all cells, and
    4. the top and bottom of the aquifer
    The number of layers (NLAY) is automatically calculated using the IBOUND.
    Thickness is calculated using the specified tops en bottoms.
    The Basic package input file is required in all models.

    Parameters
    ----------
    ibound: xr.DataArray of integers
        is the boundary variable.
        If IBOUND(J,I,K) < 0, cell J,I,K has a constant head.
        If IBOUND(J,I,K) = 0, cell J,I,K is inactive.
        If IBOUND(J,I,K) > 0, cell J,I,K is active.
    top: float or xr.DataArray of floats
        is the top elevation of layer 1. For the common situation in which the
        top layer represents a water-table aquifer, it may be reasonable to set
        `top` equal to land-surface elevation.
    bottom: xr.DataArray of floats
        is the bottom elevation of model layers or Quasi-3d confining beds. The
        DataArray should at least include the `layer` dimension.
    starting_head: float or xr.DataArray of floats
        is initial (starting) head—that is, head at the beginning of the
        simulation (STRT). starting_head must be specified for all simulations,
        including steady-state simulations. One value is read for every model
        cell. Usually, these values are read a layer at a time.
    inactive_head: float, optional
        is the value of head to be assigned to all inactive (no flow) cells
        (IBOUND = 0) throughout the simulation (HNOFLO). Because head at
        inactive cells is unused in model calculations, this does not affect
        model results but serves to identify inactive cells when head is
        printed. This value is also used as drawdown at inactive cells if the
        drawdown option is used. Even if the user does not anticipate having
        inactive cells, a value for inactive_head must be entered.
        Default value is 1.0e30.
    """

    _pkg_id = "bas"

    #TODO iMODFLOW expects seperate packages: TOP, BOT, BND, SHD, so seperate this object
    _var_order = ["ibound", "top", "bottom", "starting_head"] 

    def __init__(self, ibound, top, bottom, starting_head):
        super(__class__, self).__init__()
        self.dataset["ibound"] = ibound
        self.dataset["top"] = top
        self.dataset["bottom"] = bottom
        self.dataset["starting_head"] = starting_head

    
