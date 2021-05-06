import numpy as np
import scipy.ndimage
from imod.flow.pkgbase import Package


class Boundary(Package):
    """
    Specify the locations of active, inactive, and specified head in cells

    Parameters
    ----------
    ibound: xr.DataArray of ints
        is the boundary variable with dimensions ``("layer", "y", "x")``.

        * If IBOUND(J,I,K) < 0, cell J,I,K has a constant head.
        * If IBOUND(J,I,K) = 0, cell J,I,K is inactive.
        * If IBOUND(J,I,K) > 0, cell J,I,K is active.
    """

    _pkg_id = "bnd"
    _variable_order = ["ibound"]

    def __init__(self, ibound):
        super(__class__, self).__init__()
        self.dataset["ibound"] = ibound

    def _pkgcheck(self, active_cells=None):
        _, nlabels = scipy.ndimage.label(active_cells.values)
        if nlabels > 1:
            raise ValueError(
                f"{nlabels} disconnected model domain detected in the ibound"
            )


class Top(Package):
    """
    The top of the aquifers

    Parameters
    ----------
    top: xr.DataArray of floats
        is the top elevation with dimensions ``("layer", "y", "x")``. For the
        common situation in which the top layer represents a water-table
        aquifer, it may be reasonable to set`top` equal to land-surface
        elevation.  The DataArray should at least include the `layer`
        dimension.
    """

    _pkg_id = "top"
    _variable_order = ["top"]

    def __init__(self, top):
        super(__class__, self).__init__()
        self.dataset["top"] = top

    def _pkgcheck(self, active_cells=None):
        vars_to_check = ["top"]
        self._check_if_nan_in_active_cells(
            active_cells=active_cells, vars_to_check=vars_to_check
        )


class Bottom(Package):
    """
    The bottom of the aquifers

    Parameters
    ----------
    bottom: xr.DataArray of floats
        is the bottom elevation of model layers or Quasi-3d confining beds,
        with dimensions ``("layer", "y", "x")``. The DataArray should at least
        include the `layer` dimension.
    """

    _pkg_id = "bot"
    _variable_order = ["bottom"]

    def __init__(self, bottom):
        super(__class__, self).__init__()
        self.dataset["bottom"] = bottom

    def _pkgcheck(self, active_cells=None):
        vars_to_check = ["bottom"]
        self._check_if_nan_in_active_cells(
            active_cells=active_cells, vars_to_check=vars_to_check
        )


class StartingHead(Package):
    """
    The initial head in all cells

    Parameters
    ----------
    starting_head: float or xr.DataArray of floats
        is initial (starting) head—that is, head at the beginning of the
        simulation (SHD). starting_head must be specified for all simulations,
        including steady-state simulations. One value is read for every model
        cell. Usually, these values are read a layer at a time.
    """

    _pkg_id = "shd"
    _variable_order = ["starting_head"]

    def __init__(self, starting_head):
        super(__class__, self).__init__()
        self.dataset["starting_head"] = starting_head

    def _pkgcheck(self, active_cells=None):
        vars_to_check = ["starting_head"]
        self._check_if_nan_in_active_cells(
            active_cells=active_cells, vars_to_check=vars_to_check
        )
