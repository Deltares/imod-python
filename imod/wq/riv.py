from imod.wq.pkgbase import BoundaryCondition


class River(BoundaryCondition):
    """
    The River package is used to simulate head-dependent flux boundaries. In the
    River package if the head in the cell falls below a certain threshold, the
    flux from the river to the model cell is set to a specified lower bound.

    Parameters
    ----------
    stage: array of floats (xr.DataArray)
        is the head in the river (STAGE).
    bottom_elevation: array of floats (xr.DataArray)
        is the bottom of the riverbed (RBOT).
    conductance: array of floats (xr.DataArray)
        is the conductance of the river.
    concentration: "None", float or array of floats (xr.DataArray), optional
        is the concentration in the river.
        Default is None.
    density: "None" or float, optional
        is the density used to convert the point head to the freshwater head
        (RIVSSMDENS). If not specified, the density is calculated dynamically
        using the concentration. Default is None.
    save_budget: {True, False}, optional
        is a flag indicating if the budget should be saved (IRIVCB).
        Default is False.
    """

    _pkg_id = "riv"

    _mapping = (
        ("stage", "stage"),
        ("cond", "conductance"),
        ("rbot", "bottom_elevation"),
        ("rivssmdens", "density"),
    )

    def __init__(
        self,
        stage,
        conductance,
        bottom_elevation,
        concentration=None,
        density=None,
        save_budget=False,
    ):
        super(__class__, self).__init__()
        self["stage"] = stage
        self["conductance"] = conductance
        self["bottom_elevation"] = bottom_elevation
        if concentration is not None:
            self["concentration"] = concentration
        if density is not None:
            self["density"] = density
        self["save_budget"] = save_budget

    def _pkgcheck(self, ibound=None):
        to_check = ["conductance"]
        if "concentration" in self.data_vars:
            to_check.append("concentration")
        if "density" in self.data_vars:
            to_check.append("density")
        self._check_positive(to_check)

        to_check.append("stage")
        to_check.append("bottom_elevation")
        self._check_location_consistent(to_check)

        if (self["bottom_elevation"] > self["stage"]).any():
            raise ValueError(
                "Bottom elevation in {self} should not be higher than stage"
            )

    def add_timemap(
        self,
        stage=None,
        conductance=None,
        bottom_elevation=None,
        concentration=None,
        density=None,
        use_cftime=False,
    ):
        varnames = [
            "stage",
            "conductance",
            "bottom_elevation",
            "concentration",
            "density",
        ]
        values = [stage, conductance, bottom_elevation, concentration, density]
        for varname, value in zip(varnames, values):
            self._add_timemap(varname, value, use_cftime)
