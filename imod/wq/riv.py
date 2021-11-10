from imod.wq.pkgbase import BoundaryCondition


class River(BoundaryCondition):
    """
    The River package is used to simulate head-dependent flux boundaries. In the
    River package if the head in the cell falls below a certain threshold, the
    flux from the river to the model cell is set to a specified lower bound.

    Parameters
    ----------
    stage: float or xr.DataArray of floats
        is the head in the river (STAGE).
    bottom_elevation: float or xr.DataArray of floats
        is the bottom of the riverbed (RBOT).
    conductance: float or xr.DataArray of floats
        is the conductance of the river.
    density: float or xr.DataArray of floats
        is the density used to convert the point head to the freshwater head
        (RIVSSMDENS).
    concentration: "None", float or xr.DataArray of floats, optional
        is the concentration in the river.
        Default is None.
    save_budget: bool, optional
        is a flag indicating if the budget should be saved (IRIVCB).
        Default is False.
    """

    __slots__ = (
        "stage",
        "conductance",
        "bottom_elevation",
        "density",
        "concentration",
        "save_budget",
    )
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
        density,
        concentration=None,
        save_budget=False,
    ):
        super().__init__()
        self["stage"] = stage
        self["conductance"] = conductance
        self["bottom_elevation"] = bottom_elevation
        self["density"] = density
        if concentration is not None:
            self["concentration"] = concentration
        self["save_budget"] = save_budget

    def _pkgcheck(self, ibound=None):
        to_check = ["conductance", "density"]
        if "concentration" in self.data_vars:
            to_check.append("concentration")
        self._check_positive(to_check)

        to_check.append("stage")
        to_check.append("bottom_elevation")
        self._check_location_consistent(to_check)

        if (self["bottom_elevation"] > self["stage"]).any():
            raise ValueError(
                "Bottom elevation in {self} should not be higher than stage"
            )

    def repeat_stress(
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
            "density",
            "concentration",
        ]
        values = [stage, conductance, bottom_elevation, density, concentration]
        for varname, value in zip(varnames, values):
            self._repeat_stress(varname, value, use_cftime)
