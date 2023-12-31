from imod.flow.pkgbase import Package


class HorizontalHydraulicConductivity(Package):
    """
    Horizontal hydraulic conductivity [L/T] of the aquifers, between TOP and
    BOT.

    This variable behaves somewhat similar to the horizontal hydraulic
    conductivity in MODFLOW 2005's "Layer Property Flow" schematization.

    Note however that this does not hold for the vertical hydraulic
    conductivity: iMODFLOW uses the vertical hydraulic conductivity to specify
    the hydraulic conductivity of aquitards (between BOT and TOP)

    Parameters
    ----------
    k_horizontal : xr.DataArray
        Horizontal hydraulic conductivity, dims ``("layer", "y", "x")``.
    """

    _pkg_id = "khv"
    _variable_order = ["k_horizontal"]

    def __init__(self, k_horizontal=None):
        super().__init__()
        self.dataset["k_horizontal"] = k_horizontal

    def _pkgcheck(self, active_cells=None):
        vars_to_check = ["k_horizontal"]
        self._check_if_nan_in_active_cells(
            active_cells=active_cells, vars_to_check=vars_to_check
        )


class VerticalHydraulicConductivity(Package):
    """
    Vertical hydraulic conductivity [L/T] for aquitards (between BOT and TOP).

    Note that this is different from MODFLOW 2005's "Layer Property Flow"
    schematization.  To specify the vertical hydraulic conductivity for
    aquifers, use VerticalAnisotropy in combination with
    HorizontalHydraulicConductivity.

    Parameters
    ----------
    k_vertical : xr.DataArray
        Vertical hydraulic conductivity, dims ``("layer", "y", "x")``.
    """

    _pkg_id = "kvv"
    _variable_order = ["k_vertical"]

    def __init__(self, k_vertical=None):
        super().__init__()
        self.dataset["k_vertical"] = k_vertical

    def _pkgcheck(self, active_cells=None):
        vars_to_check = ["k_vertical"]
        self._check_if_nan_in_active_cells(
            active_cells=active_cells, vars_to_check=vars_to_check
        )


class VerticalAnisotropy(Package):
    """
    Vertical anisotropy for aquifers [-], defined as the horizontal hydraulic
    conductivity over the vertical hydraulic conductivity.

    Use this package in combination with HorizontalHydraulicConductivity to
    specify the vertical hydraulic conductivity.

    Parameters
    ----------
    vertical_anisotropy : xr.DataArray
        Vertical anisotropy factor (Kv/Kh), dims ``("layer", "y", "x")``.
    """

    _pkg_id = "kva"
    _variable_order = ["vertical_anisotropy"]

    def __init__(self, vertical_anisotropy):
        super().__init__()
        self.dataset["vertical_anisotropy"] = vertical_anisotropy

    def _pkgcheck(self, active_cells=None):
        vars_to_check = ["vertical_anisotropy"]
        self._check_if_nan_in_active_cells(
            active_cells=active_cells, vars_to_check=vars_to_check
        )


class Transmissivity(Package):
    """
    Transmissivity [L2/T] of the aquifers, between TOP and BOT.

    This variable behaves similar to the transmissivity in MODFLOW 2005's "Block
    Centred Flow (BCF)" schematization. Note that values are trimmed internally
    to be minimal 0 m2/day by iMODFLOW.

    Parameters
    ----------
    transmissivity : xr.DataArray
        Transmissivity, dims ``("layer", "y", "x")``.
    """

    _pkg_id = "kdw"
    _variable_order = ["transmissivity"]

    def __init__(self, transmissivity=None):
        super().__init__()
        self.dataset["transmissivity"] = transmissivity

    def _pkgcheck(self, active_cells=None):
        vars_to_check = ["transmissivity"]
        self._check_if_nan_in_active_cells(
            active_cells=active_cells, vars_to_check=vars_to_check
        )


class VerticalResistance(Package):
    """
    Vertical resistance [T] for aquitards (Between BOT and TOP).

    This variable is the inverse of the "VCONT" variable in MODFLOW 2005's
    "Block Centred Flow (BCF)" schematization. Note that values are trimmed
    internally for minimal 0.001 days by iMODFLOW. Also not that iMODFLOW's
    internal scaling cannot deal with value nonzero nodata values, therefore,
    for scaling, it is important to assign a 0 as the nodata value for
    VerticalResistance.

    Parameters
    ----------
    Resistance : xr.DataArray
        Resistance, dims ``("layer", "y", "x")``.

    """

    _pkg_id = "vcw"
    _variable_order = ["resistance"]

    def __init__(self, resistance=None):
        super().__init__()
        self.dataset["resistance"] = resistance
