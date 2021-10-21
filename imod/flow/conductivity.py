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
        super(__class__, self).__init__()
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
        super(__class__, self).__init__()
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
        super(__class__, self).__init__()
        self.dataset["vertical_anisotropy"] = vertical_anisotropy

    def _pkgcheck(self, active_cells=None):
        vars_to_check = ["vertical_anisotropy"]
        self._check_if_nan_in_active_cells(
            active_cells=active_cells, vars_to_check=vars_to_check
        )
