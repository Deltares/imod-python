from imod.flow.pkgbase import Package


class HorizontalHydraulicConductivity(Package):
    """
    Specify horizontal hydraulic conductivity of the aquifers.

    Assigning this package to a model means you chose MODFLOW 2005's
    "Layer Property Flow (LPF)" schematization.
    Assigning packages of the Block Centered Flow (BCF) as well
    to the model will result in errors.

    Parameters
    ----------
    k_horizontal : xr.DataArray
        Horizontal hydraulic conductivity.
    """

    _pkg_id = "khv"
    _variable_order = ["k_horizontal"]

    def __init__(self, k_horizontal=None):
        super(__class__, self).__init__()
        self.dataset["k_horizontal"] = k_horizontal


class VerticalHydraulicConductivity(Package):
    """
    Specify vertical hydraulic conductivity for aquitards (between BOT and TOP)
    To specify the vertical hydraulic conductivity for aquifers,
    use "VerticalAnisotropy" in combination with HorizontalHydraulicConductivity.

    Assigning this package to a model means you chose MODFLOW 2005's
    "Layer Property Flow (LPF)" schematization.
    Assigning packages of the Block Centered Flow (BCF) as well
    to the model will result in errors.

    Parameters
    ----------
    k_vertical : xr.DataArray
        Vertical hydraulic conductivity.
    """

    _pkg_id = "kvv"
    _variable_order = ["k_vertical"]

    def __init__(self, k_vertical=None):
        super(__class__, self).__init__()
        self.dataset["k_vertical"] = k_vertical


class VerticalAnistropy(Package):
    """
    Specify the vertical anisotropy for aquifers, defined as the
    vertical hydraulic conductivity over the horizontal
    hydraulic conductivity.

    vertical_anistropy : xr.DataArray
        Vertical anistropy factor (Kv/Kh).
    """

    _pkg_id = "kva"
    _variable_order = ["vertical_anistropy"]

    def __init__(self, vertical_anistropy=None):
        super(__class__, self).__init__()
        self.dataset["vertical_anistropy"] = vertical_anistropy
