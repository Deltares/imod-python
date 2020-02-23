import jinja2

from imod.wq.pkgbase import Package


class LayerPropertyFlow(Package):
    """
    The Layer-Property Flow (LPF) package is used to specify properties
    controlling flow between cells.

    Parameters
    ----------
    k_horizontal: float or array of floats (xarray.DataArray)
        is the hydraulic conductivity along rows (HK). HK is multiplied by
        horizontal anisotropy (see horizontal_anisotropy) to obtain hydraulic
        conductivity along columns.
    k_vertical: float or array of floats (xarray.DataArray)
        is the vertical hydraulic conductivity (VKA).
    horizontal_anisotropy: float or array of floats (xarray.DataArray)
        contains a value for each layer that is the horizontal anisotropy
        (CHANI). Use as many records as needed to enter a value of CHANI for
        each layer. The horizontal anisotropy is the ratio of the hydraulic
        conductivity along columns (the Y direction) to the hydraulic
        conductivity along rows (the X direction).
    interblock: int
        contains a flag for each layer that defines the method of calculating
        interblock transmissivity (LAYAVG). Use as many records needed to enter
        a value for each layer.
        0 = harmonic mean (This is most appropriate for confined and unconfined
        aquifers with abrupt boundaries in transmissivity at the cell boundaries
        or for confined aquifers with uniform hydraulic conductivity).
        1 = logarithmic mean (This is most appropriate for confined aquifers
        with gradually varying transmissivities).
        2 = arithmetic mean of saturated thickness and logarithmic-mean
        hydraulic conductivity. (This is most appropriate for unconfined
        aquifers with gradually varying transmissivities).
    layer_type: int
        contains a flag for each layer that specifies the layer type (LAYTYP).
        Use as many records needed to enter a value for each layer.
        0 = confined
        not 0 = convertible
    specific_storage: float or array of floats (xarray.DataArray)
        is specific storage (SS). Read only for a transient simulation (at least
        one transient stress period). Include only if at least one stress period
        is transient.
        Specific storage is the amount of water released when the head in an aquifer 
        drops by 1 m, in one meter of the aquifer (or model layer). 
        The unit is: ((m3 / m2) / m head change) / m aquifer = m-1
    specific_yield: float or array of floats (xarray.DataArray)
        is specific yield (SY). Read only for a transient simulation (at least
        one transient stress period) and if the layer is convertible (layer_type
        is not 0). Include only if at least one stress period is transient. 
        The specific yield is the volume of water released from (or added to) the
        pore matrix for one meter of head change. 
        The unit is: (m3 / m2) / m head change = dimensionless 
    save_budget: int
        is a flag and a unit number (ILPFCB).
        If save_budget > 0, it is the unit number to which cell-by-cell flow
        terms will be written when "SAVE BUDGET" or a non-zero value for
        save_budget is specified in Output Control. The terms that are saved are
        storage, constant-head flow, and flow between adjacent cells.
        If save_budget = 0, cell-by-cell flow terms will not be written.
        If save_budget < 0, cell-by-cell flow for constant-head cells will be
        written in the listing file when "SAVE BUDGET" or a non-zero value for
        ICBCFL is specified in Output Control. Cell-by-cell flow to storage and
        between adjacent cells will not be written to any file. The flow terms
        that will be saved are the flows through the right, front, and lower
        cell face. Positive values represent flows toward higher column, row, or
        layer numbers.
    layer_wet: int
        contains a flag for each layer that indicates if wetting is active. Use
        as many records as needed to enter a value for each layer.
        0 = wetting is inactive
        not 0 = wetting is active
    interval_wet: int
        is the iteration interval for attempting to wet cells. Wetting is
        attempted every interval_wet iteration (IWETIT). If using the PCG solver
        (Hill, 1990), this applies to outer iterations, not inner iterations. If
        interval_wet less than or equal to 0, it is changed to 1.
    method_wet: int
        is a flag that determines which equation is used to define the initial
        head at cells that become wet (IHDWET).
        If method_wet = 0, this equation is used:
        h = BOT + WETFCT (hn - BOT).
        (hn is the head in the neighboring cell that is causing the dry cell to
        convert to an active cell.)
        If method_wet is not 0, this equation is used:
        h = BOT + WETFCT(THRESH).
        WETFCT is a factor that is included in the calculation of the head that
        is initially established at a cell when it is converted from dry to wet.
    head_dry: float, optional
        is the head that is assigned to cells that are converted to dry during a
        simulation (HDRY). Although this value plays no role in the model calculations,
        it is useful as an indicator when looking at the resulting heads that
        are output from the model. HDRY is thus similar to HNOFLO in the Basic
        Package, which is the value assigned to cells that are no-flow cells at
        the start of a model simulation.
        Default value: 1.0e20.
    """

    __slots__ = (
        "k_horizontal",
        "k_vertical",
        "horizontal_anisotropy",
        "interblock",
        "layer_type",
        "specific_storage",
        "specific_yield",
        "save_budget",
        "layer_wet",
        "interval_wet",
        "method_wet",
        "head_dry",
    )
    _pkg_id = "lpf"

    _mapping = (
        ("laytyp", "layer_type"),
        ("layavg", "interblock"),
        ("chani", "horizontal_anisotropy"),
        ("hk", "k_horizontal"),
        ("vka", "k_vertical"),
        ("ss", "specific_storage"),
        ("sy", "specific_yield"),
        ("laywet", "layer_wet"),
    )

    _template = jinja2.Template(
        "[lpf]\n"
        "    ilpfcb = {{save_budget}}\n"
        "    hdry = {{head_dry}}\n"
        "    layvka_l? = 0\n"
        "    {%- for name, dictname in mapping -%}\n"
        "        {%- for layer, value in dicts[dictname].items() %}\n"
        "    {{name}}_l{{layer}} = {{value}}\n"
        "        {%- endfor -%}\n"
        "    {%- endfor -%}\n"
    )

    _keywords = {
        "save_budget": {False: 0, True: 1},
        "method_wet": {"wetfactor": 0, "bottom": 1},
    }

    def __init__(
        self,
        k_horizontal,
        k_vertical,
        horizontal_anisotropy=1.0,
        interblock=0,
        layer_type=0,
        specific_storage=0.0001,
        specific_yield=0.15,
        save_budget=False,
        layer_wet=0,
        interval_wet=0.001,
        method_wet="wetfactor",
        head_dry=1.0e20,
    ):
        super(__class__, self).__init__()
        self["k_horizontal"] = k_horizontal
        self["k_vertical"] = k_vertical
        self["horizontal_anisotropy"] = horizontal_anisotropy
        self["interblock"] = interblock
        self["layer_type"] = layer_type
        self["specific_storage"] = specific_storage
        self["specific_yield"] = specific_yield
        self["save_budget"] = save_budget
        self["layer_wet"] = layer_wet
        self["interval_wet"] = interval_wet
        self["method_wet"] = method_wet
        self["head_dry"] = head_dry

    def _render(self, directory, *args, **kwargs):
        d = {}
        # Don't include absentee members
        mapping = tuple([(k, v) for k, v in self._mapping if v in self.data_vars])
        d["mapping"] = mapping
        dicts = {}

        da_vars = [t[1] for t in self._mapping]
        for varname in self.data_vars.keys():
            if varname in da_vars:
                dicts[varname] = self._compose_values_layer(varname, directory)
            else:
                d[varname] = self[varname].values
                if varname == "save_budget" or varname == "method_wet":
                    self._replace_keyword(d, varname)
        d["dicts"] = dicts

        return self._template.render(d)

    def _pkgcheck(self, ibound=None):
        to_check = [
            "k_horizontal",
            "k_vertical",
            "horizontal_anisotropy",
            "specific_storage",
            "specific_yield",
        ]
        self._check_positive(to_check)
        self._check_location_consistent(to_check)
