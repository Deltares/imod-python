import numpy as np

from imod.mf6.pkgbase import Package


class NodePropertyFlow(Package):
    """
    Attributes
    ----------

    cell_averaging : str
        Method calculating horizontal cell connection conductance.
        Options: {"harmonic", "logarithmic", "mean-log_k", "mean-mean_k"}
    

    """

    __slots__ = (
        "icelltype",
        "k",
        "rewet",
        "rewet_layer",
        "rewet_factor",
        "rewet_iterations",
        "rewet_method",
        "k22",
        "k33",
        "angle1",
        "angle2",
        "angle3",
        "cell_averaging",
        "save_flows",
        "starting_head_as_confined_thickness",
        "variable_vertical_conductance",
        "dewatered",
        "perched",
        "save_specific_discharge",
    )
    _pkg_id = "npf"
    _binary_data = {
        "icelltype": np.int32,
        "k": np.float64,
        "rewet_layer": np.float64,
        "k22": np.float64,
        "k33": np.float64,
        "angle1": np.float64,
        "angle2": np.float64,
        "angle3": np.float64,
    }
    _template = Package._initialize_template(_pkg_id)

    def __init__(
        self,
        icelltype,
        k,
        rewet=False,
        rewet_layer=None,
        rewet_factor=None,
        rewet_iterations=None,
        rewet_method=None,
        k22=None,
        k33=None,
        angle1=None,
        angle2=None,
        angle3=None,
        cell_averaging="harmonic",
        save_flows=False,
        starting_head_as_confined_thickness=False,
        variable_vertical_conductance=False,
        dewatered=False,
        perched=False,
        save_specific_discharge=False,
    ):
        super(__class__, self).__init__()
        # check rewetting
        if not rewet and any(
            [rewet_layer, rewet_factor, rewet_iterations, rewet_method]
        ):
            raise ValueError(
                "rewet_layer, rewet_factor, rewet_iterations, and rewet_method should"
                " all be left at a default value of None if rewet is False."
            )
        self["icelltype"] = icelltype
        self["k"] = k
        self["rewet"] = rewet
        self["rewet_layer"] = rewet_layer
        self["rewet_factor"] = rewet_factor
        self["rewet_iterations"] = rewet_iterations
        self["rewet_method"] = rewet_method
        self["k22"] = k22
        self["k33"] = k33
        self["angle1"] = angle1
        self["angle2"] = angle2
        self["angle3"] = angle3
        self["cell_averaging"] = cell_averaging
        self["save_flows"] = save_flows
        self[
            "starting_head_as_confined_thickness"
        ] = starting_head_as_confined_thickness
        self["variable_vertical_conductance"] = variable_vertical_conductance
        self["dewatered"] = dewatered
        self["perched"] = perched
        self["save_specific_discharge"] = save_specific_discharge

    def render(self, directory, pkgname, *args, **kwargs):
        d = {}
        replace_keywords = {
            "rewet": "rewet_record",
            "rewet_factor": "wetfct",
            "rewet_method": "ihdwet",
            "rewet_layer": "wetdry",
            "variable_vertical_conductance": "variablecv",
            "starting_head_as_confined_thickness": "thickstrt",
        }
        npfdirectory = directory / "npf"
        for varname in self.data_vars:
            key = replace_keywords.get(varname, varname)

            if varname in self._binary_data:
                layered, value = self._compose_values(
                    self[varname], npfdirectory, varname
                )
                if self._valid(value):  # skip False or None
                    d[f"{key}_layered"], d[key] = layered, value
            else:
                value = self[varname].values[()]
                if self._valid(value):  # skip False or None
                    d[key] = value

        return self._template.render(d)
