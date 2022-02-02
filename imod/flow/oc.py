from imod.flow.pkgbase import Package
import numpy as np


class OutputControl(Package):
    """
    The Output Control Option determines how output is printed to the listing
    file and/or written to a separate binary output file.

    Parameters
    ----------
    save_head : {xr.DataArray, integer}
        xr.DataArray of integers or integer indicating the layer at which heads should
        be saved. If set to -1, output is saved for all layers. If set to 0,
        nothing is saved.
    save_flux : {xr.DataArray, integer}
        xr.DataArray of integers or integer indicating the layer at which spatial fluxes
        should be saved. If set to -1, output is saved for all layers. If set to
        0, nothing is saved.
    save_ghb : {xr.DataArray, integer}
        xr.DataArray of integers or integer indicating the layer at which ghb budget
        should be saved. If set to -1, output is saved for all layers. If set to
        0, nothing is saved.
    save_drn : {xr.DataArray, integer}
        xr.DataArray of integers or integer indicating the layer at which drn budget
        should be saved. If set to -1, output is saved for all layers. If set to
        0, nothing is saved.
    save_wel : {xr.DataArray, integer}
        xr.DataArray of integers or integer indicating the layer at which wel budget
        should be saved. If set to -1, output is saved for all layers. If set to
        0, nothing is saved.
    save_riv : {xr.DataArray, integer}
        xr.DataArray of integers or integer indicating the layer at which riv budget
        should be saved. If set to -1, output is saved for all layers. If set to
        0, nothing is saved.
    save_rch : {xr.DataArray, integer}
        xr.DataArray of integers or integer indicating the layer at which rch budget
        should be saved. If set to -1, output is saved for all layers. If set to
        0, nothing is saved.
    save_evt : {xr.DataArray, integer}
        xr.DataArray of integers or integer indicating the layer at which evt budget
        should be saved. If set to -1, output is saved for all layers. If set to
        0, nothing is saved.
    save_hfb : {xr.DataArray, integer}
        xr.DataArray of integers or integer indicating the layer at which hfb budget
        should be saved. If set to -1, output is saved for all layers. If set to
        0, nothing is saved.

    Examples
    --------

    Don't save heads:

    >>> oc = imod.flow.OutputControl(save_head=0)

    Save heads for all layers:

    >>> oc = imod.flow.OutputControl(save_head=-1)

    Save heads for specific layers, in this case layer 1 and 4:

    >>> specific_layers = xr.DataArray([1,4], coords={"layer": [1,4], dims=("layer",)})
    >>> oc = imod.flow.OutputControl(save_head=specific_layers)


    """

    _pkg_id = "oc"
    _variable_order = [
        "save_head",
        "save_flux",
        "save_ghb",
        "save_drn",
        "save_wel",
        "save_riv",
        "save_rch",
        "save_evt",
        "save_hfb",
    ]

    def __init__(
        self,
        save_head=0,
        save_flux=0,
        save_ghb=0,
        save_drn=0,
        save_wel=0,
        save_riv=0,
        save_rch=0,
        save_evt=0,
        save_hfb=0,
    ):
        super().__init__()
        self.dataset["save_head"] = save_head
        self.dataset["save_flux"] = save_flux
        self.dataset["save_ghb"] = save_ghb
        self.dataset["save_drn"] = save_drn
        self.dataset["save_wel"] = save_wel
        self.dataset["save_riv"] = save_riv
        self.dataset["save_rch"] = save_rch
        self.dataset["save_evt"] = save_evt
        self.dataset["save_hfb"] = save_hfb

    def _compose_oc_configuration(self, nlayer):
        # Create mapping to convert e.g. 'save_ghb' to 'saveghb'
        conf_mapping = dict(
            [(var, var.replace("_", "")) for var in self._variable_order]
        )
        conf_mapping["save_head"] = "saveshd"
        conf_mapping["save_flux"] = "saveflx"

        pkg_composition = self.compose(None, None, nlayer)["oc"]

        oc_composition = {}

        for var in self._variable_order:
            values = np.array(list(pkg_composition[var].values()))
            # integer was provided
            if np.all(values == values[0]):
                oc_composition[conf_mapping[var]] = values[0]
            elif np.any(values == 0) or np.any(values == -1):
                raise ValueError(
                    f"Received invalid layer numbers, layer array cannot contain 0 or -1, got {values} for {var}"
                )
            else:
                oc_composition[conf_mapping[var]] = ",".join(
                    str(value) for value in values
                )

        return oc_composition
