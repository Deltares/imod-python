import string

import numpy as np

from imod.mf6.pkgbase import Package, VariableMetaData


class MobileStorage(Package):
    """
    Mobile Storage.

    Parameters
    ----------
    porosity: array of floats (xr.DataArray)
        volume of interconnected voids per volume of rock (percentage).
    decay : array of floats (xr.DataArray, optional)
        is the rate coefficient for first or zero-order decay for the aqueous phase of the mobile domain.
        A negative value indicates solute production. The dimensions of decay for first-order decay is one
        over time. The dimensions of decay for zero-order decay is mass per length cubed per time. decay will
        have no effect on simulation results unless either first- or zero-order decay is specified in the
        options block.
    decay_sorbed : array of floats (xr.DataArray, optional)
        is the rate coefficient for first or zero-order decay for the sorbed phase of the mobile domain.
        A negative value indicates solute production. The dimensions of decay_sorbed for first-order decay
        is one over time. The dimensions of decay_sorbed for zero-order decay is mass of solute per mass of
        aquifer per time. If decay_sorbed is not specified and both decay and sorption are active, then the
        program will terminate with an error. decay_sorbed will have no effect on simulation results unless
        the SORPTION keyword and either first- or zero-order decay are specified in the options block.
    bulk_density : array of floats (xr.DataArray, optional)
        is the bulk density of the aquifer in mass per length cubed. bulk_density is not required unless
        the SORPTION keyword is specified.
    distcoef  : array of floats (xr.DataArray, optional)
        is the distribution coefficient for the equilibrium-controlled linear sorption isotherm in dimensions
        of length cubed per mass. distcoef is not required unless the SORPTION keyword is specified.
    sp2  : array of floats (xr.DataArray, optional)
        is the exponent for the Freundlich isotherm and the sorption capacity for the Langmuir isotherm.
    save_flows: ({True, False}, optional)
        Indicates that recharge flow terms will be written to the file specified
        with "BUDGET FILEOUT" in Output Control.
        Default is False.
    decay_order: ({ first, zero}, optional)
        Indicates wheter decay is first-order or zero-order decay. Requires decay to be specified
        (decay_sorbed too if sorption is active)
    sorption: ({Linear, Freundlich, Langmuir}, optional)
        Type of sorption, if any
    """

    _grid_data = {
        "porosity": np.float64,
        "decay": np.float64,
        "decay_sorbed": np.float64,
        "bulk_density": np.float64,
        "distcoef": np.float64,
        "sp2": np.float64,
    }

    _pkg_id = "mst"
    _template = Package._initialize_template(_pkg_id)
    _keyword_map = {}
    _metadata_dict = {
        "porosity": VariableMetaData(np.floating),
        "decay": VariableMetaData(np.floating),
        "decay_sorbed": VariableMetaData(np.floating),
        "bulk_density": VariableMetaData(np.floating),
        "distcoef": VariableMetaData(np.floating),
        "sp2": VariableMetaData(np.floating),
    }

    def __init__(
        self,
        porosity,
        decay=None,
        decay_sorbed=None,
        bulk_density=None,
        distcoef=None,
        sp2=None,
        save_flows=False,
        decay_order: string = "first",
        sorption=None,
    ):
        super().__init__(locals())
        self.dataset["porosity"] = porosity
        self.dataset["decay"] = decay
        self.dataset["decay_sorbed"] = decay_sorbed
        self.dataset["bulk_density"] = bulk_density
        self.dataset["distcoef"] = distcoef
        self.dataset["sp2"] = sp2

        self.dataset["save_flows"] = save_flows
        if decay_order.lower().strip() == "zero":
            self.dataset["zero_order_decay"] = decay
        elif decay_order.lower().strip() == "first":
            self.dataset["first_order_decay"] = decay
        elif decay_order is not None:
            raise ValueError('decay_order should be "first" or "zero" when present')
        if sorption is not None:
            self.dataset["sorption"] = sorption

        self._pkgcheck()
