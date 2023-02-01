import numpy as np

from imod.mf6.pkgbase import Package
from imod.mf6.validation import PKG_DIMS_SCHEMA
from imod.schemata import (
    AllValueSchema,
    DTypeSchema,
    IdentityNoDataSchema,
    IndexesSchema,
)


class ImmobileStorageTransfer(Package):
    """
    The Immobile Storage and Transfer (IST) package represents an immobile
    fraction of groundwater. Any number of IST Packages can be specified for a
    single GWT model. This allows the user to specify triple porosity systems,
    or systems with as many immobile domains as necessary.

    Parameters
    ----------
    initial_immmobile_concentration : array of floats (xr.DataArray)
        initial concentration of the immobile domain in mass per length cubed
        (cim).
    immobile_porosity : array of floats (xr.DataArray)
        porosity of the immobile domain specified as the volume of immobile
        pore space per total volume (dimensionless) (thetaim).
    mobile_immobile_mass_transfer_rate: array of floats (xr.DataArray)
        mass transfer rate coefficient between the mobile and immobile domains,
        in dimensions of per time (zetaim).
    decay: array of floats (xr.DataArray).
        is the rate coefficient for first or zero-order decay for the aqueous
        phase of the immobile domain. A negative value indicates solute
        production. The dimensions of decay for first-order decay is one over
        time. The dimensions of decay for zero-order decay is mass per length
        cubed per time. Decay will have no effect on simulation results unless
        either first- or zero-order decay is specified in the options block.
    decay_sorbed: array of floats (xr.DataArray)
        is the rate coefficient for first or zero-order decay for the sorbed
        phase of the immobile domain. A negative value indicates solute
        production. The dimensions of decay_sorbed for first-order decay is one
        over time. The dimensions of decay_sorbed for zero-order decay is mass
        of solute per mass of aquifer per time. If decay_sorbed is not
        specified and both decay and sorption are active, then the program will
        terminate with an error. decay_sorbed will have no effect on simulation
        results unless the SORPTION keyword and either first- or zero-order
        decay are specified in the options block.
    bulk_density: array of floats (xr.DataArray)
        is the bulk density of the aquifer in mass per length cubed.
        bulk_density will have no effect on simulation results unless the
        SORPTION keyword is specified in the options block.
    distribution_coefficient: array of floats (xr.DataArray)
        is the distribution coefficient for the equilibrium-controlled linear
        sorption isotherm in dimensions of length cubed per mass. distcoef will
        have no effect on simulation results unless the SORPTION keyword is
        specified in the options block.
    save_flows: ({True, False}, optional)
        Indicates that drain flow terms will be written to the file specified
        with "BUDGET FILEOUT" in Output Control. Default is False.
    budgetbinfile:
        name of the binary output file to write budget information.
    budgetcsvfile:
        name of the comma-separated value (CSV) output file to write budget
        summary information. A budget summary record will be written to this
        file for each time step of the simulation.
    sorption: ({True, False}, optional)
        is a text keyword to indicate that sorption will be activated. Use of
        this keyword requires that BULK_DENSITY and DISTCOEF are specified in
        the GRIDDATA block. The linear sorption isotherm is the only isotherm
        presently supported in the IST Package.
    first_order_decay: ({True, False}, optional)
        is a text keyword to indicate that first-order decay will occur. Use of
        this keyword requires that DECAY and DECAY_SORBED (if sorption is
        active) are specified in the GRIDDATA block.
    zero_order_decay: ({True, False}, optional)
        is a text keyword to indicate that zero-order decay will occur. Use of
        this keyword requires that DECAY and DECAY_SORBED (if sorption is
        active) are specified in the GRIDDATA block.
    cimfile: (str)
        name of the output file to write immobile concentrations. This file is
        a binary file that has the same format and structure as a binary head
        and concentration file. The value for the text variable written to the
        file is CIM. Immobile domain concentrations will be written to this
        file at the same interval as mobile domain concentrations are saved, as
        specified in the GWT Model Output Control file.
    columns: (int, optional), default is 10
        number of columns for writing data.
    width: (int, optional), default is 10
        width for writing each number.
    digits: (int, optional), default is 7
        number of digits to use for writing a number.
    format: (str, optional) default exponential
        One of {"exponential", "fixed", "general", "scientific"}.
    validate: {True, False}
        Flag to indicate whether the package should be validated upon
        initialization. This raises a ValidationError if package input is
        provided in the wrong manner. Defaults to True.
    """

    _pkg_id = "ist"
    _template = Package._initialize_template(_pkg_id)
    _grid_data = {
        "initial_immobile_concentration": np.float64,
        "immobile_porosity": np.float64,
        "mobile_immobile_mass_transfer_rate": np.float64,
        "decay": np.float64,
        "decay_sorbed": np.float64,
        "bulk_density": np.float64,
        "distribution_coefficient": np.float64,
    }

    _keyword_map = {
        "initial_immobile_concentration": "cim",
        "immobile_porosity": "thetaim",
        "mobile_immobile_mass_transfer_rate": "zetaim",
        "decay": "decay",
        "decay_sorbed": "decay_sorbed",
        "bulk_density": "bulk_density",
        "distribution_coefficient": "distcoef",
    }

    _init_schemata = {
        "initial_immobile_concentration": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
        ],
        "immobile_porosity": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
        ],
        "mobile_immobile_mass_transfer_rate": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
        ],
        "decay": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
        ],
        "decay_sorbed": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
        ],
        "bulk_density": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
        ],
        "distribution_coefficient": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
        ],
    }

    _write_schemata = {
        "initial_immobile_concentration": [
            AllValueSchema(">", 0.0),
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ],
        "immobile_porosity": [
            AllValueSchema(">=", 0.0),
            AllValueSchema("<", 1.0),
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ],
        "mobile_immobile_mass_transfer_rate": [
            AllValueSchema(">=", 0.0),
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ],
        "decay": [IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0))],
        "decay_sorbed": [
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0))
        ],
        "bulk_density": [
            AllValueSchema(">", 0.0),
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ],
        "distribution_coefficient": [
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0))
        ],
    }

    def __init__(
        self,
        initial_immobile_concentration,
        immobile_porosity,
        mobile_immobile_mass_transfer_rate,
        decay=None,
        decay_sorbed=None,
        bulk_density=None,
        distribution_coefficient=None,
        save_flows: bool = None,
        budgetbinfile: str = None,
        budgetcsvfile: str = None,
        sorption: bool = False,
        first_order_decay: bool = False,
        zero_order_decay: bool = False,
        cimfile: str = "cim.dat",
        columns: int = 7,
        width: int = 10,
        digits: int = 7,
        format: str = "EXPONENTIAL",
        validate: bool = True,
    ):
        # is True fails on a np.bool_ True.
        if sorption:
            if bulk_density is None or distribution_coefficient is None:
                raise ValueError(
                    "if sorption is active, a bulk density and distribution "
                    "coefficient must be provided.",
                )
        if first_order_decay or zero_order_decay:
            if decay is None:
                raise ValueError(
                    "if first_order_decay is active or if zero_order_decay is "
                    "active, decay must be provided.",
                )
            if sorption:
                if decay_sorbed is None:
                    raise ValueError(
                        "if first_order_decay is active or if zero_order_decay "
                        "is active, and sorption is active, decay_sorbed must be "
                        "provided.",
                    )

        super().__init__(locals())
        self.dataset["initial_immobile_concentration"] = initial_immobile_concentration
        self.dataset[
            "mobile_immobile_mass_transfer_rate"
        ] = mobile_immobile_mass_transfer_rate
        self.dataset["immobile_porosity"] = immobile_porosity
        self.dataset["decay"] = decay
        self.dataset["decay_sorbed"] = decay_sorbed
        self.dataset["bulk_density"] = bulk_density
        self.dataset["distribution_coefficient"] = distribution_coefficient
        self.dataset["save_flows"] = save_flows
        self.dataset["budgetfile"] = budgetbinfile
        self.dataset["budgetcsvfile"] = budgetcsvfile
        self.dataset["sorption"] = sorption
        self.dataset["first_order_decay"] = first_order_decay
        self.dataset["zero_order_decay"] = zero_order_decay
        self.dataset["cimfile"] = cimfile
        self.dataset["columns "] = columns
        self.dataset["width"] = width
        self.dataset["digits"] = digits
        self.dataset["format"] = format
        self._validate_init_schemata(validate)
