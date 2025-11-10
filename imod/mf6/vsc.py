from typing import Literal, Optional, Sequence, TypeAlias

import numpy as np

from imod.mf6.package import Package
from imod.mf6.utilities.dataset import assign_index
from imod.schemata import DTypeSchema

ThermalFormulationOption: TypeAlias = Literal["linear", "nonlinear"]


class Viscosity(Package):
    """
    The Viscosity Package in MODFLOW 6 is used to account for the effects of
    solute concentration or temperature on fluid viscosity and thereby their
    effects on hydraulic conductivity and stress-package conductance. If the
    Viscosity package is used, the Groundwater Transport process must also be
    used. In addition, the flow and transport models must be part of the same
    simulation. The Viscosity package will adjust the conductances in the model
    based on the solute concentrations.

    Parameters
    ----------

    reference_viscosity: float
        Fluid reference viscosity used in the equation of state.
    viscosity_concentration_slope: sequence of floats
        Slope of the (linear) viscosity concentration line used in the viscosity
        equation of state. This value will be used when ``thermal_formulation``
        is equal to ``"linear"`` (the default) in the OPTIONS block. When
        ``thermal_formulation`` is set to ``"nonlinear"``, a value for DVISCDC
        must be specified though it is not used.
    reference_concentration: sequence of floats
        Reference concentration used in the viscosity equation of state.
    modelname: sequence of strings
        Name of the GroundwaterTransport (GWT) model used for the
        concentrations.
    species: sequence of str
        Name of the species used to calculate a viscosity value.
    temperature_species_name: str
        Name of the species to be interpreted as temperature. This species is
        used to calculate the temperature-dependent viscosity, using all
        ``thermal_`` arguments.
    thermal_formulation: str, optional
        The thermal formulation to use for the temperature-dependent viscosity.
    thermal_a2: float, optional
        Is an empirical parameter specified by the user for calculating
        viscosity using a nonlinear formulation. If thermal_a2 is not specified,
        a default value of 10.0 is assigned (Voss, 1984).
    thermal_a3: float, optional
        Is an empirical parameter specified by the user for calculating
        viscosity using a nonlinear formulation. If thermal_a3 is not specified,
        a default value of 248.37 is assigned (Voss, 1984).
    thermal_a4: float, optional
        Is an empirical parameter specified by the user for calculating
        viscosity using a nonlinear formulation. If thermal_a4 is not specified,
        a default value of 133.15 is assigned (Voss, 1984).
    viscosityfile: str, optional
        Name of the binary output file to write viscosity information.
    validate: bool, optional
        Flag to indicate whether the package should be validated upon
        initialization. This raises a ValidationError if package input is
        provided in the wrong manner. Defaults to True.

    Examples
    --------

    The viscosity input for a single species called "salinity", which is
    simulated by a GWT model called "gwt-1" are specified as follows:

    >>> vsc = imod.mf6.Viscosity(
    ...     reference_viscosity=8.904E-04,
    ...     viscosity_concentration_slope=[1.92e-6],
    ...     reference_concentration=[0.0],
    ...     modelname=["gwt-1"],
    ...     species=["salinity"],
    ... )

    Multiple species can be specified by presenting multiple values with an
    associated species coordinate. Two species, called "c1" and "c2", simulated
    by the GWT models "gwt-1" and "gwt-2" are specified as:

    >>> coords = {"species": ["c1", "c2"]}
    >>> vsc = imod.mf6.Viscosity(
    ...     reference_viscosity=8.904E-04,
    ...     viscosity_concentration_slope=[1.92e-6, 3.4e-6],
    ...     reference_concentration=[0.0, 0.0],
    ...     modelname=["gwt-1", "gwt-2],
    ...     species=["c1", "c2"],
    ... )

    You can also specify thermal properties, even with a nonlinear thermal
    formulation.

    >>> coords = {"species": ["salinity", "temperature"]}
    >>> vsc = imod.mf6.Viscosity(
    ...     reference_viscosity=8.904E-04,
    ...     viscosity_concentration_slope=[1.92e-6, 0.0],
    ...     reference_concentration=[0.0, 25.0],
    ...     modelname=["gwt-1", "gwt-2"],
    ...     species=["salinity", "temperature"],
    ...     temperature_species_name="temperature",
    ...     thermal_formulation="nonlinear",
    ...     thermal_a2=10.0,
    ...     thermal_a3=248.37,
    ...     thermal_a4=133.15,
    ... )

    """

    _pkg_id = "vsc"
    _template = Package._initialize_template(_pkg_id)

    _init_schemata = {
        "reference_viscosity": [DTypeSchema(np.floating)],
        "viscosity_concentration_slope": [DTypeSchema(np.floating)],
        "reference_concentration": [DTypeSchema(np.floating)],
    }
    _write_schemata = {}

    def __init__(
        self,
        reference_viscosity: float,
        viscosity_concentration_slope: Sequence[float],
        reference_concentration: Sequence[float],
        modelname: Sequence[str],
        species: Sequence[str],
        temperature_species_name: Optional[str] = None,
        thermal_formulation: ThermalFormulationOption = "linear",
        thermal_a2: float = 10.0,
        thermal_a3: float = 248.37,
        thermal_a4: float = 133.15,
        viscosityfile: Optional[str] = None,
        validate: bool = True,
    ):
        dict_dataset = {
            "reference_viscosity": reference_viscosity,
            # Assign a shared index: this also forces equal lengths
            "viscosity_concentration_slope": assign_index(
                viscosity_concentration_slope
            ),
            "reference_concentration": assign_index(reference_concentration),
            "modelname": assign_index(modelname),
            "species": assign_index(species),
            "temperature_species_name": temperature_species_name,
            "thermal_formulation": thermal_formulation,
            "thermal_a2": thermal_a2,
            "thermal_a3": thermal_a3,
            "thermal_a4": thermal_a4,
            "viscosityfile": viscosityfile,
        }
        super().__init__(dict_dataset)
        self._validate_init_schemata(validate)

    def _render(self, directory, pkgname, globaltimes, binary):
        ds = self.dataset
        packagedata = []

        for i, (a, b, c, d) in enumerate(
            zip(
                ds["viscosity_concentration_slope"].values,
                ds["reference_concentration"].values,
                ds["modelname"].values,
                ds["species"].values,
            )
        ):
            packagedata.append((i + 1, a, b, c, d))

        d = {
            "nviscspecies": self.dataset["species"].size,
            "packagedata": packagedata,
        }

        for varname in [
            "temperature_species_name",
            "thermal_formulation",
            "thermal_a2",
            "thermal_a3",
            "thermal_a4",
            "reference_viscosity",
            "viscosityfile",
        ]:
            value = self.dataset[varname].values[()]
            if self._valid(value):
                d[varname] = value

        return self._template.render(d)

    def _update_transport_models(self, new_modelnames: Sequence[str]):
        """
        The names of the transport models can change in some cases, for example
        when partitioning. Use this function to update the names of the
        transport models.
        """
        transport_model_names = self._get_transport_model_names()
        if not len(transport_model_names) == len(new_modelnames):
            raise ValueError("the number of transport models cannot be changed.")
        for modelname, new_modelname in zip(transport_model_names, new_modelnames):
            if modelname not in new_modelname:
                raise ValueError(
                    "new transport model names do not match the old ones. The new names should be equal to the old ones, with a suffix."
                )
        self.dataset["modelname"] = assign_index(new_modelnames)

    def _get_transport_model_names(self) -> list[str]:
        """
        Returns the names of the transport  models used by this buoyancy package.
        """
        return list(self.dataset["modelname"].values)
