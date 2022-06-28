from imod.mf6.pkgbase import Package


class Buoyancy(Package):
    """
        Buoyancy package. This package must be included when performing variable
        density simulation.
        This package is to be used as follows: first initialize the package, and then
        call a function to add dependencies on species concentration, once for
        each species that affects density. The dependency of density on each species is linear.

        Parameters
        ----------
    hhformulation_rhs:  bool, optional
        use the variable-density hydraulic head formulation and add off-diagonal
        terms to the right-hand. This option will prevent the BUY Package from
        adding asymmetric terms to the flow matrix.
    denseref: real, optional
        fluid reference density used in the equation of state. This value is set to
        1000 if not specified as an option.
    densityfile:
        name of the binary output file to write density information. The density
        file has the same format as the head file. Density values will be written to
        the density file whenever heads are written to the binary head file. The
        settings for controlling head output are contained in the Output Control
        option.

    Examples
        --------
        Initialize the Buoyancy package. We use an option to keep the flow matrix symmetric,
        and a freshwater density of 996 g/l. We also specify an output file for density.  :

        >>> buy = imod.mf6.Buoyancy(
             hhformulation_rhs=True, denseref=996, densityfile="density_out.dat"
        )

        Second, we add dependencies on one or more species. We must provide the reference
        concentration ( this is the concentration at which the freshwater density provided above
        was measured) and also the derivative of density to species concentration. This slope is assumed to
        be constant across the concentration range. For each dependency we must also provide the name of the species,
        and of the transport model that governs this species.

        >>> buy.add_species_dependency(0.7, 4, "gwt-1", "salinity")
        >>> buy.add_species_dependency(-0.375, 25, "gwt-2", "temperature")

    """

    _pkg_id = "buy"
    _template = Package._initialize_template(_pkg_id)
    _metadata_dict = {}

    def __init__(
        self,
        hhformulation_rhs: bool = None,
        denseref: float = None,
        densityfile: str = None,
    ):
        super().__init__(locals())
        self.dataset["hhformulation_rhs"] = hhformulation_rhs
        self.dataset["denseref"] = denseref
        self.dataset["densityfile"] = densityfile
        self.dataset["nrhospecies"] = 0

        self.dependencies = []
        self._pkgcheck()

    def add_species_dependency(
        self,
        d_rho_dc: float,
        reference_concentration: float,
        modelname: str,
        speciesname: str,
    ):
        self.dataset["nrhospecies"] += 1
        self.dependencies.append(
            {
                "ispec": self.dataset["nrhospecies"].values[()],
                "modelname": modelname,
                "speciesname": speciesname,
                "slope": d_rho_dc,
                "reference_conc": reference_concentration,
            }
        )

    def render(self, directory, pkgname, globaltimes, binary):
        d = {}
        d["dependencies"] = self.dependencies
        for varname in ["hhformulation_rhs", "denseref", "densityfile", "nrhospecies"]:
            if self._valid(self.dataset[varname].values[()]):
                d[varname] = self.dataset[varname].values[()]
        d["nrhospecies"] = self.dataset["nrhospecies"].values[()]


        return self._template.render(d)
