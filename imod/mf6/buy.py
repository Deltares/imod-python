from imod.mf6.pkgbase import Package


class Buoyancy(Package):
    """
        Buoyancy package. This package must be included when performing variable
        density simulation.

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
        if self.dataset["hhformulation_rhs"].values[()] is not None:
            d["hhformulation_rhs"] = self.dataset["hhformulation_rhs"].values[()]
        if self.dataset["denseref"].values[()] is not None:
            d["denseref"] = self.dataset["denseref"].values[()]
        if self.dataset["densityfile"].values[()] is not None:
            d["densityfile"] = self.dataset["densityfile"].values[()]
        d["nrhospecies"] = self.dataset["nrhospecies"].values[()]

        return self._template.render(d)
