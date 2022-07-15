import xarray as xr

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
        reference_density: float,
        modelname: xr.DataArray,
        reference_concentration: xr.DataArray,
        density_concentration_slope: xr.DataArray,
        hhformulation_rhs: bool = False,
        densityfile: str = None,
    ):
        super().__init__(locals())
        self.dataset["reference_density"] = reference_density
        self.dataset["modelname"] = modelname
        self.dataset["reference_concentration"] = reference_concentration
        self.dataset["density_concentration_slope"] = density_concentration_slope
        self.dataset["hhformulation_rhs"] = hhformulation_rhs
        self.dataset["densityfile"] = densityfile

        self.dependencies = []
        self._pkgcheck()

    def render(self, directory, pkgname, globaltimes, binary):
        # Ensure modelname and species are iterable dimensions in case of a
        # single species and/or single modelname.
        ds = self.dataset.copy()
        if "species" not in ds.dims:
            ds = ds.expand_dims("species")

        packagedata = []
        for i, species in enumerate(ds["species"].values):
            species_ds = ds.sel(species=species)
            variables = (
                "density_concentration_slope",
                "reference_concentration",
                "modelname",
            )
            values = []
            for var in variables:
                value = species_ds[var].values[()]
                if not self._valid(value):
                    raise ValueError(f"Invalid value of {var} for {species}: {value}")
                values.append(values)
            packagedata.append((i + 1, *values, species))

        d = {
            "nrhospecies": self.dataset.coords["species"].size,
            "packagedata": packagedata,
        }

        for varname in ["hhformulation_rhs", "reference_density", "densityfile"]:
            if self._valid(self.dataset[varname].values[()]):
                d[varname] = self.dataset[varname].values[()]

        return self._template.render(d)
