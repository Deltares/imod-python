import xarray as xr

from imod.mf6.pkgbase import Package


class Buoyancy(Package):
    """
    Buoyancy package. This package must be included when performing variable
    density simulation.

    Note that ``reference_density`` is a single value, but
    ``density_concentration_slope``, ``reference_concentration`` and
    ``modelname`` require an entry for every active species, marked by a
    species coordinate. Please refer to the examples.

    Parameters
    ----------
    reference_density: float,
        fluid reference density used in the equation of state.
    density_concentration_slope: xr.DataArray of floats
        Slope of the (linear) density concentration line used in the density
        equation of state.
    reference_concentration: xr.DataArray of floats
        Reference concentration used in the density equation of
        state.
    modelname: xr.DataArray of strings,
        Name of the GroundwaterTransport (GWT) model used for the
        concentrations.
    hhformulation_rhs: bool, optional.
        use the variable-density hydraulic head formulation and add
        off-diagonal terms to the right-hand. This option will prevent the BUY
        Package from adding asymmetric terms to the flow matrix. Default value
        is ``False``.
    densityfile:
        name of the binary output file to write density information. The density
        file has the same format as the head file. Density values will be written to
        the density file whenever heads are written to the binary head file. The
        settings for controlling head output are contained in the Output Control
        option.

    Examples
    --------

    The buoyancy input for a single species called "salinity", which is
    simulated by a GWT model called "gwt-1" are specified as follows:

    >>> buy = imod.mf6.Buoyance(
    ...     reference_density=1000.0,
    ...     density_concentration_slope=xr.DataArray(0.025, coords={"species": "salinity"}),
    ...     reference_concentration=xr.DataArray(0.0, coords={"species": "salinity"}),
    ...     modelname=xr.DataArray("gwt-1", coords={"species": "salinity"})
    ...     )
    ... )

    Multiple species can be specified by presenting multiple values with an
    associated species coordinate. Two species, called "c1" and "c2", simulated
    by the GWT models "gwt-1" and "gwt-2" are specified as:

    >>> coords = {"species": ["c1", "c2"]}
    >>> buy = imod.mf6.Buoyance(
    ...     reference_density=1000.0,
    ...     density_concentration_slope=xr.DataArray([0.025, 0.01], coords=coords)
    ...     reference_concentration=xr.DataArray([0.0, 0.0], coords=coords)
    ...     modelname=xr.DataArray(["gwt-1", "gwt-2"], coords=coords)
    ...     )
    ... )
    """

    _pkg_id = "buy"
    _template = Package._initialize_template(_pkg_id)
    _metadata_dict = {}

    def __init__(
        self,
        reference_density: float,
        density_concentration_slope: xr.DataArray,
        reference_concentration: xr.DataArray,
        modelname: xr.DataArray,
        hhformulation_rhs: bool = False,
        densityfile: str = None,
    ):
        super().__init__(locals())
        self.dataset["reference_density"] = reference_density
        self.dataset["density_concentration_slope"] = density_concentration_slope
        self.dataset["reference_concentration"] = reference_concentration
        self.dataset["modelname"] = modelname
        self.dataset["hhformulation_rhs"] = hhformulation_rhs
        self.dataset["densityfile"] = densityfile

        self.dependencies = []
        self._pkgcheck()

    def render(self, directory, pkgname, globaltimes, binary):
        # Ensure species is an iterable dimensions in case of a single species.
        ds = self.dataset.copy()
        if "species" not in ds.dims:
            ds = ds.expand_dims("species")

        packagedata = []
        variables = (
            "density_concentration_slope",
            "reference_concentration",
            "modelname",
        )  # This order matters!
        for i, species in enumerate(ds["species"].values):
            species_ds = ds.sel(species=species)
            values = []
            for var in variables:
                value = species_ds[var].values[()]
                if not self._valid(value):
                    raise ValueError(f"Invalid value of {var} for {species}: {value}")
                values.append(value)
            packagedata.append((i + 1, *values, species))

        d = {
            "nrhospecies": self.dataset.coords["species"].size,
            "packagedata": packagedata,
        }

        for varname in ["hhformulation_rhs", "reference_density", "densityfile"]:
            value = self.dataset[varname].values[()]
            if self._valid(value):
                d[varname] = value

        return self._template.render(d)
