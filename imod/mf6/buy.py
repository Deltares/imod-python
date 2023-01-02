from typing import Sequence

import numpy as np
import xarray as xr

from imod.mf6.pkgbase import Package
from imod.schemata import DTypeSchema


def assign_index(arg):
    if isinstance(arg, xr.DataArray):
        arg = arg.values
    elif not isinstance(arg, (np.ndarray, list, tuple)):
        raise TypeError("should be a tuple, list, or numpy array")

    arr = np.array(arg)
    if arr.ndim != 1:
        raise ValueError("should be 1D")

    return xr.DataArray(arr, dims=("index",))


class Buoyancy(Package):
    """
    Buoyancy package. This package must be included when performing variable
    density simulation.

    Note that ``reference_density`` is a single value, but
    ``density_concentration_slope``, ``reference_concentration`` and
    ``modelname`` require an entry for every active species. Please refer to
    the examples.

    Parameters
    ----------
    reference_density: float,
        fluid reference density used in the equation of state.
    density_concentration_slope: sequence of floats
        Slope of the (linear) density concentration line used in the density
        equation of state.
    reference_concentration: sequence of floats
        Reference concentration used in the density equation of
        state.
    modelname: sequence of strings,
        Name of the GroundwaterTransport (GWT) model used for the
        concentrations.
    species: sequence of str,
        Name of the species used to calculate a density value.
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
    ...     density_concentration_slope=[0.025],
    ...     reference_concentration=[0.0],
    ...     modelname=["gwt-1"],
    ...     species=["salinity"],
    ... )

    Multiple species can be specified by presenting multiple values with an
    associated species coordinate. Two species, called "c1" and "c2", simulated
    by the GWT models "gwt-1" and "gwt-2" are specified as:

    >>> coords = {"species": ["c1", "c2"]}
    >>> buy = imod.mf6.Buoyance(
    ...     reference_density=1000.0,
    ...     density_concentration_slope=[0.025, 0.01],
    ...     reference_concentration=[0.0, 0.0],
    ...     modelname=["gwt-1", "gwt-2"],
    ...     species=["c1", "c2"],
    ... )
    """

    _pkg_id = "buy"
    _template = Package._initialize_template(_pkg_id)
    _metadata_dict = {}

    _init_schemata = {
        "reference_density": [DTypeSchema(np.floating)],
        "density_concentration_slope": [DTypeSchema(np.floating)],
        "reference_concentration": [DTypeSchema(np.floating)],
    }

    def __init__(
        self,
        reference_density: float,
        density_concentration_slope: Sequence[float],
        reference_concentration: Sequence[float],
        modelname: Sequence[str],
        species: Sequence[str],
        hhformulation_rhs: bool = False,
        densityfile: str = None,
    ):
        super().__init__(locals())
        self.dataset["reference_density"] = reference_density
        # Assign a shared index: this also forces equal lenghts
        self.dataset["density_concentration_slope"] = assign_index(
            density_concentration_slope
        )
        self.dataset["reference_concentration"] = assign_index(reference_concentration)
        self.dataset["modelname"] = assign_index(modelname)
        self.dataset["species"] = assign_index(species)
        self.dataset["hhformulation_rhs"] = hhformulation_rhs
        self.dataset["densityfile"] = densityfile

        self.dependencies = []
        self._validate_at_init()

    def render(self, directory, pkgname, globaltimes, binary):
        ds = self.dataset
        packagedata = []

        for i, (a, b, c, d) in enumerate(
            zip(
                ds["density_concentration_slope"].values,
                ds["reference_concentration"].values,
                ds["modelname"].values,
                ds["species"].values,
            )
        ):
            packagedata.append((i + 1, a, b, c, d))

        d = {
            "nrhospecies": self.dataset["species"].size,
            "packagedata": packagedata,
        }

        for varname in ["hhformulation_rhs", "reference_density", "densityfile"]:
            value = self.dataset[varname].values[()]
            if self._valid(value):
                d[varname] = value

        return self._template.render(d)
