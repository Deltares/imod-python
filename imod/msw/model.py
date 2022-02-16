import collections
import jinja2
from copy import copy
from pathlib import Path
from imod.msw.timeutil import to_metaswap_timeformat

DEFAULT_SETTINGS = dict(
    vegetation_mdl=1,
    evapotranspiration_mdl=1,
    saltstress_mdl=0,
    surfacewater_mdl=0,
    infilimsat_opt=0,
    netcdf_per=0,
    postmsw_opt=0,
    dtgw=1.0,
    dtsw=1.0,
    ipstep=2,
    nxlvage_dim=366,
    co2=404.32,
    fact_beta2=1.0,
    rcsoil=0.15,
    iterur1=3,
    iterur2=5,
    tdbgsm=91.0,
    tdedsm=270.0,
    clocktime=0,
)


class Model(collections.UserDict):
    def __setitem__(self, key, value):
        # TODO: Add packagecheck
        super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v


class MetaSwapModel(Model):
    """
    Contains data and writes consistent model input files

    Parameters
    ----------
    unsaturated_database: Path-like or str
        Path to the MetaSWAP soil physical database folder.


    STEPS
    *****
    - Get unsat_svat_path
    - create time discretization
    - get reference time (idbg, iybg)
    - render para_sim.inp
    """

    _pkg_id = "model"

    _template = jinja2.Template(
        """{%- for setting, value in settings.items()%}
        {{setting}} = {{value}}
        {%- endfor%}
        """
    )

    def __init__(self, unsaturated_database):
        super().__init__()

        self.simulation_settings = copy(DEFAULT_SETTINGS)
        self.simulation_settings[
            "unsat_svat_path"
        ] = self._render_unsaturated_database_path(unsaturated_database)

    def _render_unsaturated_database_path(self, unsaturated_database):
        # Force to Path object
        unsaturated_database = Path(unsaturated_database)

        # Render to string for MetaSWAP
        if unsaturated_database.is_absolute():
            return f'"{unsaturated_database}\\"'
        else:
            # TODO: Test if this is how MetaSWAP accepts relative paths
            return f'"${unsaturated_database}\\"'

    def _get_starttime(self):
        """
        Loop over all packages to get the minimum time.
        """

        starttimes = []

        for pkgname in self:
            ds = self[pkgname].dataset
            if "time" in ds.coords:
                starttimes.append(ds["time"].min().values)

        starttime = min(starttimes)

        year, time_since_start_year = to_metaswap_timeformat([starttime])

        year = int(year.values)
        time_since_start_year = float(time_since_start_year.values)

        return year, time_since_start_year

    def write(self, directory):
        """
        Write packages
        """

        year, time_since_start_year = self._get_starttime()

        self.simulation_settings["iybg"] = year
        self.simulation_settings["tdbg"] = time_since_start_year

        # TODO:
        # - Output control of all the svat_csv options
        # - Output control of the idf

        # write package contents
        for pkgname in self:
            self[pkgname].write(directory)
