import warnings

import numpy as np

from imod.mf6.pkgbase import Package
from imod.mf6.validation import PKG_DIMS_SCHEMA
from imod.schemata import (
    DTypeSchema,
    IdentityNoDataSchema,
    IndexesSchema,
    OtherCoordsSchema,
)


class InitialConditions(Package):
    """
    Initial Conditions (IC) Package information is read from the file that is
    specified by "IC6" as the file type. Only one IC Package can be specified
    for a GWF model.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=46

    Parameters
    ----------
    head: array of floats (xr.DataArray)
        for backwards compatibility this argument is maintained, but please use
        the start-argument instead.
    start: array of floats (xr.DataArray)
        is the initial (starting) head or concentration—that is, the simulation's
        initial state.
        STRT must be specified for all simulations, including steady-state simulations.
        One value is read for every model cell. For
        simulations in which the first stress period is steady state, the values
        used for STRT generally do not affect the simulation (exceptions may
        occur if cells go dry and (or) rewet). The execution time, however, will
        be less if STRT includes hydraulic heads that are close to the
        steadystate solution. A head value lower than the cell bottom can be
        provided if a cell should start as dry. (strt)
    """

    _pkg_id = "ic"
    _grid_data = {"head": np.float64}
    _keyword_map = {"head": "strt"}
    _template = Package._initialize_template(_pkg_id)

    _init_schemata = {
        "start": [
            DTypeSchema(np.floating),
            IndexesSchema(),
            PKG_DIMS_SCHEMA,
        ],
    }
    _write_schemata = {
        "start": [
            IdentityNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
            OtherCoordsSchema("idomain"),
        ],
    }

    _grid_data = {"start": np.float64}
    _keyword_map = {"start": "strt"}
    _template = Package._initialize_template(_pkg_id)

    def __init__(self, start=None, head=None):

        super().__init__(locals())
        if start is None:
            start = head
            warnings.warn(
                'The keyword argument "head" is deprecated. Please use the start argument.',
                DeprecationWarning,
            )
            if head is None:
                raise ValueError("start and head arguments cannot both be None")
        else:
            if head is not None:
                raise ValueError("start and head arguments cannot both be defined")

        self.dataset["start"] = start

        self._validate_at_init()

    def render(self, directory, pkgname, globaltimes, binary):
        d = {}
        icdirectory = directory / "ic"
        d["layered"], d["strt"] = self._compose_values(
            self["start"], icdirectory, "strt", binary=binary
        )
        return self._template.render(d)
