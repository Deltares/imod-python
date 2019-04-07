import jinja2

from imod.pkg.pkgbase import Package


class BasicTransport(Package):
    _mapping = (
        ("icbund", "icbund"),
        ("sconc", "sconc"),
        ("dz", "thickness"),
        ("prsity", "porosity"),
        ("laycon", "layer_type"),
    )

    def __init__(
        icbund,
        sconc,
        porosity,
        n_species=1
        conc_inactive=1.0e30,
        minimum_active_thickness=0.01,
        ifmtcn,
        ifmtnp,
        ifmtrf,
        ifmtdp,
        savunc,
        timprs,
        nprobs,
        obs,
        chkmas,
        nprmas,
        nstep,
        tsmult,
        tslength,
        dt0,
        mxstrn,
        ttsmult,  # belongs to FiniteDifferenceAdvection
        ttsmax,  # belongs to FiniteDifferenceAdvection
    ):
    super(__class__, self).__init__()


