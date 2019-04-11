import jinja2

from imod.pkg.pkgbase import Package


class BasicTransport(Package):
    _pkg_id = "btn"

    _mapping = (
        ("icbund", "icbund"),
        ("sconc", "sconc"),
        ("dz", "thickness"),
        ("prsity", "porosity"),
        ("laycon", "layer_type"),
    )

    def __init__(
        icbund,
        starting_concentration,
        porosity=0.3,
        n_species=1,
        conc_inactive=1.0e30,
        minimum_active_thickness=0.01,
    ):
        super(__class__, self).__init__()


