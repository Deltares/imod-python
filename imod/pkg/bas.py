class Basic(xr.Dataset):
    def __init__(
        self,
        ibound,
        icbund,
        top,
        bot,
        shead,
        sconc,
        porosity,
        n_species,
        conc_inactive=1.0e30,
    ):
        super(__class__, self).__init__()
