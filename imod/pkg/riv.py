class River(xr.Dataset):
    _pkg_id = "riv"
    def __init__(self, stage, cond, bot, conc, dens):
        super(__class__, self).__init__()