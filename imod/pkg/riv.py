class River(xr.Dataset):
    def __init__(self, stage, cond, bot, conc, dens):
        super(__class__, self).__init__()