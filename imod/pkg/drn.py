class Drainage(xr.Dataset):
    _pkg_id = "drn"
    def __init__(self, elev, cond):
        super(__class__, self).__init__()
