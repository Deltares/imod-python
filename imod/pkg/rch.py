from imod.pkg.pkgbase import BoundaryCondition


class RechargeTopLayer(BoundaryCondition):
    _pkg_id = "rch"
    def __init__(self, rate, conc, save_budget=False):
        super(__class__, self).__init__()

class RechargeLayers(BoundaryCondition):
    _pkg_id = "rch"
    def __init__(self, rate, conc, save_budget=False):
        super(__class__, self).__init__()

class RechargeHighestActive(BoundaryCondition):
    _pkg_id = "rch"
    def __init__(self, rate, conc, save_budget=False):
        super(__class__, self).__init__()