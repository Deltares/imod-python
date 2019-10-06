import numpy as np

from imod.mf6.pkgbase import Package


class InitialConditions(Package):
    __slots__ = ("head",)
    _pkg_id = "ic"
    _binary_data = {"head": np.float64}

    def __init__(self, head):
        super(__class__, self).__init__()
        self["head"] = head
        self._initialize_template()

    def render(self, directory, pkgname, *args, **kwargs):
        d = {}
        icdirectory = directory / "ic"
        d["layered"], d["strt"] = self._compose_values(
            self["head"], icdirectory, "head"
        )
        return self._template.render(d)
