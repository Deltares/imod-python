from imod.mf6.pkgbase import Package


class InitialConditions(Package):
    _pkg_id = "ic"
    _binary_data = ("head",)

    def __init__(self, head):
        super(__class__, self).__init__()
        self["head"] = head
        self._initialize_template()

    def render(self, directory, pkgname, *args, **kwargs):
        d = {}
        d["layered"], d["strt"] = self._compose_values(self["head"], directory)
        return self._template.render(d)