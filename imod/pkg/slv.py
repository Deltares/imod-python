import jinja2

from imod.pkg.pkgbase import Package


class PreconditionedConjugateGradientSolver(Package):
    _pkg_id = "pcg"
    _template = (
        "[pcg]\n"
        "    mxiter = {max_iter}\n"
        "    iter1 = {inner_iter}\n"
        "    npcond = 1\n"
        "    hclose = {hclose}\n"
        "    rclose = {rclose}\n"
        "    relax = {relax}\n"
        "    iprpcg = 1\n"
        "    mutpcg = 0\n"
        "    damp = {damp}\n"
    )

    def __init__(self, max_iter, inner_iter, rclose, hclose, relax=0.98, damp=1.0):
        super(__class__, self).__init__()
        self["max_iter"] = max_iter
        self["inner_iter"] = inner_iter
        self["rclose"] = rclose
        self["hclose"] = hclose
        self["relax"] = relax
        self["damp"] = damp

    def _render(self):
        d = {k: v.values for k, v in self.data_vars.items()}
        return self._template.format(**d)

    def _pkgcheck(self):
        if not self["hclose"] > 0:
            raise ValueError


class GeneralizedConjugateGradientSolver(Package):
    _pkg_id = "gcg"
    _template = (
        "[gcg]\n"
        "    mxiter = {max_iter}\n"
        "    iter1 = {inner_iter}\n"
        "    isolve = {preconditioner}\n"
        "    ncrs = {lump_dispersion}\n"
        "    cclose = {cclose}\n"
        "    iprgcg = 0\n"
    )

    _keywords = {
        "preconditioner": {"jacobi": 1, "ssor": 2, "mic": 3},
        "lump_dispersion": {True: 0, False: 1},
    }

    def __init__(
        self,
        max_iter,
        inner_iter,
        cclose=1.0e-6,
        preconditioner="mic",
        lump_dispersion=True,
    ):
        super(__class__, self).__init__()
        self["max_iter"] = max_iter
        self["inner_iter"] = inner_iter
        self["cclose"] = cclose
        self["preconditioner"] = preconditioner
        self["lump_dispersion"] = lump_dispersion


class ParallelKrylovFlowSolver(Package):
    _pkg_id = "pksf"

    def __init__(
        self,
        max_iter,
        inner_iter,
        hclose,
        rclose,
        relax,
        partition="uniform",
        solver="pcg",
        preconditioner="ilu",
        deflate=False,
        debug=False,
    ):
        super(__class__, self).__init__()


class ParallelKrylovTransportSolver(Package):
    _pkg_id = "pkst"

    def __init__(
        self,
        max_iter,
        inner_iter,
        cclose=1.0e-6,
        relax=0.98,
        partition="uniform",
        solver="bicgstab",
        preconditioner="ilu",
        debug=False,
    ):
        super(__class__, self).__init__()
