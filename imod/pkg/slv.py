import jinja2
import xarray as xr


class PreconditionedConjugateGradientSolver(xr.Dataset):
    _template = """
    [pcg]
    mxiter = {max_iter}
    iter1 = {inner_iter}
    npcond = 1
    hclose = {hclose}
    rclose = {rclose}
    relax = {relax}
    iprpcg = 
    mutpcg = 
    damp = {damp}
    """

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
        return self._template.format(d)

    def _pkgcheck(self):
        if not self["hclose"] > 0:
            raise ValueError


class GeneralizedConjugateGradientSolver(xr.Dataset):
    _template = """
    [gcg]
    mxiter = {max_iter}
    iter1 = {inner_iter}
    isolve = {preconditioner}
    ncrs = {lump_dispersion}
    cclose = {cclose}
    iprgcg = {print_iteration}
    """

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
        print_iteration=0,
    ):
        super(__class__, self).__init__()
        self["max_iter"] = max_iter
        self["inner_iter"] = inner_iter
        self["cclose"] = cclose
        self["preconditioner"] = preconditioner
        self["lump_dispersion"] = lump_dispersion
        self["print_iteration"] = print_iteration

    def _replace_keyword(self, d, key):
        keyword = d[key][()]  # Get value from 0d np.array
        value = self._keywords[key][keyword]
        d[key] = value

    def _render(self):
        d = {k: v.values for k, v in self.data_vars.items()}
        self._replace_keyword(d, "preconditioner")
        self._replace_keyword(d, "lump_dispersion")
        return self._template.format(d)


class ParallelKrylovFlowSolver(xr.Dataset):
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


class ParallelKrylovTransportSolver(xr.Dataset):
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

