class PreconditionedConjugateGradientSolver(xr.Dataset):
    def __init__(self, max_iter, inner_iter, rclose, hclose, relax=0.98, damp=1.0):
        super(__class__, self).__init__()
        self["max_iter"] = mxiter
        self["inner_iter"] = innerit
        self["rclose"] = rclose
        self["hclose"] = hclose

    def _pkgcheck(self):
        if not self["hclose"] > 0:
            raise ValueError


class GeneralizedConjugateGradientSolver(xr.Dataset):
    def __init__(self, max_iter, inner_iter, cclose=1.0e-6, preconditioner="MIC", lump_dispersion=True, print_iteration=False):
        super(__class__, self).__init__()


class ParallelKrylovFlowSolver(xr.Dataset):
    def __init__(self, max_iter, inner_iter, hclose, rclose, relax, partition="uniform", solver="pcg", preconditioner="ilu", deflate=False, debug=False):
        super(__class__, self).__init__()


class ParallelKrylovTransportSolver(xr.Dataset):
    def __init__(self, max_iter, inner_iter, cclose=1.0e-6, relax=0.98, partition="uniform", solver="bicgstab", preconditioner="ilu", debug=False):
        super(__class__, self).__init__()

