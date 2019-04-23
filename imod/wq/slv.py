from imod.wq.pkgbase import Package


class PreconditionedConjugateGradientSolver(Package):
    """
    The Preconditioned Conjugate Gradient Solver is used to solve the finite
    difference equations in each step of a MODFLOW stress period.

    Parameters
    ----------
    max_iter: int
        is the maximum number of outer iterations - that is, calss to the
        solutions routine (MXITER). For a linear problem max_iter should be 1, unless
        more than 50 inner iterations are required, when max_iter could be as
        large as 10. A larger number (generally less than 100) is required for a
        nonlinear problem.
    inner_iter: int
        is the number of inner iterations (iter1). For nonlinear problems,
        inner_iter usually ranges from 10 to 30; a value of 30 will be
        sufficient for most linear problems.
    rclose: float
        is the residual criterion for convergence, in units of cubic length per
        time. The units for length and time are the same as established for all
        model data. When the maximum absolute value of the residual at all nodes
        during an iteration is less than or equal to RCLOSE, and the criterion
        for HCLOSE is also satisfied (see below), iteration stops.
    hclose: float
        is the head change criterion for convergence, in units of length. When
        the maximum absolute value of head change from all nodes during an
        iteration is less than or equal to HCLOSE, and the criterion for RCLOSE
        is also satisfied (see above), iteration stops.
    relax: float, optional
        is the relaxation parameter used. Usually, RELAX = 1.0, but for some
        problems a value of 0.99, 0.98, or 0.97 will reduce the number of
        iterations required for convergence.
        Default value: 0.98
    damp: float, optional
        is the damping factor. It is typically set equal to one, which indicates
        no damping. A value less than 1 and greater than 0 causes damping. DAMP
        does not affect inner iterations; instead, it affects outer iterations.
        Default value: 1.0
    """

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
        "    damp = {damp}"
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
    """
    The Generalized Conjugate Gradient Solver solves the matrix equations
    resulting from the implicit solution of the transport equation.

    Parameters
    ----------
    max_iter: int
        is the maximum number of outer iterations (MXITER); it should be set to an
        integer greater than one (1) only when a nonlinear sorption isotherm is
        included in simulation.
    iter1: int
        is the maximum number of inner iterations (iter1); a value of 30-50
        should be adequate for most problems.
    isolve: int
        is the type of preconditioners to be used with the Lanczos/ORTHOMIN
        acceleration scheme:
        isolve = 1: Jacobi
        isolve = 2: SSOR
        isolve = 3: Modified Incomplete Cholesky (MIC)
        (MIC usually converges faster, but it needs significantly more memory)
    ncrs: int
        is an integer flag for treatment of dispersion tensor cross terms:
        ncrs = 0: lump all dispersion cross terms to the right-hand-side
        (approximate but highly efficient).
        ncrs = 1: include full dispersion tensor (memory intensive).
    cclose: float
        is the convergence criterion in terms of relative concentration; a real
        value between 10-4 and 10-6 is generally adequate.    
    """

    _pkg_id = "gcg"
    _template = (
        "[gcg]\n"
        "    mxiter = {max_iter}\n"
        "    iter1 = {inner_iter}\n"
        "    isolve = {preconditioner}\n"
        "    ncrs = {lump_dispersion}\n"
        "    cclose = {cclose}\n"
        "    iprgcg = 0"
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
    """
    The Parallel Krylov Flow Solver is used for parallel solving of the flow
    model.

    Parameters
    ----------
    max_iter: int
        is the maximum number of outer iterations (MXITER); it should be set to an
        integer greater than one (1) only when a nonlinear sorption isotherm is
        included in simulation.
    inner_iter: int
        is the maximum number of inner iterations (INNERIT); a value of 30-50
        should be adequate for most problems.
    rclose: float
        is the residual criterion for convergence (RCLOSEPKS), in units of cubic length per
        time. The units for length and time are the same as established for all
        model data. When the maximum absolute value of the residual at all nodes
        during an iteration is less than or equal to RCLOSE, and the criterion
        for HCLOSE is also satisfied (see below), iteration stops.
    hclose: float
        is the head change criterion for convergence (HCLOSEPKS), in units of length. When
        the maximum absolute value of head change from all nodes during an
        iteration is less than or equal to HCLOSE, and the criterion for RCLOSE
        is also satisfied (see above), iteration stops.    
    relax: float
        is the relaxation parameter used. Usually, RELAX = 1.0, but for some
        problems a value of 0.99, 0.98, or 0.97 will reduce the number of
        iterations required for convergence.
    partition: {"uniform"}, optional
        Partitioning option (PARTOPT).
        Default value: "uniform"
    solver: {"pcg"}, optional
        Flag indicating the linear solver to be used (ISOLVER).
        Default value: "pcg"
    preconditioner: {"ilu"}, optional
        Flag inicating the preconditioner to be used (NPC).
        Devault value: "ilu"
    deflate: {True, False}, optional
        Flag for deflation preconditioner.
        Default value: False
    debug: {True, False}, optional
        Debug option.
        Default value: False

    """

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
    """
    The Parallel Krylov Transport Solver is used for parallel solving of the
    transport model.

    Parameters
    ----------
    max_iter: int
        is the maximum number of outer iterations (MXITER); it should be set to an
        integer greater than one (1) only when a nonlinear sorption isotherm is
        included in simulation.    
    inner_iter: int
        is the maximum number of inner iterations (INNERIT); a value of 30-50
        should be adequate for most problems.
    cclose: float, optional
        is the convergence criterion in terms of relative concentration; a real
        value between 10-4 and 10-6 is generally adequate.    
        Default value: 1.0e-6
    relax: float, optional
        is the relaxation parameter used. Usually, RELAX = 1.0, but for some
        problems a value of 0.99, 0.98, or 0.97 will reduce the number of
        iterations required for convergence.
        Default value: 0.98
    partition: {"uniform"}, optional
        Partitioning option (PARTOPT).
        Default value: "uniform"
    solver: {"bicgstab"}, optional
        Flag indicating the linear solver to be used (ISOLVER).
        Default value: "bicgstab"
    preconditioner: {"ilu"}, optional
        Flag inicating the preconditioner to be used (NPC).
        Devault value: "ilu"
    debug: {True, False}, optional
        Debug option.
        Default value: False

    """

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
