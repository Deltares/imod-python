from imod.flow.pkgbase import Package
import jinja2


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

        Default value: 100.0. **Nota bene**: this is aimed at regional modelling.
        For detailed studies (e.g. lab experiments) much smaller values can be
        required.
        Very general rule of thumb: should be less than 10% of smallest cell volume.
    hclose: float
        is the head change criterion for convergence, in units of length. When
        the maximum absolute value of head change from all nodes during an
        iteration is less than or equal to HCLOSE, and the criterion for RCLOSE
        is also satisfied (see above), iteration stops.
        Default value: 1.0e-4. **Nota bene**: This is aimed at regional modelling, `
        for detailed studies (e.g. lab experiments) much smaller values can be
        required.
    relax: float, optional
        is the relaxation parameter used. Usually, RELAX = 1.0, but for some
        problems a value of 0.99, 0.98, or 0.97 will reduce the number of
        iterations required for convergence.
        Default value: 0.98.
    matrix_conditioning_method: int, optional
        the flag used to select the matrix conditioning method
            1 is for Modified Incomplete Cholesky (for use on scalar computers)
            2 is for Polynomial (for use on vector computers or to conserve computer memory)
    damp: float, optional
        the damping factor. 
        It is typically set equal to one, which indicates
        no damping. A value less than 1 and greater than 0 causes damping. DAMP
        does not affect inner iterations; instead, it affects outer iterations.
        Default value: 1.0.
    damp_transient: float, optional
        the damping factor for transient stress periods. 
        it is read only when `damp` is specified as a negative value. 
        If damp_transient is not read, then the single damping factor, 
        `damp`, is used for both transient and steady-state stress periods.
    printout_interval: int, optional
        is the printout interval for PCG. 
        If equal to zero, it is changed to 999. 
        The maximum head change (positive or negative) and 
        residual change are printed for each iteration of a time step 
        whenever the time step is an even multiple of printout_interval. 
        This printout also occurs at the end of each stress period 
        regardless of the value of printout_interval.
    print_convergence_info: int, optional
        a flag that controls printing of convergence information from the solver:
            0 is for printing tables of maximum head change and residual each iteration
            1 is for printing only the total number of iterations
            2 is for no printing
            3 is for printing only if convergence fails

    """

    _pkg_id = "pcg"

    # TODO: update when all options are known
    _variable_order = [
        "max_iter",
        "inner_iter",
        "hclose",
        "rclose",
        "relax",
        "matrix_conditioning_method",
        "damp",
        "damp_transient",
        "printout_interval",
        "print_convergence_info",
    ]

    _template_projectfile = jinja2.Template(
        "0001, ({{pkg_id}}), 1, Precondition Conjugate-Gradient\n"
        "{{max_iter}}, {{inner_iter}}, {{hclose}}, {{rclose}}, {{relax}}, "
        "{{matrix_conditioning_method}}, {{printout_interval}}, {{print_convergence_info}}, {{damp}}, {{damp_transient}}, "
        "{{IQERROR}}, {{QERROR}}\n"
    )

    def __init__(
        self,
        max_iter=150,
        inner_iter=100,
        rclose=1.0,
        hclose=1.0e-4,
        relax=0.98,
        damp=1.0,
        damp_transient=1.0,
        matrix_conditioning_method=1,
        printout_interval=0,
        print_convergence_info=1
    ):
        super(__class__, self).__init__()
        self.dataset["max_iter"] = max_iter
        self.dataset["inner_iter"] = inner_iter
        self.dataset["rclose"] = rclose
        self.dataset["hclose"] = hclose
        self.dataset["relax"] = relax
        self.dataset["matrix_conditioning_method"] = matrix_conditioning_method
        self.dataset["damp"] = damp
        self.dataset["damp_transient"] = damp_transient
        self.dataset["printout_interval"] = printout_interval
        self.dataset["print_convergence_info"] = print_convergence_info

        # TODO Check with Peter what these settings do
        # I now used the settings from the LHM here...
        self.dataset["IQERROR"] = 1
        self.dataset["QERROR"] = 5.0

    def _render(self, **kwargs):
        d = {k: v.item() for k, v in self.dataset.items()}
        d["pkg_id"] = self._pkg_id

        return self._template_projectfile.render(**d)
