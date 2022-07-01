import numpy as np

from imod.mf6.pkgbase import Package, VariableMetaData


class Solution(Package):
    """
    Iterative Model Solution.
    The model solution will solve all of the models that are added to it, as
    specified in the simulation name file, and will include Numerical Exchanges,
    if they are present. The iterative model solution requires specification of
    both nonlinear and linear settings.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=147

    Three predifined solutions settings are available: SolutionPresetSimple,
    SolutionPresetModerate and SolutionPresetComplex. When using one of the
    predefined solutions only the print_option, csv_output, and no_ptc have to
    be defined. The default values for each are described below.

    Parameters
    ----------
    outer_dvclose: float
        real value defining the head change criterion for convergence of the
        outer (nonlinear) iterations, in units of length. When the maximum
        absolute value of the head change at all nodes during an iteration is
        less than or equal to outer_dvclose, iteration stops. Commonly,
        outer_dvclose equals 0.01.
        SolutionPresetSimple: 0.001
        SolutionPresetModerate: 0.01
        SolutionPresetComplex: 0.1
    outer_maximum: int
        integer value defining the maximum number of outer (nonlinear)
        iterations – that is, calls to the solution routine. For a linear
        problem outer_maximum should be 1.
        SolutionPresetSimple: 25
        SolutionPresetModerate: 50
        SolutionPresetComplex: 100
    inner_maximum: int
        integer value defining the maximum number of inner (linear) iterations.
        The number typically depends on the characteristics of the matrix
        solution scheme being used. For nonlinear problems, inner_maximum
        usually ranges from 60 to 600; a value of 100 will be sufficient for
        most linear problems.
        SolutionPresetSimple: 50
        SolutionPresetModerate: 100
        SolutionPresetComplex: 500
    inner_dvclose: float
        real value defining the head change criterion for convergence of the
        inner (linear) iterations, in units of length. When the maximum absolute
        value of the head change at all nodes during an iteration is less than
        or equal to inner_dvclose, the matrix solver assumes convergence.
        Commonly, inner_dvclose is set an order of magnitude less than the
        outer_dvclose value.
        SolutionPresetSimple: 0.001
        SolutionPresetModerate: 0.01
        SolutionPresetComplex: 0.1
    inner_rclose: float
        real value that defines the flow residual tolerance for convergence of
        the IMS linear solver and specific flow residual criteria used. This
        value represents the maximum allowable residual at any single node.
        Value is in units of length cubed per time, and must be consistent with
        MODFLOW 6 length and time units. Usually a value of 1.0 × 10−1 is
        sufficient for the flow-residual criteria when meters and seconds are
        the defined MODFLOW 6 length and time.
        SolutionPresetSimple: 0.1
        SolutionPresetModerate: 0.1
        SolutionPresetComplex: 0.1
    linear_acceleration: str
        options: {"cg", "bicgstab"}
        a keyword that defines the linear acceleration method used by the
        default IMS linear solvers. CG - preconditioned conjugate gradient
        method. BICGSTAB - preconditioned bi-conjugate gradient stabilized
        method.
        SolutionPresetSimple: "cg"
        SolutionPresetModerate: "bicgstab"
        SolutionPresetComplex: "bicgstab"
    under_relaxation: str, optional
        options: {"None", "simple", "cooley", "bdb"}
        is an optional keyword that defines the nonlinear relative_rclose
        schemes used. By default under_relaxation is not used.
        None - relative_rclose is not used.
        simple - Simple relative_rclose scheme with a fixed relaxation factor is
        used.
        cooley - Cooley relative_rclose scheme is used.
        dbd - delta-bar-delta relative_rclose is used.
        Note that the relative_rclose schemes are used in conjunction with
        problems that use the Newton-Raphson formulation, however, experience
        has indicated that the Cooley relative_rclose and damping work well also
        for the Picard scheme with the wet/dry options of MODFLOW 6.
        Default value: None
        SolutionPresetSimple: None
        SolutionPresetModerate: "dbd"
        SolutionPresetComplex: "dbd"
    under_relaxation_theta: float, optional
        real value defining the reduction factor for the learning rate
        (underrelaxation term) of the delta-bar-delta algorithm. The value of
        under relaxation theta is between zero and one. If the change in the
        variable (head) is of opposite sign to that of the previous iteration,
        the relative_rclose term is reduced by a factor of under relaxation
        theta. The value usually ranges from 0.3 to 0.9; a value of 0.7 works
        well for most problems. under relaxation theta only needs to be
        specified if under relaxation is dbd.
        Default value: None
        SolutionPresetSimple: 0.0
        SolutionPresetModerate: 0.9
        SolutionPresetComplex: 0.8
    under_relaxation_kappa: float, optional
        real value defining the increment for the learning rate (relative_rclose
        term) of the delta-bar-delta algorithm. The value of under relaxation
        kappa is between zero and one. If the change in the variable (head) is
        of the same sign to that of the previous iteration, the relative_rclose
        term is increased by an increment of under_relaxation_kappa. The value
        usually ranges from 0.03 to 0.3; a value of 0.1 works well for most
        problems. under relaxation kappa only needs to be specified if under
        relaxation is dbd.
        Default value: None
        SolutionPresetSimple: 0.0
        SolutionPresetModerate: 0.0001
        SolutionPresetComplex: 0.0001
    under_relaxation_gamma: float, optional
        real value defining the history or memory term factor of the
        delta-bardelta algorithm. under relaxation gamma is between zero and 1
        but cannot be equal to one. When under relaxation gamma is zero, only
        the most recent history (previous iteration value) is maintained. As
        under relaxation gamma is increased, past history of iteration changes
        has greater influence on the memory term. The memory term is maintained
        as an exponential average of past changes. Retaining some past history
        can overcome granular behavior in the calculated function surface and
        therefore helps to overcome cyclic patterns of nonconvergence. The value
        usually ranges from 0.1 to 0.3; a value of 0.2 works well for most
        problems. under relaxation gamma only needs to be specified if under
        relaxation is not none.
        Default value: None
        SolutionPresetSimple: 0.0
        SolutionPresetModerate: 0.0
        SolutionPresetComplex: 0.0
    under_relaxation_momentum: float, optional
        real value defining the fraction of past history changes that is added
        as a momentum term to the step change for a nonlinear iteration. The
        value of under relaxation momentum is between zero and one. A large
        momentum term should only be used when small learning rates are
        expected. Small amounts of the momentum term help convergence. The value
        usually ranges from 0.0001 to 0.1; a value of 0.001 works well for most
        problems. under relaxation momentum only needs to be specified if under
        relaxation is dbd.
        Default value: None
        SolutionPresetSimple: 0.0
        SolutionPresetModerate: 0.0
        SolutionPresetComplex: 0.0
    backtracking_number: int, optional
        integer value defining the maximum number of backtracking iterations
        allowed for residual reduction computations. If backtracking number = 0
        then the backtracking iterations are omitted. The value usually ranges
        from 2 to 20; a value of 10 works well for most problems.
        Default value: None
        SolutionPresetSimple: 0
        SolutionPresetModerate: 0
        SolutionPresetComplex: 20
    backtracking_tolerance: float, optional
        real value defining the tolerance for residual change that is allowed
        for residual reduction computations. backtracking tolerance should not
        be less than one to avoid getting stuck in local minima. A large value
        serves to check for extreme residual increases, while a low value serves
        to control step size more severely. The value usually ranges from 1.0 to
        106; a value of 104 works well for most problems but lower values like
        1.1 may be required for harder problems. backtracking tolerance only
        needs to be specified if backtracking_number is greater than zero.
        Default value: None
        SolutionPresetSimple: 0.0
        SolutionPresetModerate: 0.0
        SolutionPresetComplex: 1.05
    backtracking_reduction_factor: float, optional
        real value defining the reduction in step size used for residual
        reduction computations. The value of backtracking reduction factor is
        between 142 MODFLOW 6 – Description of Input and Output zero and one.
        The value usually ranges from 0.1 to 0.3; a value of 0.2 works well for
        most problems. backtracking_reduction_factor only needs to be specified
        if backtracking number is greater than zero.
        Default value: None
        SolutionPresetSimple: 0.0
        SolutionPresetModerate: 0.0
        SolutionPresetComplex: 0.1
    backtracking_residual_limit: float, optional
        real value defining the limit to which the residual is reduced with
        backtracking. If the residual is smaller than
        backtracking_residual_limit, then further backtracking is not performed.
        A value of 100 is suitable for large problems and residual reduction to
        smaller values may only slow down computations. backtracking residual
        limit only needs to be specified if backtracking_number is greater than
        zero.
        Default value: None
        SolutionPresetSimple: 0.0
        SolutionPresetModerate: 0.0
        SolutionPresetComplex: 0.002
    rclose_option: str, optional
        options: {"strict", "l2norm_rclose", "relative_rclose"}
        an optional keyword that defines the specific flow residual criterion
        used.
        strict– an optional keyword that is used to specify that inner rclose
        represents a infinity-norm (absolute convergence criteria) and that the
        head and flow convergence criteria must be met on the first inner
        iteration (this criteria is equivalent to the criteria used by the
        MODFLOW-2005 PCG package (Hill, 1990)).
        l2norm_rclose – an optionalkeyword that is used to specify that inner
        rclose represents a l-2 norm closure criteria instead of a infinity-norm
        (absolute convergence criteria). When l2norm_rclose is specified, a
        reasonable initial inner rclose value is 0.1 times the number of active
        cells when meters and seconds are the defined MODFLOW 6 length and time.
        relative_rclose – an optional keyword that is used to specify that
        inner_rclose represents a relative L-2 Norm reduction closure criteria
        instead of a infinity-Norm (absolute convergence criteria). When
        relative_rclose is specified, a reasonable initial inner_rclose value is
        1.0 × 10−4 and convergence is achieved for a given inner (linear)
        iteration when ∆h ≤ inner_dvclose and the current L-2 Norm is ≤ the
        product of the relativ_rclose and the initial L-2 Norm for the current
        inner (linear) iteration. If rclose_option is not specified, an absolute
        residual (infinity-norm) criterion is used.
        Default value: None
        SolutionPresetSimple: "strict"
        SolutionPresetModerate: "strict"
        SolutionPresetComplex: "strict"
    relaxation_factor: float, optional
        optional real value that defines the relaxation factor used by the
        incomplete LU factorization preconditioners (MILU(0) and MILUT).
        relaxation_factor is unitless and should be greater than or equal to 0.0
        and less than or equal to 1.0. relaxation_factor Iterative Model
        Solution 143 values of about 1.0 are commonly used, and experience
        suggests that convergence can be optimized in some cases with relax
        values of 0.97. A relaxation_factor value of 0.0 will result in either
        ILU(0) or ILUT preconditioning (depending on the value specified for
        preconditioner_levels and/or preconditioner_drop_tolerance). By default,
        relaxation_factor is zero.
        Default value: None
        SolutionPresetSimple: 0.0
        SolutionPresetModerate: 0
        SolutionPresetComplex: 0.0
    preconditioner_levels: int, optional
        optional integer value defining the level of fill for ILU decomposition
        used in the ILUT and MILUT preconditioners. Higher levels of fill
        provide more robustness but also require more memory. For optimal
        performance, it is suggested that a large level of fill be applied (7 or
        8) with use of a drop tolerance. Specification of a
        preconditioner_levels value greater than zero results in use of the ILUT
        preconditioner. By default, preconditioner_levels is zero and the
        zero-fill incomplete LU factorization preconditioners (ILU(0) and
        MILU(0)) are used.
        Default value: None
        SolutionPresetSimple: 0
        SolutionPresetModerate: 0
        SolutionPresetComplex: 5
    preconditioner_drop_tolerance: float, optional
        optional real value that defines the drop tolerance used to drop
        preconditioner terms based on the magnitude of matrix entries in the
        ILUT and MILUT preconditioners. A value of 10−4 works well for most
        problems. By default, preconditioner_drop_tolerance is zero and the
        zero-fill incomplete LU factorization preconditioners (ILU(0) and
        MILU(0)) are used.
        Default value: None
        SolutionPresetSimple: 0
        SolutionPresetModerate: 0.0
        SolutionPresetComplex: 0.0001
    number_orthogonalizations: int, optional
        optional integer value defining the interval used to explicitly
        recalculate the residual of the flow equation using the solver
        coefficient matrix, the latest head estimates, and the right hand side.
        For problems that benefit from explicit recalculation of the residual, a
        number between 4 and 10 is appropriate. By default,
        number_orthogonalizations is zero.
        Default value: None
        SolutionPresetSimple: 0
        SolutionPresetModerate: 0
        SolutionPresetComplex: 2
    scaling_method: str
        options: {"None", "diagonal", "l2norm"}
        an optional keyword that defines the matrix scaling approach used. By
        default, matrix scaling is not applied.
        None - no matrix scaling applied.
        diagonal - symmetric matrix scaling using the POLCG preconditioner
        scaling method in Hill (1992).
        l2norm - symmetric matrix scaling using the L2 norm.
        Default value: None
        SolutionPresetSimple: None
        SolutionPresetModerate: None
        SolutionPresetComplex: None
    reordering_method: str
        options: {"None", "rcm", "md"}
        an optional keyword that defines the matrix reordering approach used. By
        default, matrix reordering is not applied.
        None - original ordering.
        rcm - reverse Cuthill McKee ordering.
        md - minimum degree ordering
        Default value: None
        SolutionPresetSimple: None
        SolutionPresetModerate: None
        SolutionPresetComplex: None
    print_option: str
        options: {"None", "summary", "all"}
        is a flag that controls printing of convergence information from the
        solver.
        None - means print nothing.
        summary - means print only the total
        number of iterations and nonlinear residual reduction summaries.
        all - means print linear matrix solver convergence information to the
        solution listing file and model specific linear matrix solver
        convergence information to each model listing file in addition to
        SUMMARY information.
        Default value: "summary"
        SolutionPresetSimple: No Default
        SolutionPresetModerate: No Default
        SolutionPresetComplex: No Default
    csv_output: str, optional
        False if no csv is to be written for the output, enter str of filename
        if csv is to be written.
        Default value: False
        SolutionPresetSimple: No Default
        SolutionPresetModerate: No Default
        SolutionPresetComplex: No Default
    no_ptc: ({True, False}, optional)
        is a flag that is used to disable pseudo-transient continuation (PTC).
        Option only applies to steady-state stress periods for models using the
        Newton-Raphson formulation. For many problems, PTC can significantly
        improve convergence behavior for steady-state simulations, and for this
        reason it is active by default. In some cases, however, PTC can worsen
        the convergence behavior, especially when the initial conditions are
        similar to the solution. When the initial conditions are similar to, or
        exactly the same as, the solution and convergence is slow, then this NO
        PTC option should be used to deactivate PTC. This NO PTC option should
        also be used in order to compare convergence behavior with other MODFLOW
        versions, as PTC is only available in MODFLOW 6.
        Default value: False
        SolutionPresetSimple: No Default
        SolutionPresetModerate: No Default
        SolutionPresetComplex: No Default
    """

    _pkg_id = "ims"
    _keyword_map = {}
    _metadata_dict = {
        "outer_dvclose": VariableMetaData(np.floating),
        "outer_maximum": VariableMetaData(np.integer),
        "inner_maximum": VariableMetaData(np.integer),
        "inner_dvclose": VariableMetaData(np.floating),
        "inner_rclose": VariableMetaData(np.floating),
        "under_relaxation_theta": VariableMetaData(np.floating),
        "under_relaxation_kappa": VariableMetaData(np.floating),
        "under_relaxation_gamma": VariableMetaData(np.floating),
        "under_relaxation_momentum": VariableMetaData(np.floating),
        "backtracking_number": VariableMetaData(np.integer),
        "backtracking_tolerance": VariableMetaData(np.floating),
        "backtracking_reduction_factor": VariableMetaData(np.floating),
        "backtracking_residual_limit": VariableMetaData(np.floating),
        "number_orthogonalizations": VariableMetaData(np.integer),
    }
    _template = Package._initialize_template(_pkg_id)

    def __init__(
        self,
        outer_dvclose,
        outer_maximum,
        inner_maximum,
        inner_dvclose,
        inner_rclose,
        linear_acceleration,
        under_relaxation=None,
        under_relaxation_theta=None,
        under_relaxation_kappa=None,
        under_relaxation_gamma=None,
        under_relaxation_momentum=None,
        backtracking_number=None,
        backtracking_tolerance=None,
        backtracking_reduction_factor=None,
        backtracking_residual_limit=None,
        rclose_option=None,
        relaxation_factor=None,
        preconditioner_levels=None,
        preconditioner_drop_tolerance=None,
        number_orthogonalizations=None,
        scaling_method=None,
        reordering_method=None,
        print_option="summary",
        csv_output=False,
        no_ptc=False,
    ):
        super().__init__()
        self.dataset["outer_dvclose"] = outer_dvclose
        self.dataset["outer_maximum"] = outer_maximum
        self.dataset["under_relaxation"] = under_relaxation
        self.dataset["under_relaxation_theta"] = under_relaxation_theta
        self.dataset["under_relaxation_kappa"] = under_relaxation_kappa
        self.dataset["under_relaxation_gamma"] = under_relaxation_gamma
        self.dataset["under_relaxation_momentum"] = under_relaxation_momentum
        self.dataset["backtracking_number"] = backtracking_number
        self.dataset["backtracking_tolerance"] = backtracking_tolerance
        self.dataset["backtracking_reduction_factor"] = backtracking_reduction_factor
        self.dataset["backtracking_residual_limit"] = backtracking_residual_limit
        self.dataset["inner_maximum"] = inner_maximum
        self.dataset["inner_dvclose"] = inner_dvclose
        self.dataset["inner_rclose"] = inner_rclose
        self.dataset["rclose_option"] = rclose_option
        self.dataset["linear_acceleration"] = linear_acceleration
        self.dataset["relaxation_factor"] = relaxation_factor
        self.dataset["preconditioner_levels"] = preconditioner_levels
        self.dataset["preconditioner_drop_tolerance"] = preconditioner_drop_tolerance
        self.dataset["number_orthogonalizations"] = number_orthogonalizations
        self.dataset["scaling_method"] = scaling_method
        self.dataset["reordering_method"] = reordering_method
        self.dataset["print_option"] = print_option
        self.dataset["csv_output"] = csv_output
        self.dataset["no_ptc"] = no_ptc

        self._pkgcheck_at_init()


def SolutionPresetSimple(print_option, csv_output, no_ptc):
    solution = Solution(
        print_option=print_option,
        csv_output=csv_output,
        no_ptc=no_ptc,
        outer_dvclose=0.001,
        outer_maximum=25,
        under_relaxation=None,
        under_relaxation_theta=0.0,
        under_relaxation_kappa=0.0,
        under_relaxation_gamma=0.0,
        under_relaxation_momentum=0.0,
        backtracking_number=0,
        backtracking_tolerance=0.0,
        backtracking_reduction_factor=0.0,
        backtracking_residual_limit=0.0,
        inner_maximum=50,
        inner_dvclose=0.001,
        inner_rclose=0.1,
        rclose_option="strict",
        linear_acceleration="cg",
        relaxation_factor=0.0,
        preconditioner_levels=0,
        preconditioner_drop_tolerance=0,
        number_orthogonalizations=0,
        scaling_method=None,
        reordering_method=None,
    )
    return solution


def SolutionPresetModerate(print_option, csv_output, no_ptc):
    solution = Solution(
        print_option=print_option,
        csv_output=csv_output,
        no_ptc=no_ptc,
        outer_dvclose=0.01,
        outer_maximum=50,
        under_relaxation="dbd",
        under_relaxation_theta=0.9,
        under_relaxation_kappa=0.0001,
        under_relaxation_gamma=0.0,
        under_relaxation_momentum=0.0,
        backtracking_number=0,
        backtracking_tolerance=0.0,
        backtracking_reduction_factor=0.0,
        backtracking_residual_limit=0.0,
        inner_maximum=100,
        inner_dvclose=0.01,
        inner_rclose=0.1,
        rclose_option="strict",
        linear_acceleration="bicgstab",
        relaxation_factor=0,
        preconditioner_levels=0,
        preconditioner_drop_tolerance=0.0,
        number_orthogonalizations=0,
        scaling_method=None,
        reordering_method=None,
    )
    return solution


def SolutionPresetComplex(print_option, csv_output, no_ptc):
    solution = Solution(
        print_option=print_option,
        csv_output=csv_output,
        no_ptc=no_ptc,
        outer_dvclose=0.1,
        outer_maximum=100,
        under_relaxation="dbd",
        under_relaxation_theta=0.8,
        under_relaxation_kappa=0.0001,
        under_relaxation_gamma=0.0,
        under_relaxation_momentum=0.0,
        backtracking_number=20,
        backtracking_tolerance=1.05,
        backtracking_reduction_factor=0.1,
        backtracking_residual_limit=0.002,
        inner_maximum=500,
        inner_dvclose=0.1,
        inner_rclose=0.1,
        rclose_option="strict",
        linear_acceleration="bicgstab",
        relaxation_factor=0.0,
        preconditioner_levels=5,
        preconditioner_drop_tolerance=0.0001,
        number_orthogonalizations=2,
        scaling_method=None,
        reordering_method=None,
    )
    return solution
