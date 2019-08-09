from imod.mf6.pkgbase import Package


class Solution(Package):
    def __init__(
        self,
        print_option,
        csv_output,
        no_ptc,
        outer_hclose,
        outer_rclosebnd,
        outer_maximum,
        under_relaxation,
        under_relaxation_theta,
        under_relaxation_kappa,
        under_relaxation_gamma,
        under_relaxation_momentum,
        backtracking_number,
        backtracking_tolerance,
        backtracking_reduction_factor,
        backtracking_residual_limit,
        inner_maximum,
        inner_hclose,
        inner_rclose,
        rclose_option,
        linear_acceleration,
        relaxation_factor,
        preconditioner_levels,
        preconditioner_drop_tolerance,
        number_orthogonalizations,
        scaling_method,
        reordering_method,
    ):
        super(__class__, self).__init__()
        self.outer_hclose = outer_hclose
        self.outer_rclosebnd = outer_rclosebnd
        self.outer_maximum = outer_maximum
        self.under_relaxation = under_relaxation
        self.under_relaxation_theta = under_relaxation_theta
        self.under_relaxation_kappa = under_relaxation_kappa
        self.under_relaxation_gamma = under_relaxation_gamma
        self.under_relaxation_momentum = under_relaxation_momentum
        self.backtracking_number = backtracking_number
        self.backtracking_tolerance = backtracking_tolerance
        self.backtracking_reduction_factor = backtracking_reduction_factor
        self.backtracking_residual_limit = backtracking_residual_limit
        self.inner_maximum = inner_maximum
        self.inner_hclose = inner_hclose
        self.inner_rclose = inner_rclose
        self.rclose_option = rclose_option
        self.linear_acceleration = linear_acceleration
        self.relaxation_factor = relaxation_factor
        self.preconditioner_levels = preconditioner_levels
        self.preconditioner_drop_tolerance = preconditioner_drop_tolerance
        self.number_orthogonalizations = number_orthogonalizations
        self.scaling_method = scaling_method
        self.reordering_method = reordering_method

    def write(self, directory, solvername):
        ims_path = directory / f"{solvername}.ims"
        ims_content = self.render()
        with open(ims_path, "w") as f:
            f.write(ims_content)


def SolutionPresetSimple(print_option, csv_output, no_ptc):
    solution = Solution(
        print_option=print_option,
        csv_output=csv_output,
        no_ptc=no_ptc,
        outer_hclose=0.001,
        outer_rclosebnd=0.1,
        outer_maximum=25,
        under_relaxation=None,
        under_relaxation_theta=0.0,
        under_relaxation_kappa=0.0,
        under_relaxation_gamma=0.0,
        under_relaxation_momentum=0.0,
        backtracking_number=0.0,
        backtracking_tolerance=0.0,
        backtracking_reduction_factor=0.0,
        backtracking_residual_limit=0.0,
        inner_maximum=50,
        inner_hclose=0.001,
        inner_rclose=0.1,
        rclose_option="infinity-norm",
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
        outer_hclose=0.01,
        outer_rclosebnd=0.1,
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
        inner_hclose=0.01,
        inner_rclose=0.1,
        rclose_option="infinity-norm",
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
        outer_hclose=0.1,
        outer_rclosebnd=0.1,
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
        inner_hclose=0.1,
        inner_rclose=0.1,
        rclose_option="infinity-norm",
        linear_acceleration="bicgstab",
        relaxation_factor=0.0,
        preconditioner_levels=5,
        preconditioner_drop_tolerance=0.0001,
        number_orthogonalizations=2,
        scaling_method=None,
        reordering_method=None,
    )
    return solution
