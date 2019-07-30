from imod.mf6.pkgbase import Package


class SolutionBase(Package):
    def __init__(self, print_option, csv_output, no_ptc):
        self.print_option = print_option
        self.csv_output = csv_output
        self.no_ptc = no_ptc


class SolutionPresetSimple(SolutionBase):
    def render(self):
        d = {"complexity": "simple"}
        for k, v in self.data_vars.items():
            d[k] = v
        self._template.render(**d)


class SolutionPresetModerate(SolutionBase):
    def render(self):
        d = {"complexity": "moderate"}
        for k, v in self.data_vars.items():
            d[k] = v
        self._template.render(**d)


class SolutionPresetComplex(SolutionBase):
    def render(self):
        d = {"complexity": "complex"}
        for k, v in self.data_vars.items():
            d[k] = v
        self._template.render(**d)


class Solution(SolutionBase):
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
