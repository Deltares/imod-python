import pathlib

import jinja2
import numpy as np
import pandas as pd

from imod import util
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
    damp: float, optional
        is the damping factor. It is typically set equal to one, which indicates
        no damping. A value less than 1 and greater than 0 causes damping. DAMP
        does not affect inner iterations; instead, it affects outer iterations.
        Default value: 1.0.
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

    def __init__(
        self,
        max_iter=150,
        inner_iter=100,
        rclose=1000.0,
        hclose=1.0e-4,
        relax=0.98,
        damp=1.0,
    ):
        super(__class__, self).__init__()
        self["max_iter"] = max_iter
        self["inner_iter"] = inner_iter
        self["rclose"] = rclose
        self["hclose"] = hclose
        self["relax"] = relax
        self["damp"] = damp

    def _pkgcheck(self, ibound=None):
        to_check = ["max_iter", "inner_iter", "rclose", "hclose", "damp"]
        self._check_positive(to_check)


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
    lump_dispersion: bool
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
        max_iter=1,
        inner_iter=50,
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

    def _pkgcheck(self, ibound=None):
        to_check = ["max_iter", "inner_iter", "cclose"]
        self._check_positive(to_check)


class ParallelSolver(Package):
    """
    Base package for the parallel solvers.
    """

    __slots__ = ()

    def _compute_load_balance_weight(self, ibound):
        if self["partition"] == "rcb":
            if self["load_balance_weight"].values[()] is None:
                self["load_balance_weight"] = (ibound != 0).sum("layer").astype(float)

    def _render(self, directory):
        d = {k: v.values for k, v in self.dataset.data_vars.items()}
        if hasattr(self, "_keywords"):
            for key in self._keywords.keys():
                self._replace_keyword(d, key)

        if self["partition"] == "rcb":
            d["load_balance_weight"] = self._compose(
                {
                    "directory": directory,
                    "name": "load_balance_weight",
                    "extension": ".asc",
                }
            )

        return self._template.render(**d)

    def save(self, directory):
        """
        Overloaded method to write .asc instead of .idf.
        (This is an idiosyncracy of the parallel iMODwq code.)
        """
        # TODO: remove this when iMOD_wq accepts idfs for the load grid.
        da = self["load_balance_weight"]
        if not da.dims == ("y", "x"):
            raise ValueError(
                "load_balance_weight dimensions must be exactly ('y', 'x')."
            )
        path = pathlib.Path(directory).joinpath("load_balance_weight.asc")
        path.parent.mkdir(exist_ok=True, parents=True)
        pd.DataFrame(da.values).to_csv(
            path, sep="\t", header=False, index=False, float_format="%8.2f"
        )


class ParallelKrylovFlowSolver(ParallelSolver):
    """
    The Parallel Krylov Flow Solver is used for parallel solving of the flow
    model.

    Parameters
    ----------
    max_iter: int
        is the maximum number of outer iterations (MXITER); it should be set to
        an integer greater than one (1) only when a nonlinear sorption isotherm
        is included in simulation.
    inner_iter: int
        is the maximum number of inner iterations (INNERIT); a value of 30-50
        should be adequate for most problems.
    hclose: float
        is the head change criterion for convergence (HCLOSEPKS), in units of
        length. When the maximum absolute value of head change from all nodes
        during an iteration is less than or equal to HCLOSE, and the criterion
        for RCLOSE is also satisfied (see below), iteration stops.
    rclose: float
        is the residual criterion for convergence (RCLOSEPKS), in units of cubic
        length per time. The units for length and time are the same as
        established for all model data. When the maximum absolute value of the
        residual at all nodes during an iteration is less than or equal to
        RCLOSE, and the criterion for HCLOSE is also satisfied (see above),
        iteration stops.
    relax: float
        is the relaxation parameter used. Usually, RELAX = 1.0, but for some
        problems a value of 0.99, 0.98, or 0.97 will reduce the number of
        iterations required for convergence.
    h_fstrict: float, optional
        is a factor to apply to HCLOSE to set a stricter hclose for the linear
        inner iterations (H_FSTRICTPKS). HCLOSE_inner is calculated as follows:
        HCLOSEPKS * H_FSTRICTPKS.
    r_fstrict: float, optional
        is a factor to apply to RCLOSE to set a stricter rclose for the linear
        inner iterations (R_FSTRICTPKS). RCLOSE_inner is calculated as follows:
        RCLOSEPKS * R_FSTRICTPKS.
    partition: {"uniform", "rcb"}, optional
        Partitioning option (PARTOPT). "uniform" partitions the model domain
        into equally sized subdomains. "rcb" (Recursive Coordinate Bisection)
        uses a 2D pointer grid with weights to partition the model domain.
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
    load_balance_weight: xarray.DataArray, optional
        2D grid with load balance weights, used when partition = "rcb"
        (Recursive Coordinate Bisection). If None (default), then the module
        will create a load balance grid by summing active cells over layers:
        `(ibound != 0).sum("layer")`

        Note that even though the iMOD-SEAWAT helpfile states .idf is
        accepted, it is not. This load balance grid should be a .asc file
        (without a header). Formatting is done as follows:
        `pd.DataFrame(load_balance_weight.values).to_csv(path, sep='\\t',
        header=False, index=False, float_format = "%8.2f")`
    """

    _pkg_id = "pksf"
    _template = jinja2.Template(
        "[pksf]\n"
        "    mxiter = {{max_iter}}\n"
        "    innerit = {{inner_iter}}\n"
        "    hclosepks = {{hclose}}\n"
        "    rclosepks = {{rclose}}\n"
        "    relax = {{relax}}\n"
        "    h_fstrictpks = {{h_fstrict}}\n"
        "    r_fstrictpks = {{r_fstrict}}\n"
        "    partopt = {{partition}}\n"
        "    isolver = {{solver}}\n"
        "    npc = {{preconditioner}}\n"
        "    npcdef = {{deflate}}\n"
        "{% if load_balance_weight %}    loadptr = {{load_balance_weight}}\n{% endif %}"
        "    pressakey = {{debug}}\n"
    )
    _keywords = {
        "partition": {"uniform": 0, "rcb": 5},
        "solver": {"pcg": 1},
        "preconditioner": {"ilu": 2},
        "deflate": {False: 0, True: 1},
    }

    def __init__(
        self,
        max_iter=150,
        inner_iter=100,
        hclose=1.0e-4,
        rclose=1000.0,
        relax=0.98,
        h_fstrict=1.0,
        r_fstrict=1.0,
        partition="uniform",
        solver="pcg",
        preconditioner="ilu",
        deflate=False,
        debug=False,
        load_balance_weight=None,
    ):
        super(__class__, self).__init__()
        self["max_iter"] = max_iter
        self["inner_iter"] = inner_iter
        self["hclose"] = hclose
        self["rclose"] = rclose
        self["relax"] = relax
        self["h_fstrict"] = h_fstrict
        self["r_fstrict"] = r_fstrict
        self["partition"] = partition
        self["solver"] = solver
        self["preconditioner"] = preconditioner
        self["deflate"] = deflate
        self["debug"] = debug
        self["load_balance_weight"] = load_balance_weight

    def _pkgcheck(self, ibound=None):
        to_check = [
            "hclose",
            "rclose",
            "h_fstrict",
            "r_fstrict",
            "max_iter",
            "inner_iter",
            "relax",
        ]
        self._check_positive(to_check)
        # TODO: fix
        ## Check whether option is actually an available option
        # for opt_arg in self._keywords.keys():
        #    if self[opt_arg] not in self._keywords[opt_arg].keys():
        #        raise ValueError(
        #            "Argument for {} not in {}, instead got {}".format(
        #                opt_arg, self._keywords[opt_arg].keys(), self[opt_arg]
        #            )
        #        )
        #


class ParallelKrylovTransportSolver(ParallelSolver):
    """
    The Parallel Krylov Transport Solver is used for parallel solving of the
    transport model.

    Parameters
    ----------
    max_iter: int
        is the maximum number of outer iterations (MXITER); it should be set to
        an integer greater than one (1) only when a nonlinear sorption isotherm
        is included in simulation.
    inner_iter: int
        is the maximum number of inner iterations (INNERIT); a value of 30-50
        should be adequate for most problems.
    cclose: float, optional
        is the convergence criterion in terms of relative concentration; a real
        value between 10-4 and 10-6 is generally adequate.
        Default value: 1.0e-6.
    relax: float, optional
        is the relaxation parameter used. Usually, RELAX = 1.0, but for some
        problems a value of 0.99, 0.98, or 0.97 will reduce the number of
        iterations required for convergence.
        Default value: 0.98.
    partition: {"uniform", "rcb"}, optional
        Partitioning option (PARTOPT). "uniform" partitions the model domain
        into equally sized subdomains. "rcb" (Recursive Coordinate Bisection)
        uses a 2D pointer grid with weights to partition the model domain.
        Default value: "uniform".
    solver: {"bicgstab", "gmres", "gcr"}, optional
        Flag indicating the linear solver to be used (ISOLVER).
        Default value: "bicgstab"
    preconditioner: {"ilu"}, optional
        Flag inicating the preconditioner to be used (NPC).
        Devault value: "ilu".
    debug: {True, False}, optional
        Debug option.
        Default value: False
    load_balance_weight: xarray.DataArray, optional
        2D grid with load balance weights, used when partition = "rcb"
        (Recursive Coordinate Bisection). If None (default), then the module
        will create a load balance grid by summing active cells over layers:
        `(ibound != 0).sum("layer")`

        Note that even though the iMOD-SEAWAT helpfile states .idf is
        accepted, it is not. This load balance grid should be a .asc file
        (without a header). Formatting is done as follows:
        `pd.DataFrame(load_balance_weight.values).to_csv(path, sep='\\t',
        header=False, index=False, float_format = "%8.2f")`
    """

    __slots__ = ()
    _pkg_id = "pkst"
    _template = jinja2.Template(
        "[pkst]\n"
        "    mxiter = {{max_iter}}\n"
        "    innerit = {{inner_iter}}\n"
        "    cclosepks = {{cclose}}\n"
        "    relax = {{relax}}\n"
        "    partopt = {{partition}}\n"
        "    isolver = {{solver}}\n"
        "    npc = {{preconditioner}}\n"
        "{% if load_balance_weight %}    loadptr = {{load_balance_weight}}\n{% endif %}"
        "    pressakey = {{debug}}\n"
    )

    _keywords = {
        "partition": {"uniform": 0, "rcb": 5},
        "solver": {"bicgstab": 2, "gmres": 3, "gcr": 4},
        "preconditioner": {"ilu": 2},
    }

    def __init__(
        self,
        max_iter=1,
        inner_iter=50,
        cclose=1.0e-6,
        relax=0.98,
        partition="uniform",
        solver="bicgstab",
        preconditioner="ilu",
        debug=False,
        load_balance_weight=None,
    ):
        super(__class__, self).__init__()
        self["max_iter"] = max_iter
        self["inner_iter"] = inner_iter
        self["cclose"] = cclose
        self["relax"] = relax
        self["partition"] = partition
        self["solver"] = solver
        self["preconditioner"] = preconditioner
        self["debug"] = debug
        self["load_balance_weight"] = load_balance_weight

    def _pkgcheck(self, ibound=None):
        to_check = ["cclose", "max_iter", "inner_iter", "relax"]
        self._check_positive(to_check)
        # TODO: fix
        ## Check whether option is actually an available option
        # for opt_arg in self._keywords.keys():
        #    if self[opt_arg] not in self._keywords[opt_arg].keys():
        #        raise ValueError(
        #            "Argument for {} not in {}, instead got {}".format(
        #                opt_arg, self._keywords[opt_arg].keys(), self[opt_arg]
        #            )
        #        )
