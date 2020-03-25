import textwrap

from imod.wq.pkgbase import Package


class AdvectionFiniteDifference(Package):
    """
    Solve the advection term using the explicit Finite Difference method
    (MIXELM = 0) with upstream weighting

    Attributes
    ----------
    courant: real
        Courant number (PERCEL) is the number of cells (or a fraction of a cell)
        advection will be allowed in any direction in one transport step. For
        implicit finite-difference or particle tracking based schemes, there is
        no limit on PERCEL, but for accuracy reasons, it is generally not set
        much greater than one. Note, however, that the PERCEL limit is checked
        over the entire model grid. Thus, even if PERCEL > 1, advection may not
        be more than one cell’s length at most model locations. For the explicit
        finite-difference, PERCEL is also a stability constraint, which must not
        exceed one and will be automatically reset to one if a value greater
        than one is specified.
    weighting : {"upstream", "central"}, optional
        Indication of which weighting scheme should be used, set to default
        value "upstream" (NADVFD = 0 or 1)
        Default value: "upstream"
    """

    __slots__ = ("courant", "weighting")
    _pkg_id = "adv"

    _keywords = {
        "weighting": {"upstream": 0, "central": 1},
    }

    _template = (
        "[adv]\n"
        "    mixelm = 0\n"
        "    percel = {courant}\n"
        "    nadvfd = {weighting}\n"
    )

    def __init__(self, courant=0.75, weighting="upstream"):
        super(__class__, self).__init__()
        self["courant"] = courant
        self["weighting"] = weighting

    def _pkgcheck(self, ibound=None):
        self._check_positive(["courant"])


class AdvectionMOC(Package):
    """
    Solve the advection term using the Method of Characteristics (MIXELM = 1)

    Nota bene: number of particles settings have not been tested. The defaults
    here are chosen conservatively, with many particles. This increases both
    memory usage and computational effort.

    Attributes
    -----------
    courant: real
        Courant number (PERCEL) is the number of cells (or a fraction of a cell)
        advection will be allowed in any direction in one transport step. For
        implicit finite-difference or particle tracking based schemes, there is
        no limit on PERCEL, but for accuracy reasons, it is generally not set
        much greater than one. Note, however, that the PERCEL limit is checked
        over the entire model grid. Thus, even if PERCEL > 1, advection may not
        be more than one cell’s length at most model locations. For the explicit
        finite-difference, PERCEL is also a stability constraint, which must not
        exceed one and will be automatically reset to one if a value greater
        than one is specified.
    max_nparticles: int
        is the maximum total number of moving particles allowed (MXPART).
    tracking: {"euler", "runge-kutta", "hybrid"}, optional
        indicates which particle tracking algorithm is selected for the
        Eulerian-Lagrangian methods. ITRACK = 1, the first-order Euler algorithm is
        used; ITRACK = 2, the fourth-order Runge-Kutta algorithm is used; this
        option is computationally demanding and may be needed only when PERCEL is
        set greater than one. ITRACK = 3, the hybrid 1st and 4th order algorithm is
        used; the Runge- Kutta algorithm is used in sink/source cells and the cells
        next to sinks/sources while the Euler algorithm is used elsewhere.
        Default value is "hybrid".
    weighting_factor: float, optional
        is a concentration weighting factor (WD) between 0.5 and 1. It is used for
        operator splitting in the particle tracking based methods. The value of
        0.5 is generally adequate. The value may be adjusted to achieve better
        mass balance. Generally, it can be increased toward 1.0 as advection
        becomes more dominant.
        Default value: 0.5.
    dconcentration_epsilon: float, optional
        is a small Relative Cell Concentration Gradient below which advective
        transport is considered negligible. A value around 10-5 is generally
        adequate.
        Default value: 1.0e-5.
    nplane: int, optional
        is a flag indicating whether the random or fixed pattern is selected for
        initial placement of moving particles. NPLANE = 0, the random pattern is
        selected for initial placement. Particles are distributed randomly in
        both the horizontal and vertical directions by calling a random number
        generator. This option is usually preferred and leads to smaller mass
        balance discrepancy in nonuniform or diverging/converging flow fields.
        NPLANE > 0, the fixed pattern is selected for initial placement. The
        value of NPLANE serves as the number of vertical “planes” on which
        initial particles are placed within each cell block. The fixed pattern
        may work better than the random pattern only in relatively uniform flow
        fields. For two-dimensional simulations in plan view, set NPLANE = 1.
        For cross sectional or three-dimensional simulations, NPLANE = 2 is
        normally adequate. Increase NPLANE if more resolution in the vertical
        direction is desired.
        Default value: 2.
    nparticles_no_advection: int, optional
        is number of initial particles per cell to be placed at cells where the
        Relative Cell Concentration Gradient is less than or equal to DCEPS.
        Generally, NPL can be set to zero since advection is considered
        insignificant when the Relative Cell Concentration Gradient is less than
        or equal to DCEPS. Setting NPL equal to NPH causes a uniform number of
        particles to be placed in every cell over the entire grid (i.e., the
        uniform approach).
        Default value: 10.
    nparticles_advection: int, optional
        is number of initial particles per cell to be placed at cells where the
        Relative Cell Concentration Gradient is greater than DCEPS. The
        selection of NPH depends on the nature of the flow field and also the
        computer memory limitation. Generally, use a smaller number in
        relatively uniform flow fields and a larger number in relatively
        nonuniform flow fields. However, values exceeding 16 in twodimensional
        simulation or 32 in three-dimensional simulation are rarely necessary.
        If the random pattern is chosen, NPH particles are randomly distributed
        within the cell block. If the fixed pattern is chosen, NPH is divided by
        NPLANE to yield the number of particles to be placed per vertical plane.
        Default value: 40.
    cell_min_nparticles: int, optional
        is the minimum number of particles allowed per cell. If the number of
        particles in a cell at the end of a transport step is fewer than NPMIN,
        new particles are inserted into that cell to maintain a sufficient
        number of particles. NPMIN can be set to zero in relatively uniform flow
        fields, and a number greater than zero in diverging/converging flow
        fields. Generally, a value between zero and four is adequate.
        Default value is 5.
    cell_max_nparticles: int, optional
        is the maximum number of particles allowed per cell. If the number of
        particles in a cell exceeds NPMAX, all particles are removed from that
        cell and replaced by a new set of particles equal to NPH to maintain
        mass balance. Generally, NPMAX can be set to approximately twice of NPH.
        Default value: 80.
    """

    __slots__ = (
        "courant",
        "tracking",
        "weighting_factor",
        "dconcentration_epsilon",
        "nplane",
        "nparticles_no_advection",
        "nparticles_advection",
        "cell_min_nparticles",
        "cell_max_nparticles",
    )
    _pkg_id = "adv"
    _keywords = {
        "tracking": {"euler": 1, "runge-kutta": 2, "hybrid": 3},
    }

    _template = textwrap.dedent(
        """
        [adv]
            mixelm = 1
            percel = {courant}
            mxpart = {max_nparticles}
            itrack = {tracking}
            wd = {weighting_factor}
            dceps = {dconcentration_epsilon}
            nplane = {nplane}
            npl = {nparticles_no_advection}
            nph = {nparticles_advection}
            npmin = {cell_min_nparticles}
            npmax = {cell_max_nparticles}
    """
    )

    def __init__(
        self,
        courant=0.75,
        tracking="hybrid",
        weighting_factor=0.5,
        dconcentration_epsilon=1.0e-5,
        nplane=2,
        nparticles_no_advection=10,
        nparticles_advection=40,
        cell_min_nparticles=5,
        cell_max_nparticles=80,
    ):
        super(__class__, self).__init__()
        self["courant"] = courant
        self["tracking"] = tracking
        self["weighting_factor"] = weighting_factor
        self["dconcentration_epsilon"] = dconcentration_epsilon
        self["nplane"] = nplane
        self["nparticles_no_advection"] = nparticles_no_advection
        self["nparticles_advection"] = nparticles_advection
        self["cell_min_nparticles"] = cell_min_nparticles
        self["cell_max_nparticles"] = cell_max_nparticles

    def _pkgcheck(self, ibound=None):
        self._check_positive(["courant", "weighting_factor"])


class AdvectionModifiedMOC(Package):
    """
    Solve the advention term using the Modified Method of Characteristics (MIXELM = 2)
    Courant number (PERCEL) is the number of cells (or a fraction of a
    cell) advection will be allowed in any direction in one transport step.

    Attributes
    ----------
    courant: real
        Courant number (PERCEL) is the number of cells (or a fraction of a cell)
        advection will be allowed in any direction in one transport step. For
        implicit finite-difference or particle tracking based schemes, there is
        no limit on PERCEL, but for accuracy reasons, it is generally not set
        much greater than one. Note, however, that the PERCEL limit is checked
        over the entire model grid. Thus, even if PERCEL > 1, advection may not
        be more than one cell’s length at most model locations. For the explicit
        finite-difference, PERCEL is also a stability constraint, which must not
        exceed one and will be automatically reset to one if a value greater
        than one is specified.
    tracking: str, {"euler", "runge-kutta", "hybrid"}
        indicates which particle tracking algorithm is selected for the
        Eulerian-Lagrangian methods. ITRACK = 1, the first-order Euler algorithm is
        used; ITRACK = 2, the fourth-order Runge-Kutta algorithm is used; this
        option is computationally demanding and may be needed only when PERCEL is
        set greater than one. ITRACK = 3, the hybrid 1st and 4th order algorithm is
        used; the Runge- Kutta algorithm is used in sink/source cells and the cells
        next to sinks/sources while the Euler algorithm is used elsewhere.
    weighting_factor: real
        is a concentration weighting factor (WD) between 0.5 and 1. It is used for
        operator splitting in the particle tracking based methods. The value of
        0.5 is generally adequate. The value may be adjusted to achieve better
        mass balance. Generally, it can be increased toward 1.0 as advection
        becomes more dominant.
    dconcentration_epsilon: float, optional
        is a small Relative Cell Concentration Gradient (DCEPS) below which advective
        transport is considered negligible. A value around 1.0e-5 is generally
        adequate.
        Default value: 1.0e-5.
    sink_particle_placement: int
        indicates whether the random or fixed pattern is selected for initial
        placement of particles to approximate sink cells in the MMOC scheme.
        (NLSINK)
    sink_nparticles: int
        is the number of particles used to approximate sink cells in the MMOC
        scheme. (NPSINK)
    """

    __slots__ = (
        "courant",
        "tracking",
        "weighting_factor",
        "dconcentration_epsilon",
        "sink_particle_placement",
        "sink_nparticles",
    )
    _pkg_id = "adv"

    _keywords = {"tracking": {"euler": 1, "runge-kutta": 2, "hybrid": 3}}

    _template = (
        "[adv]\n"
        "    mixelm = 2\n"
        "    percel = {courant}\n"
        "    itrack = {tracking}\n"
        "    wd = {weighting_factor}\n"
        "    interp = 1\n"
        "    nlsink = {sink_particle_placement}\n"
        "    npsink = {sink_nparticles}\n"
    )

    def __init__(
        self,
        courant=1.0,
        tracking="hybrid",
        weighting_factor=0.5,
        dconcentration_epsilon=1.0e-5,
        sink_particle_placement=2,
        sink_nparticles=40,
    ):
        super(__class__, self).__init__()
        self["courant"] = courant
        self["tracking"] = tracking
        self["weighting_factor"] = weighting_factor
        self["sink_particle_placement"] = sink_particle_placement
        self["sink_nparticles"] = sink_nparticles


class AdvectionHybridMOC(Package):
    """
    Hybrid Method of Characteristics and Modified Method of Characteristics with
    MOC or MMOC automatically and dynamically selected (MIXELM = 3)

    Attributes
    ----------
    courant: float
        Courant number (PERCEL) is the number of cells (or a fraction of a cell)
        advection will be allowed in any direction in one transport step. For
        implicit finite-difference or particle tracking based schemes, there is
        no limit on PERCEL, but for accuracy reasons, it is generally not set
        much greater than one. Note, however, that the PERCEL limit is checked
        over the entire model grid. Thus, even if PERCEL > 1, advection may not
        be more than one cell’s length at most model locations. For the explicit
        finite-difference, PERCEL is also a stability constraint, which must not
        exceed one and will be automatically reset to one if a value greater
        than one is specified.
    max_particles: int
        is the maximum total number of moving particles allowed (MXPART).
    tracking: int
        indicates which particle tracking algorithm is selected for the
        Eulerian-Lagrangian methods. ITRACK = 1, the first-order Euler algorithm is
        used; ITRACK = 2, the fourth-order Runge-Kutta algorithm is used; this
        option is computationally demanding and may be needed only when PERCEL is
        set greater than one. ITRACK = 3, the hybrid 1st and 4th order algorithm is
        used; the Runge- Kutta algorithm is used in sink/source cells and the cells
        next to sinks/sources while the Euler algorithm is used elsewhere.
    weighting_factor: real
        is a concentration weighting factor (WD) between 0.5 and 1. It is used for
        operator splitting in the particle tracking based methods. The value of
        0.5 is generally adequate. The value may be adjusted to achieve better
        mass balance. Generally, it can be increased toward 1.0 as advection
        becomes more dominant.
    dceps: real
        is a small Relative Cell Concentration Gradient below which advective
        transport is considered negligible. A value around 10-5 is generally
        adequate.
    nplane: int
        is a flag indicating whether the random or fixed pattern is selected for
        initial placement of moving particles. NPLANE = 0, the random pattern is
        selected for initial placement. Particles are distributed randomly in
        both the horizontal and vertical directions by calling a random number
        generator. This option is usually preferred and leads to smaller mass
        balance discrepancy in nonuniform or diverging/converging flow fields.
        NPLANE > 0, the fixed pattern is selected for initial placement. The
        value of NPLANE serves as the number of vertical “planes” on which
        initial particles are placed within each cell block. The fixed pattern
        may work better than the random pattern only in relatively uniform flow
        fields. For two-dimensional simulations in plan view, set NPLANE = 1.
        For cross sectional or three-dimensional simulations, NPLANE = 2 is
        normally adequate. Increase NPLANE if more resolution in the vertical
        direction is desired.
    npl: int
        is number of initial particles per cell to be placed at cells where the
        Relative Cell Concentration Gradient is less than or equal to DCEPS.
        Generally, NPL can be set to zero since advection is considered
        insignificant when the Relative Cell Concentration Gradient is less than
        or equal to DCEPS. Setting NPL equal to NPH causes a uniform number of
        particles to be placed in every cell over the entire grid (i.e., the
        uniform approach).
    nph: int
        is number of initial particles per cell to be placed at cells where the
        Relative Cell Concentration Gradient is greater than DCEPS. The
        selection of NPH depends on the nature of the flow field and also the
        computer memory limitation. Generally, use a smaller number in
        relatively uniform flow fields and a larger number in relatively
        nonuniform flow fields. However, values exceeding 16 in twodimensional
        simulation or 32 in three-dimensional simulation are rarely necessary.
        If the random pattern is chosen, NPH particles are randomly distributed
        within the cell block. If the fixed pattern is chosen, NPH is divided by
        NPLANE to yield the number of particles to be placed per vertical plane.
    npmin: int
        is the minimum number of particles allowed per cell. If the number of
        particles in a cell at the end of a transport step is fewer than NPMIN,
        new particles are inserted into that cell to maintain a sufficient
        number of particles. NPMIN can be set to zero in relatively uniform flow
        fields, and a number greater than zero in diverging/converging flow
        fields. Generally, a value between zero and four is adequate.
    npmax: int
        is the maximum number of particles allowed per cell. If the number of
        particles in a cell exceeds NPMAX, all particles are removed from that
        cell and replaced by a new set of particles equal to NPH to maintain
        mass balance. Generally, NPMAX can be set to approximately twice of NPH.
    dchmoc: real
        is the critical Relative Concentration Gradient for controlling the
        selective use of either MOC or MMOC in the HMOC solution scheme. The MOC
        solution is selected at cells where the Relative Concentration Gradient
        is greater than DCHMOC; The MMOC solution is selected at cells where the
        Relative Concentration Gradient is less than or equal to DCHMOC
    """

    __slots__ = (
        "courant",
        "tracking",
        "weighting_factor",
        "dconcentration_epsilon",
        "nplane",
        "nparticles_no_advection",
        "nparticles_advection",
        "cell_min_nparticles",
        "cell_max_nparticles",
        "sink_particle_placement",
        "sink_nparticles",
        "dconcentration_hybrid",
    )
    _pkg_id = "adv"

    def __init__(
        self,
        courant=0.75,
        tracking="hybrid",
        weighting_factor=0.5,
        dconcentration_epsilon=1.0e-5,
        nplane=2,
        nparticles_no_advection=10,
        nparticles_advection=40,
        cell_min_nparticles=5,
        cell_max_nparticles=80,
        sink_particle_placement=2,
        sink_nparticles=40,
        dconcentration_hybrid=1.0e-4,
    ):
        super(__class__, self).__init__()
        self["courant"] = courant
        self["tracking"] = tracking
        self["weighting_factor"] = weighting_factor
        self["dconcentration_epsilon"] = dconcentration_epsilon
        self["nplane"] = nplane
        self["nparticles_no_advection"] = nparticles_no_advection
        self["nparticles_advection"] = nparticles_advection
        self["cell_min_nparticles"] = cell_min_nparticles
        self["cell_max_nparticles"] = cell_max_nparticles
        self["sink_particle_placement"] = sink_particle_placement
        self["sink_nparticles"] = sink_nparticles
        self["dconcentration_hybrid"] = dconcentration_hybrid


class AdvectionTVD(Package):
    """
    Total Variation Diminishing (TVD) formulation (ULTIMATE, MIXELM = -1).

    Attributes
    ----------
    courant : float
        Courant number (PERCEL) is the number of cells (or a fraction of a cell)
        advection will be allowed in any direction in one transport step. For
        implicit finite-difference or particle tracking based schemes, there is
        no limit on PERCEL, but for accuracy reasons, it is generally not set
        much greater than one. Note, however, that the PERCEL limit is checked
        over the entire model grid. Thus, even if PERCEL > 1, advection may not
        be more than one cell’s length at most model locations. For the explicit
        finite-difference, PERCEL is also a stability constraint, which must not
        exceed one and will be automatically reset to one if a value greater
        than one is specified.
    """

    __slots__ = ("courant",)
    _pkg_id = "adv"

    _template = "[adv]\n" "    mixelm = -1\n" "    percel = {courant}\n"

    def __init__(self, courant=0.75):
        super(__class__, self).__init__()
        self["courant"] = courant

    def _pkgcheck(self, ibound=None):
        self._check_positive(["courant"])
