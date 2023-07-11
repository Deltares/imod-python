from imod.mf6.boundary_condition import BoundaryCondition


class StreamFlowRouting(BoundaryCondition):
    """
    Stream flow routing package
    """

    _pkg_id = "sfr"
    _period_data = []
    _keyword_map = {}
    _template = BoundaryCondition._initialize_template(_pkg_id)

    def __init__(
        self,
        length,
        width,
        gradient,
        top_elevation,
        bed_thickness,
        bed_conductivity,
        fraction_upstream,
        manning_n,
        stage,
        inflow,
        rainfall,
        evaporation,
        runoff,
        maximum_picard_iterations=None,
        maximum_iterations=None,
        maximum_depth_change=None,
        print_input=False,
        print_stage=False,
        print_flows=False,
        save_flows=False,
        stage_fileout=None,
        budget_fileout=None,
        convergence_fileout=None,
    ):
        pass
