from imod.mf6.pkgbase import BoundaryCondition


class Transport_Sink_Sources(BoundaryCondition):
    _pkg_id = "ssm"
    _flow_boundary_packages = None
    _template = BoundaryCondition._initialize_template(_pkg_id)
    _aux_variable_name = ""

    def __init__(self, flow_packages, aux_variable_name):
        self._flow_boundary_packages = flow_packages
        self._aux_variable_name = aux_variable_name

    def render(self, directory, pkgname, globaltimes, binary):
        d = {}
        flow_packages_data = {}
        for flowpack_name, flowpack in self._flow_boundary_packages.items():
            if not isinstance(flowpack, BoundaryCondition):
                continue
            flow_packages_data[flowpack_name] = flowpack.string_data[
                "concentration_boundary_type"
            ]
        d["flowboundaries"] = flow_packages_data
        d["auxname"] = self._aux_variable_name
        return self._template.render(d)
