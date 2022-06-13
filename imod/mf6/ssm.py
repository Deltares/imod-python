from typing import Dict

from imod.mf6.pkgbase import BoundaryCondition, Package


class Transport_Sink_Sources(BoundaryCondition):
    _pkg_id = "ssm"
    _template = BoundaryCondition._initialize_template(_pkg_id)

    def __init__(self, flow_packages: Dict[str, Package], aux_variable_name: str):

        super().__init__()
        self.flow_boundary_packages = flow_packages
        self.aux_variable_name = aux_variable_name

    def render(self, directory, pkgname, globaltimes, binary):
        d = {}
        flow_packages_data = {}
        for flowpack_name, flowpack in self.flow_boundary_packages.items():
            if not isinstance(flowpack, BoundaryCondition):
                continue
            flow_packages_data[flowpack_name] = flowpack.string_data[
                "concentration_boundary_type"
            ]
        d["flowboundaries"] = flow_packages_data
        d["auxname"] = self.aux_variable_name
        return self._template.render(d)
