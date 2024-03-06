from __future__ import annotations

from collections import defaultdict
from typing import Optional

import cftime
import numpy as np

from imod.mf6 import ConstantHead
from imod.mf6.clipped_boundary_condition_creator import create_clipped_boundary
from imod.mf6.model import Modflow6Model
from imod.mf6.utilities.regrid import RegridderType
from imod.typing import GridDataArray
from imod.mf6.interfaces.iregridpackage import IRegridPackage

class GroundwaterFlowModel(Modflow6Model):
    _mandatory_packages = ("npf", "ic", "oc", "sto")
    _model_id = "gwf6"
    _template = Modflow6Model._initialize_template("gwf-nam.j2")

    def __init__(
        self,
        listing_file: Optional[str] = None,
        print_input: bool = False,
        print_flows: bool = False,
        save_flows: bool = False,
        newton: bool = False,
        under_relaxation: bool = False,
    ):
        super().__init__()
        self._options = {
            "listing_file": listing_file,
            "print_input": print_input,
            "print_flows": print_flows,
            "save_flows": save_flows,
            "newton": newton,
            "under_relaxation": under_relaxation,
        }


    def _get_unique_regridder_types(self) -> defaultdict[RegridderType, list[str]]:
        """
        This function loops over the packages and  collects all regridder-types that are in use.
        """
        methods: defaultdict = defaultdict(list)
        for pkg_name, pkg in self.items():
            if  isinstance(pkg, IRegridPackage):
                pkg_methods = pkg.get_regrid_methods()
                for variable in pkg_methods:
                    if (
                        variable in pkg.dataset.data_vars
                        and pkg.dataset[variable].values[()] is not None
                    ):
                        regriddertype = pkg_methods[variable][0]
                        functiontype = pkg_methods[variable][1]
                        if functiontype not in  methods[regriddertype]:
                            methods[regriddertype].append(functiontype)
            else:
                raise NotImplementedError(
                    f"regridding is not implemented for package {pkg_name} of type {type(pkg)}"
                )
        return methods

    def clip_box(
        self,
        time_min: Optional[cftime.datetime | np.datetime64 | str] = None,
        time_max: Optional[cftime.datetime | np.datetime64 | str] = None,
        layer_min: Optional[int] = None,
        layer_max: Optional[int] = None,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        state_for_boundary: Optional[GridDataArray] = None,
    ):
        clipped = super().clip_box(
            time_min, time_max, layer_min, layer_max, x_min, x_max, y_min, y_max
        )

        clipped_boundary_condition = self.__create_boundary_condition_clipped_boundary(
            self, clipped, state_for_boundary
        )
        if clipped_boundary_condition is not None:
            clipped["chd_clipped"] = clipped_boundary_condition

        clipped.purge_empty_packages()

        return clipped

    def __create_boundary_condition_clipped_boundary(
        self,
        original_model: Modflow6Model,
        clipped_model: Modflow6Model,
        state_for_boundary: Optional[GridDataArray],
    ):
        unassigned_boundary_original_domain = (
            self.__create_boundary_condition_for_unassigned_boundary(
                original_model, state_for_boundary
            )
        )

        return self.__create_boundary_condition_for_unassigned_boundary(
            clipped_model, state_for_boundary, [unassigned_boundary_original_domain]
        )

    @staticmethod
    def __create_boundary_condition_for_unassigned_boundary(
        model: Modflow6Model,
        state_for_boundary: Optional[GridDataArray],
        additional_boundaries: Optional[list[ConstantHead]] = None,
    ):
        if state_for_boundary is None:
            return None

        constant_head_packages = [
            pkg for name, pkg in model.items() if isinstance(pkg, ConstantHead)
        ]

        additional_boundaries = [
            item for item in additional_boundaries or [] if item is not None
        ]

        constant_head_packages.extend(additional_boundaries)

        return create_clipped_boundary(
            model.domain, state_for_boundary, constant_head_packages
        )

    def is_use_newton(self):
        return self._options["newton"]

    def set_newton(self, is_newton: bool) -> None:
        self._options["newton"] = is_newton

    def update_buoyancy_package(self, transport_models_per_flow_model)->None:
        '''
        If the simulation is partitioned, then the buoyancy package, if present, 
        must be updated for the renamed transport models.
        '''
        buoyancy_key = self._get_pkgkey("buy")
        if buoyancy_key is None:
            return
        buoyancy_package = self[buoyancy_key]
        transport_models_old = buoyancy_package.get_transport_model_names()
        if len(transport_models_old) == len(transport_models_per_flow_model):
            buoyancy_package.update_transport_models(transport_models_per_flow_model)
        


            