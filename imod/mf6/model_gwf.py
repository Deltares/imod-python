from __future__ import annotations

from datetime import datetime
from typing import Optional, cast

import cftime
import numpy as np

from imod.flow.ghb import GeneralHeadBoundary
from imod.logging import init_log_decorator
from imod.logging.logging_decorators import standard_log_decorator
from imod.mf6 import ConstantHead
from imod.mf6.clipped_boundary_condition_creator import create_clipped_boundary
from imod.mf6.dis import StructuredDiscretization
from imod.mf6.drn import Drainage
from imod.mf6.hfb import SingleLayerHorizontalFlowBarrierResistance
from imod.mf6.ic import InitialConditions
from imod.mf6.model import Modflow6Model
from imod.mf6.npf import NodePropertyFlow
from imod.mf6.rch import Recharge
from imod.mf6.regrid.regrid_schemes import (
    ConstantHeadRegridMethod,
    DiscretizationRegridMethod,
    DrainageRegridMethod,
    InitialConditionsRegridMethod,
    NodePropertyFlowRegridMethod,
    RechargeRegridMethod,
    RegridMethodType,
    RiverRegridMethod,
    StorageCoefficientRegridMethod,
)
from imod.mf6.riv import River
from imod.mf6.sto import StorageCoefficient
from imod.mf6.utilities.regrid import RegridderWeightsCache
from imod.mf6.wel import Well
from imod.prepare.topsystem.default_allocation_methods import (
    SimulationAllocationOptions,
    SimulationDistributingOptions,
)
from imod.typing import GridDataArray


class GroundwaterFlowModel(Modflow6Model):
    """
    The GroundwaterFlowModel (GWF) simulates flow of (liquid) groundwater.
    More information can be found here:
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.4.2.pdf#page=27

    Parameters
    ----------

    listing_file: Optional[str] = None
        name of the listing file to create for this GWF model. If not specified,
        then the name of the list file will be the basename of the GWF model
        name file and the 'lst' extension.
    print_input: bool = False
        keyword to indicate that the list of all model stress package
        information will be written to the listing file immediately after it is
        read.
    print_flows: bool = False
        keyword to indicate that the list of all model package flow rates will
        be printed to the listing file for every stress period time step in
        which "BUDGET PRINT" is specified in Output Control.
    save_flows: bool = False
        indicate that all model package flow terms will be written to the file
        specified with "BUDGET FILEOUT" in Output Control.
    newton: bool = False
        activates the Newton-Raphson formulation for groundwater flow between
        connected, convertible groundwater cells and stress packages that
        support calculation of Newton-Raphson terms for groundwater exchanges.
    under_relaxation: bool = False,
        indicates whether the groundwater head in a cell will be under-relaxed when
        water levels fall below the bottom of the model below any given cell. By
        default, Newton-Raphson UNDER_RELAXATION is not applied.
    """

    _mandatory_packages = ("npf", "ic", "oc", "sto")
    _model_id = "gwf6"
    _template = Modflow6Model._initialize_template("gwf-nam.j2")

    @init_log_decorator()
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

    def update_buoyancy_package(self, transport_models_per_flow_model) -> None:
        """
        If the simulation is partitioned, then the buoyancy package, if present,
        must be updated for the renamed transport models.
        """
        buoyancy_key = self._get_pkgkey("buy")
        if buoyancy_key is None:
            return
        buoyancy_package = self[buoyancy_key]
        transport_models_old = buoyancy_package.get_transport_model_names()
        if len(transport_models_old) == len(transport_models_per_flow_model):
            buoyancy_package.update_transport_models(transport_models_per_flow_model)

    @classmethod
    @standard_log_decorator()
    def from_imod5_data(
        cls,
        imod5_data: dict[str, dict[str, GridDataArray]],
        period_data: dict[str, list[datetime]],
        allocation_options: SimulationAllocationOptions,
        distributing_options: SimulationDistributingOptions,
        times: list[datetime],
        regridder_types: dict[str, RegridMethodType],
    ) -> "GroundwaterFlowModel":
        """
        Imports a GroundwaterFlowModel (GWF) from the data in an IMOD5 project file.
        It adds the packages for which import from imod5 is supported.
        Some packages (like OC) must be added manually later.


        Parameters
        ----------
        imod5_data: dict[str, dict[str, GridDataArray]]
            dictionary containing the arrays mentioned in the project file as xarray datasets,
            under the key of the package type to which it belongs
        allocation_options: SimulationAllocationOptions
            object containing the allocation options per package type.
            If you want a package to have a different allocation option,
            then it should be imported separately
        distributing_options: SimulationDistributingOptions
            object containing the conductivity distribution options per package type.
            If you want a package to have a different allocation option,
            then it should be imported separately
        time_min: datetime
            Begin-time of the simulation.
        time_max: datetime
            End-time of the simulation.
        regridder_types:  dict[str, RegridMethodType]
            the key is the package name. The value is a subclass of RegridMethodType.

        Returns
        -------
        A GWF model containing the packages that could be imported form IMOD5. Users must still
        add the OC package to the model.

        """
        # first import the singleton packages
        # import discretization
        regrid_cache = RegridderWeightsCache()

        dis_pkg = StructuredDiscretization.from_imod5_data(
            imod5_data,
            cast(DiscretizationRegridMethod, regridder_types.get("dis")),
            regrid_cache,
            False,
        )
        grid = dis_pkg.dataset["idomain"]

        # import npf
        npf_pkg = NodePropertyFlow.from_imod5_data(
            imod5_data,
            grid,
            cast(NodePropertyFlowRegridMethod, regridder_types.get("npf")),
            regrid_cache,
        )

        # import sto
        sto_pkg = StorageCoefficient.from_imod5_data(
            imod5_data,
            grid,
            cast(StorageCoefficientRegridMethod, regridder_types.get("sto")),
            regrid_cache,
        )

        # import initial conditions
        ic_pkg = InitialConditions.from_imod5_data(
            imod5_data,
            grid,
            cast(InitialConditionsRegridMethod, regridder_types.get("ic")),
            regrid_cache,
        )

        # import recharge
        rch_pkg = Recharge.from_imod5_data(
            imod5_data,
            dis_pkg,
            cast(RechargeRegridMethod, regridder_types.get("ic")),
            regrid_cache,
        )

        result = GroundwaterFlowModel()
        result["dis"] = dis_pkg
        result["npf"] = npf_pkg
        result["sto"] = sto_pkg
        result["ic"] = ic_pkg
        result["rch"] = rch_pkg

        # now import the non-singleton packages'

        # import wells
        # import wells
        imod5_keys = list(imod5_data.keys())
        wel_keys = [key for key in imod5_keys if key[0:3] == "wel"]
        for wel_key in wel_keys:
            result[wel_key] = Well.from_imod5_data(wel_key, imod5_data, times)

                    
        imod5_keys = list(imod5_data.keys())
        ghb_keys = [key for key in imod5_keys if key[0:3] == "ghb"]
        for ghb_key in ghb_keys:
            ghb_pkg = GeneralHeadBoundary.from_imod5_data(
                ghb_key,
                imod5_data,
                period_data,
                dis_pkg,
                npf_pkg,
                time_min,
                time_max,
                allocation_options.ghb,
                distributing_options.ghb,
                regridder_types,
            )
            result[ghb_key] = ghb_pkg

        # import drainage

        drainage_keys = [key for key in imod5_keys if key[0:3] == "drn"]
        for drn_key in drainage_keys:
            drn_pkg = Drainage.from_imod5_data(
                drn_key,
                imod5_data,
                period_data,
                dis_pkg,
                npf_pkg,
                allocation_options.drn,
                distributing_option=distributing_options.drn,
                time_min=times[0],
                time_max=times[-1],
                regridder_types=cast(
                    DrainageRegridMethod, regridder_types.get(drn_key)
                ),
                regrid_cache=regrid_cache,
            )
            result[drn_key] = drn_pkg

        # import rivers ( and drainage to account for infiltration factor)
        riv_keys = [key for key in imod5_keys if key[0:3] == "riv"]
        for riv_key in riv_keys:
            riv_pkg, riv_drn_pkg = River.from_imod5_data(
                riv_key,
                imod5_data,
                period_data,
                dis_pkg,
                times[0],
                times[-1],
                allocation_options.riv,
                distributing_options.riv,
                cast(RiverRegridMethod, regridder_types.get(riv_key)),
                regrid_cache,
            )
            if riv_pkg is not None:
                result[riv_key + "riv"] = riv_pkg
            if riv_drn_pkg is not None:
                result[riv_key + "drn"] = riv_drn_pkg

        # import hfb
        hfb_keys = [key for key in imod5_keys if key[0:3] == "hfb"]
        if len(hfb_keys) != 0:
            for hfb_key in hfb_keys:
                result[hfb_key] = (
                    SingleLayerHorizontalFlowBarrierResistance.from_imod5_dataset(
                        hfb_key, imod5_data
                    )
                )

        # import chd
        chd_keys = [key for key in imod5_keys if key[0:3] == "chd"]
        if len(chd_keys) == 0:
            result["chd_from_shd"] = ConstantHead.from_imod5_shd_data(
                imod5_data,
                dis_pkg,
                cast(ConstantHeadRegridMethod, regridder_types.get("chd_from_shd")),
                regrid_cache,
            )
        else:
            for chd_key in chd_keys:
                result[chd_key] = ConstantHead.from_imod5_data(
                    chd_key,
                    imod5_data,
                    dis_pkg,
                    cast(ConstantHeadRegridMethod, regridder_types.get(chd_key)),
                    regrid_cache,
                )

        return result
