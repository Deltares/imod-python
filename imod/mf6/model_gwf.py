from __future__ import annotations

import textwrap
from datetime import datetime
from typing import Optional, cast

import cftime
import numpy as np

from imod.logging import init_log_decorator
from imod.logging.logging_decorators import standard_log_decorator
from imod.mf6 import ConstantHead
from imod.mf6.clipped_boundary_condition_creator import create_clipped_boundary
from imod.mf6.dis import StructuredDiscretization
from imod.mf6.drn import Drainage
from imod.mf6.ghb import GeneralHeadBoundary
from imod.mf6.hfb import SingleLayerHorizontalFlowBarrierResistance
from imod.mf6.ic import InitialConditions
from imod.mf6.model import Modflow6Model
from imod.mf6.npf import NodePropertyFlow
from imod.mf6.rch import Recharge
from imod.mf6.regrid.regrid_schemes import (
    ConstantHeadRegridMethod,
    DiscretizationRegridMethod,
    DrainageRegridMethod,
    GeneralHeadBoundaryRegridMethod,
    InitialConditionsRegridMethod,
    NodePropertyFlowRegridMethod,
    RechargeRegridMethod,
    RegridMethodType,
    RiverRegridMethod,
    StorageCoefficientRegridMethod,
)
from imod.mf6.riv import River
from imod.mf6.sto import StorageCoefficient
from imod.mf6.utilities.chd_concat import concat_layered_chd_packages
from imod.mf6.utilities.regrid import RegridderWeightsCache
from imod.mf6.wel import LayeredWell, Well
from imod.prepare.topsystem.default_allocation_methods import (
    SimulationAllocationOptions,
    SimulationDistributingOptions,
)
from imod.typing import GridDataArray, StressPeriodTimesType
from imod.typing.grid import zeros_like


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
        times: list[datetime],
        allocation_options: Optional[SimulationAllocationOptions] = None,
        distributing_options: Optional[SimulationDistributingOptions] = None,
        regridder_types: Optional[dict[str, RegridMethodType]] = None,
    ) -> "GroundwaterFlowModel":
        """
        Imports a GroundwaterFlowModel (GWF) from the data in an iMOD5 project
        file and puts it in a simulation. Quasi-3D iMOD5 models, i.e. models
        where there is only horizontal flow in aquifers and vertical flow in
        aquitards, are not supported.

        This method adds all static and boundary condition packages from the
        projectfile to the simulation. Output Control (OC) must be added
        manually after importing.

        Parameters
        ----------
        imod5_data: dict[str, dict[str, GridDataArray]]
            dictionary containing the arrays mentioned in the project file as xarray datasets,
            under the key of the package type to which it belongs
        period_data: dict[str, list[datetime]]
            dictionary containing the package names mapped to a list of repeated
            stress periods. These are set as ``repeat_stress``.
        times: list[datetime]
            Time discretization of the simulation. These times are used for the
            following:

                * Times of wells with associated timeseries are resampled to these times
                * Start- and end times in the list are used to repeat the stresses
                  of periodic data (e.g. river stages in iMOD5 for "summer", "winter")

        allocation_options: SimulationAllocationOptions
            object containing the allocation options per package type.
            If you want a package to have a different allocation option,
            then it should be imported separately
        distributing_options: SimulationDistributingOptions
            object containing the conductivity distribution options per package type.
            If you want a package to have a different allocation option,
            then it should be imported separately
        regridder_types:  dict[str, RegridMethodType]
            the key is the package name. The value is a subclass of RegridMethodType.

        Returns
        -------
        A GWF model containing the packages that could be imported form IMOD5. Users must still
        add the OC package to the model.

        """
        if allocation_options is None:
            allocation_options = SimulationAllocationOptions()
        if distributing_options is None:
            distributing_options = SimulationDistributingOptions()
        if regridder_types is None:
            regridder_types = {}

        # first import the singleton packages
        # import dis
        regrid_cache = RegridderWeightsCache()
        result = GroundwaterFlowModel()

        dis_pkg = StructuredDiscretization.from_imod5_data(
            imod5_data,
            cast(DiscretizationRegridMethod, regridder_types.get("dis")),
            regrid_cache,
            False,
        )
        grid = dis_pkg.dataset["idomain"]
        result["dis"] = dis_pkg

        # import npf
        npf_pkg = NodePropertyFlow.from_imod5_data(
            imod5_data,
            grid,
            cast(NodePropertyFlowRegridMethod, regridder_types.get("npf")),
            regrid_cache,
        )
        result["npf"] = npf_pkg

        # import sto
        is_transient = "sto" in imod5_data.keys()
        if is_transient:
            result["sto"] = StorageCoefficient.from_imod5_data(
                imod5_data,
                grid,
                cast(StorageCoefficientRegridMethod, regridder_types.get("sto")),
                regrid_cache,
            )
        else:
            zeros = zeros_like(grid, dtype=float)
            result["sto"] = StorageCoefficient(
                storage_coefficient=zeros,
                specific_yield=zeros,
                transient=False,
                convertible=zeros.astype(int),
            )

        # import initial conditions
        result["ic"] = InitialConditions.from_imod5_data(
            imod5_data,
            grid,
            cast(InitialConditionsRegridMethod, regridder_types.get("ic")),
            regrid_cache,
        )

        # import recharge
        if "rch" in imod5_data.keys():
            result["rch"] = Recharge.from_imod5_data(
                imod5_data,
                dis_pkg,
                cast(RechargeRegridMethod, regridder_types.get("rch")),
                regrid_cache,
            )

        # now import the non-singleton packages'
        imod5_keys = list(imod5_data.keys())

        # import wells
        wel_times: StressPeriodTimesType = times if is_transient else "steady-state"
        wel_keys = [key for key in imod5_keys if key[0:3] == "wel"]
        for wel_key in wel_keys:
            wel_key_truncated = wel_key[:16]
            if wel_key_truncated in result.keys():
                # Remove this when https://github.com/Deltares/imod-python/issues/1167
                # is resolved
                msg = textwrap.dedent(
                    f"""Truncated key: '{wel_key_truncated}' already assigned to
                    imported model, please rename wells so that unique names are
                    formed after they are truncated to 16 characters for MODFLOW
                    6.
                    """
                )
                raise KeyError(msg)

            wel_layer = imod5_data[wel_key]["layer"].values
            is_allocated = np.any(wel_layer == 0)
            wel_args = (wel_key, imod5_data, wel_times)
            result[wel_key_truncated] = (
                Well.from_imod5_data(*wel_args)
                if is_allocated
                else LayeredWell.from_imod5_data(*wel_args)
            )

        if "cap" in imod5_keys:
            result["msw-sprinkling"] = LayeredWell.from_imod5_cap_data(imod5_data)  # type: ignore
            result["msw-rch"] = Recharge.from_imod5_cap_data(imod5_data)  # type: ignore

        # import ghb's
        ghb_keys = [key for key in imod5_keys if key[0:3] == "ghb"]
        for ghb_key in ghb_keys:
            ghb_pkg = GeneralHeadBoundary.from_imod5_data(
                ghb_key,
                imod5_data,
                period_data,
                dis_pkg,
                npf_pkg,
                times[0],
                times[-1],
                allocation_options.ghb,
                distributing_options.ghb,
                regridder_types=cast(
                    GeneralHeadBoundaryRegridMethod, regridder_types.get(ghb_key)
                ),
                regrid_cache=regrid_cache,
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
                times[0],
                times[-1],
                allocation_options.drn,
                distributing_option=distributing_options.drn,
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
                npf_pkg,
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
                    SingleLayerHorizontalFlowBarrierResistance.from_imod5_data(
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
            chd_packages = {}
            for chd_key in chd_keys:
                chd_packages[chd_key] = ConstantHead.from_imod5_data(
                    chd_key,
                    imod5_data,
                    dis_pkg,
                    cast(ConstantHeadRegridMethod, regridder_types.get(chd_key)),
                    regrid_cache,
                )
            merged_chd = concat_layered_chd_packages(
                "chd", chd_packages, remove_merged_packages=True
            )
            if merged_chd is not None:
                result["chd_merged"] = merged_chd
            for key, chd_package in chd_packages.items():
                result[key] = chd_package

        return result
