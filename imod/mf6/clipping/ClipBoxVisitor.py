from copy import deepcopy
from typing import Dict, Optional
from uuid import UUID

import cftime
import numpy as np
import xarray as xr

from imod.common.interfaces.ilinedatapackage import ILineDataPackage
from imod.common.interfaces.imodel import IModel
from imod.common.interfaces.ipackage import IPackage
from imod.common.interfaces.ipointdatapackage import ILayeredPointDataPackage, IPointDataPackage, IZRangePointDataPackage
from imod.common.interfaces.isimulation import ISimulation
from imod.common.interfaces.ivisitee import IVisitor

from imod.common.utilities.clip import bounding_polygon_from_line_data_and_clip_box, clip_box_dataset, clip_line_gdf_by_bounding_polygon
from imod.mf6.gwfgwt import GWFGWT
from imod.mf6.utilities.clipped_bc_creator import StateType, create_clipped_boundary
from imod.typing import GridDataArray
from imod.util.structured import values_within_range
from imod.typing.grid import is_spatial_grid


class ClipBoxVisitor(IVisitor):
    def __init__(
        self,
        time_min: Optional[cftime.datetime | np.datetime64 | str] = None,
        time_max: Optional[cftime.datetime | np.datetime64 | str] = None,
        layer_min: Optional[int] = None,
        layer_max: Optional[int] = None,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        model_to_boundary_state: Optional[dict[UUID, GridDataArray]] = None,
        ignore_time_purge_empty: Optional[bool] = None,
    ) -> None:
        self.time_min = time_min
        self.time_max = time_max
        self.layer_min = layer_min
        self.layer_max = layer_max
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.model_to_boundary_state = model_to_boundary_state
        self.ignore_time_purge_empty = ignore_time_purge_empty

        if self.model_to_boundary_state is None:
            self.model_to_boundary_state = {}

    def visit_simulation(self, simulation: ISimulation) -> ISimulation:
        clipped = type(simulation)(
            name=simulation.name, validation_settings=simulation._validation_context
        )

        for key, value in simulation.items():
            if isinstance(value, IModel) or isinstance(value, IPackage):
                clipped[key] = value.accept(self)
            elif isinstance(value, list) and all(
                isinstance(item, GWFGWT) for item in value
            ):
                continue
            else:
                raise ValueError(
                    f"object {key} of type {type(value)} cannot be clipped."
                )
        return clipped

    def visit_model(self, model: IModel) -> IModel:
        clip_box_args = {
            "time_min": self.time_min,
            "time_max": self.time_max,
            "layer_min": self.layer_min,
            "layer_max": self.layer_max,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
        }

        # Clip the model domain
        clipped = type(model)(**model._options)

        for key, pkg in model.items():
            clipped[key] = pkg.accept(self)

        state_for_boundary = self.model_to_boundary_state.get(model.uuid) 
        clipped_boundary_condition = _create_boundary_condition_clipped_boundary(
            model, clipped, state_for_boundary, clip_box_args
        )

        # Add clipped boundary condition package to model
        state_pkg_id = model._boundary_state_pkg_type._pkg_id
        pkg_name = f"{state_pkg_id}_clipped"
        if clipped_boundary_condition is not None:
            clipped[pkg_name] = clipped_boundary_condition

        clipped.purge_empty_packages(ignore_time=self.ignore_time_purge_empty)

        return clipped
    
    def visit_package(self, package: IPackage) -> IPackage:
        
        if isinstance(package, ILineDataPackage):
            clipped = deepcopy(package)
            if self.x_min or self.x_max or self.y_min or self.y_max:
                # Create bounding polygon
                bounding_gdf = bounding_polygon_from_line_data_and_clip_box(
                    package.line_data, self.x_min, self.x_max, self.y_min, self.y_max
                )
                clipped.line_data = clip_line_gdf_by_bounding_polygon(
                    package.line_data, bounding_gdf
                )
        elif isinstance(package, ILayeredPointDataPackage):
            # The super method will select in the time dimension without issues.
            selection = clip_box_dataset(
                package.dataset,
                self.time_min,
                self.time_max,
            )
            
            cls = type(package)
            clipped =  cls._from_dataset(selection)
            
            ds = clipped.dataset
            
            # Initiate array of True with right shape to deal with case no spatial
            # selection needs to be done.
            in_bounds = np.full(ds.sizes["index"], True)
            # Select all variables along "index" dimension
            in_bounds &= values_within_range(clipped.x, self.x_min, self.x_max)
            in_bounds &= values_within_range(clipped.y, self.y_min, self.y_max)
            
            in_bounds &= values_within_range(clipped.layer, self.layer_min, self.layer_max)
                
            # Replace dataset with reduced dataset based on booleans
            clipped.dataset = ds.loc[{"index": in_bounds}]
        elif isinstance(package, IZRangePointDataPackage):
             # The super method will select in the time dimension without issues.
            selection = clip_box_dataset(
                package.dataset,
                self.time_min,
                self.time_max,
            )
            
            cls = type(package)
            clipped =  cls._from_dataset(selection)
            
            ds = clipped.dataset
            
            if package.parent is not None:
                top, bottom, _ = package.parent._get_domain_geometry()
            else:
                top = bottom = None

            z_max, in_bounds_z_max = self._find_well_value_at_layer(ds, top, self.layer_max)
            z_min, in_bounds_z_min = self._find_well_value_at_layer(ds, bottom, self.layer_min)
            
            # Prior to the actual clipping of z_max/z_min, replace the dataset in case a
            # spatial selection needs to be done when a spatial grid is present (top/bottom).
            ds = ds.loc[{"index": in_bounds_z_max & in_bounds_z_min}]
            
            if z_max is not None:
                ds["screen_top"] = ds["screen_top"].clip(None, z_max)
            if z_min is not None:
                ds["screen_bottom"] = ds["screen_bottom"].clip(z_min, None)
                
            # Initiate array of True with right shape to deal with case no spatial
            # selection needs to be done.
            in_bounds = np.full(ds.sizes["index"], True)
            # Select all variables along "index" dimension
            in_bounds &= values_within_range(clipped.x, self.x_min, self.x_max)
            in_bounds &= values_within_range(clipped.y, self.y_min, self.y_max)
            in_bounds &= values_within_range(ds["screen_top"], z_min, z_max)
            in_bounds &= values_within_range(ds["screen_bottom"], z_min, z_max)
            # remove wells where the screen bottom and top are the same
            in_bounds &= abs(ds["screen_bottom"] - ds["screen_top"]) > 1e-5
            # Replace dataset with reduced dataset based on booleans
            clipped.dataset = ds.loc[{"index": in_bounds}]
            
            
        else:
            selection = clip_box_dataset(
                package.dataset,
                self.time_min,
                self.time_max,
                self.layer_min,
                self.layer_max,
                self.x_min,
                self.x_max,
                self.y_min,
                self.y_max,
            )
            
            cls = type(package)
            clipped =  cls._from_dataset(selection)
            
        return clipped
    
    @staticmethod
    def _find_well_value_at_layer(
        well_dataset: xr.Dataset, grid: GridDataArray, layer: Optional[int]
    ):
        from imod.select.points import points_values
        
        value = None if layer is None else grid.isel(layer=layer)

        # if value is a grid select the values at the well locations and drop the dimensions
        if (value is not None) and is_spatial_grid(value):
            value = points_values(
                value,
                x=well_dataset["x"].values,
                y=well_dataset["y"].values,
                out_of_bounds="ignore",
            )
            in_bounds = np.full(well_dataset.sizes["index"], False)
            in_bounds[value["index"]] = True
            value = value.drop_vars(lambda x: x.coords)
        else:
            in_bounds = np.full(well_dataset.sizes["index"], True)

        return value, in_bounds



############## MODEL METHODS ##############
def _create_boundary_condition_for_unassigned_boundary(
    model: IModel,
    state_for_boundary: Optional[GridDataArray],
    additional_boundaries: list[Optional[StateType]] = [None],
) -> Optional[StateType]:
    if state_for_boundary is None:
        return None

    pkg_type = model._boundary_state_pkg_type
    constant_state_packages = [
        pkg for _, pkg in model.items() if isinstance(pkg, pkg_type)
    ]

    additional_boundaries = [
        item for item in additional_boundaries or [] if item is not None
    ]

    constant_state_packages.extend(additional_boundaries)

    return create_clipped_boundary(
        model.domain, state_for_boundary, constant_state_packages, pkg_type
    )


def _create_boundary_condition_clipped_boundary(
    original_model: IModel,
    clipped_model: IModel,
    state_for_boundary: Optional[GridDataArray],
    clip_box_args: Dict,
) -> Optional[StateType]:
    # Create temporary boundary condition for the original model boundary. This
    # is used later to see which boundaries can be ignored as they were already
    # present in the original model. We want to just end up with the boundary
    # created by the clip.
    unassigned_boundary_original_domain = (
        _create_boundary_condition_for_unassigned_boundary(
            original_model, state_for_boundary
        )
    )
    # Clip the unassigned boundary to the clipped model's domain, required to
    # avoid topological errors later.
    if unassigned_boundary_original_domain is not None:
        unassigned_boundary_clipped = unassigned_boundary_original_domain.clip_box(
            **clip_box_args
        )
    else:
        unassigned_boundary_clipped = None

    if state_for_boundary is not None:
        # Clip box as dataset, temporarily add variable name to convert to
        # dataset, then turn back into DataArray.
        varname = original_model._boundary_state_pkg_type._period_data[0]
        state_for_boundary = state_for_boundary.to_dataset(name=varname)
        state_for_boundary_clipped = clip_box_dataset(
            state_for_boundary, **clip_box_args
        )[varname]
    else:
        state_for_boundary_clipped = None

    bc_constant_pkg = _create_boundary_condition_for_unassigned_boundary(
        clipped_model, state_for_boundary_clipped, [unassigned_boundary_clipped]
    )

    # Remove all indices before first timestep of state_for_clipped_boundary.
    # This to prevent empty dataarrays unnecessarily being made for these
    # indices, which can lead to them to be removed when purging empty packages
    # with ignore_time=True. Unfortunately, this is needs to be handled here and
    # not in _create_boundary_condition_for_unassigned_boundary, as otherwise
    # this function is called twice which could result in broadcasting errors in
    # the second call if the time domain of state_for_boundary and assigned
    # packages have no overlap.
    if (
        (state_for_boundary is not None)
        and (state_for_boundary.indexes.get("time") is not None)
        and (bc_constant_pkg is not None)
    ):
        start_time = state_for_boundary.indexes["time"][0]
        bc_constant_pkg.dataset = bc_constant_pkg.dataset.sel(
            time=slice(start_time, None)
        )

    return bc_constant_pkg
