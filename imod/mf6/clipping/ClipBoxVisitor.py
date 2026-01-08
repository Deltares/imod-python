import cftime
from typing import Dict, Optional
import numpy as np
from typing import Optional

from imod.common.interfaces.ipackage import IPackage
from imod.common.interfaces.isimulation import ISimulation
from imod.common.interfaces.imodel import IModel
from imod.common.interfaces.ivisitee import IVisitor

from imod.mf6.gwfgwt import GWFGWT

from imod.mf6.utilities.clipped_bc_creator import StateType, create_clipped_boundary
from imod.typing import GridDataArray
from imod.common.utilities.clip import clip_box_dataset


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
      states_for_boundary: Optional[dict[str, GridDataArray]] = None,
      ignore_time_purge_empty: Optional[bool] = None) -> None:
    
      self.time_min = time_min
      self.time_max = time_max
      self.layer_min = layer_min
      self.layer_max = layer_max
      self.x_min = x_min
      self.x_max = x_max
      self.y_min = y_min
      self.y_max = y_max
      self.states_for_boundary = states_for_boundary
      self.ignore_time_purge_empty = ignore_time_purge_empty
      
      self.model_to_boundary_state = {}
      

    def visit_simulation(self, simulation: ISimulation, name: str) -> ISimulation:
        # Create a mapping of model name to boundary state package
        if self.states_for_boundary is not None:
            for model_name, model in simulation.get_models().items():
                boundary_state = self.states_for_boundary.get(model_name)
                self.model_to_boundary_state[model.uuid] = boundary_state
      
        clipped = type(simulation)(
                name=name, validation_settings=simulation._validation_context
            )
      
        for key, value in simulation.items():        
          if isinstance(value, IModel):
                state_for_boundary = (self.model_to_boundary_state.get(value.uuid))
                clipped[key] = value.clip_box(
                    time_min=self.time_min,
                    time_max=self.time_max,
                    layer_min=self.layer_min,
                    layer_max=self.layer_max,
                    x_min=self.x_min,
                    x_max=self.x_max,
                    y_min=self.y_min,
                    y_max=self.y_max,
                    state_for_boundary=state_for_boundary,
                    ignore_time_purge_empty=self.ignore_time_purge_empty,
                )
          elif isinstance(value, IPackage):
                clipped[key] = value.clip_box(
                    time_min=self.time_min,
                    time_max=self.time_max,
                    layer_min=self.layer_min,
                    layer_max=self.layer_max,
                    x_min=self.x_min,
                    x_max=self.x_max,
                    y_min=self.y_min,
                    y_max=self.y_max,
                )
          elif isinstance(value, list) and all(
                isinstance(item, GWFGWT) for item in value
            ):
                continue
          else:
                raise ValueError(
                    f"object {key} of type {type(value)} cannot be clipped."
                )
        return clipped
    
    def visit_model(self, model: IModel, name: Optional[str] = None) -> IModel:
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
      top, bottom, _ = model._get_domain_geometry()
      clipped = type(model)(**model._options)
      
      for key, pkg in model.items():
            clipped[key] = pkg.clip_box(
                time_min=self.time_min,
                time_max=self.time_max,
                layer_min=self.layer_min,
                layer_max=self.layer_max,
                x_min=self.x_min,
                x_max=self.x_max,
                y_min=self.y_min,
                y_max=self.y_max,
                top=top,
                bottom=bottom,
            )
      
      # Create clipped boundary condition
      if self.model_to_boundary_state :
          state_for_boundary = (self.model_to_boundary_state.get(model.uuid))
      else:
          if self.states_for_boundary is None :
            state_for_boundary = None 
          else:
            if len(self.states_for_boundary) == 1:
              state_for_boundary = next(iter(self.states_for_boundary.values()))
            else   :
                raise ValueError(
                    "states_for_boundary must be provided as a dictionary mapping model names to data arrays when clipping a single model."
                )  

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
    
    
############## PACKAGE METHODS ##############