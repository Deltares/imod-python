from pydantic.dataclasses import dataclass

from imod.util.regrid_method_type import (
    _CONFIG,
    RegridderType,
    RegridMethodType,
    _RegridVarType,
)


@dataclass(config=_CONFIG)
class SprinklingRegridMethod(RegridMethodType):
    """
    Object containing regridder methods for the
    :class:`imod.msw.Sprinkling` package. This can be provided to the
    ``regrid_like`` method to regrid with custom settings.

    Parameters
    ----------
    max_abstraction_groundwater: tuple, default (RegridderType.OVERLAP, "mean")
    max_abstraction_surfacewater: tuple, default (RegridderType.OVERLAP, "mean")

    Examples
    --------
    Regrid with custom settings:

    >>> regrid_method = SprinklingRegridMethod(max_abstraction_groundwater=(RegridderType.BARYCENTRIC,))
    >>> sprinking.regrid_like(target_grid, RegridderWeightsCache(), regrid_method)

    The RegridderType.OVERLAP and RegridderType.RELATIVEOVERLAP require an extra
    method as string.

    >>> regrid_method = SprinklingRegridMethod(max_abstraction_groundwater=(RegridderType.OVERLAP, "max",))
    """

    max_abstraction_groundwater: _RegridVarType = (RegridderType.OVERLAP, "mean")
    max_abstraction_surfacewater: _RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass(config=_CONFIG)
class MeteoGridRegridMethod(RegridMethodType):
    """
    Object containing regridder methods for the
    :class:`imod.msw.MeteoGrid` package. This can be provided to the
    ``regrid_like`` method to regrid with custom settings.

    Parameters
    ----------
    precipitation: tuple, default (RegridderType.OVERLAP, "mean")
    evapotranspiration: tuple, default (RegridderType.OVERLAP, "mean")

    Examples
    --------
    Regrid with custom settings:

    >>> regrid_method = MeteoGridRegridMethod(precipitation=(RegridderType.BARYCENTRIC,))
    >>> meteo.regrid_like(target_grid, RegridderWeightsCache(), regrid_method)

    The RegridderType.OVERLAP and RegridderType.RELATIVEOVERLAP require an extra
    method as string.

    >>> regrid_method = MeteoGridRegridMethod(precipitation=(RegridderType.OVERLAP, "max",))
    """

    precipitation: _RegridVarType = (RegridderType.OVERLAP, "mean")
    evapotranspiration: _RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass(config=_CONFIG)
class GridDataRegridMethod(RegridMethodType):
    """
    Object containing regridder methods for the
    :class:`imod.msw.Grid_data` package. This can be provided to the
    ``regrid_like`` method to regrid with custom settings.

    Parameters
    ----------
    area: tuple, default (RegridderType.RELATIVEOVERLAP, "conductance")
    landuse: tuple, default (RegridderType.OVERLAP, "mean")
    rootzone_depth:  tuple, default (RegridderType.OVERLAP, "mean")
    surface_elevation: tuple, default  (RegridderType.OVERLAP, "mean")
    soil_physical_unit: tuple, default  (RegridderType.OVERLAP, "mean")
    active: tuple, default (RegridderType.OVERLAP, "mean")

    Examples
    --------
    Regrid with custom settings:

    >>> regrid_method = GridDataRegridMethod(area=(RegridderType.BARYCENTRIC,))
    >>> grid_data.regrid_like(target_grid, RegridderWeightsCache(), regrid_method)

    The RegridderType.OVERLAP and RegridderType.RELATIVEOVERLAP require an extra
    method as string.

    >>> regrid_method = GridDataRegridMethod(area=(RegridderType.OVERLAP, "max",))
    """

    area: _RegridVarType = (RegridderType.RELATIVEOVERLAP, "conductance")
    landuse: _RegridVarType = (RegridderType.OVERLAP, "mean")
    rootzone_depth: _RegridVarType = (RegridderType.OVERLAP, "mean")
    surface_elevation: _RegridVarType = (RegridderType.OVERLAP, "mean")
    soil_physical_unit: _RegridVarType = (RegridderType.OVERLAP, "mean")
    active: _RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass(config=_CONFIG)
class PondingRegridMethod(RegridMethodType):
    """
    Object containing regridder methods for the
    :class:`imod.msw.ponding` package. This can be provided to the
    ``regrid_like`` method to regrid with custom settings.

    Parameters
    ----------
    ponding_depth: tuple, default(RegridderType.OVERLAP, "mean")
    runon_resistance: tuple, default(RegridderType.OVERLAP, "mean")
    runoff_resistance:tuple, default ( RegridderType.OVERLAP,  "mean")

    Examples
    --------
    Regrid with custom settings:

    >>> regrid_method = PondingRegridMethod(runoff_resistance=(RegridderType.BARYCENTRIC,))
    >>> ponding.regrid_like(target_grid, RegridderWeightsCache(), regrid_method)

    The RegridderType.OVERLAP and RegridderType.RELATIVEOVERLAP require an extra
    method as string.

    >>> regrid_method = PondingRegridMethod(runoff_resistance=(RegridderType.OVERLAP, "max",))
    """

    ponding_depth: _RegridVarType = (RegridderType.OVERLAP, "mean")
    runon_resistance: _RegridVarType = (RegridderType.OVERLAP, "mean")
    runoff_resistance: _RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass(config=_CONFIG)
class InfiltrationRegridMethod(RegridMethodType):
    """
    Object containing regridder methods for the
    :class:`imod.msw.infiltration` package. This can be provided to the
    ``regrid_like`` method to regrid with custom settings.

    Parameters
    ----------
    infiltration_capacity: tuple, default (RegridderType.OVERLAP, "mean")
    downward_resistance:  tuple, default(RegridderType.OVERLAP, "mean")
    upward_resistance:  tuple, default ( RegridderType.OVERLAP, "mean"  )
    longitudinal_vertical: tuple, default  (RegridderType.OVERLAP, "mean")
    bottom_resistance:  tuple, default(RegridderType.OVERLAP, "mean")
    extra_storage_coefficient: tuple, default (RegridderType.OVERLAP, "mean")

    Examples
    --------
    Regrid with custom settings:

    >>> regrid_method = InfiltrationRegridMethod(bottom_resistance=(RegridderType.BARYCENTRIC,))
    >>> infiltration.regrid_like(target_grid, RegridderWeightsCache(), regrid_method)

    The RegridderType.OVERLAP and RegridderType.RELATIVEOVERLAP require an extra
    method as string.

    >>> regrid_method = PondingRegridMethod(bottom_resistance=(RegridderType.OVERLAP, "max",))
    """

    infiltration_capacity: _RegridVarType = (RegridderType.OVERLAP, "mean")
    downward_resistance: _RegridVarType = (RegridderType.OVERLAP, "mean")
    upward_resistance: _RegridVarType = (RegridderType.OVERLAP, "mean")
    longitudinal_vertical: _RegridVarType = (RegridderType.OVERLAP, "mean")
    bottom_resistance: _RegridVarType = (RegridderType.OVERLAP, "mean")
    extra_storage_coefficient: _RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass(config=_CONFIG)
class ScalingRegridMethod(RegridMethodType):
    """
    Object containing regridder methods for the
    :class:`imod.msw.scaling_factors` package. This can be provided to the
    ``regrid_like`` method to regrid with custom settings.

    Parameters
    ----------
    scale_soil_moisture: tuple, default (RegridderType.OVERLAP, "mean")
    scale_hydraulic_conductivity:  tuple, default(RegridderType.OVERLAP, "mean")
    scale_pressure_head:  tuple, default ( RegridderType.OVERLAP, "mean"  )
    depth_perched_water_table: tuple, default  (RegridderType.OVERLAP, "mean")

    Examples
    --------
    Regrid with custom settings:

    >>> regrid_method = ScalingRegridMethod(scale_soil_moisture=(RegridderType.BARYCENTRIC,))
    >>> scaling_factors.regrid_like(target_grid, RegridderWeightsCache(), regrid_method)

    The RegridderType.OVERLAP and RegridderType.RELATIVEOVERLAP require an extra
    method as string.

    >>> regrid_method = ScalingRegridMethod(scale_soil_moisture=(RegridderType.OVERLAP, "max",))
    """

    scale_soil_moisture: _RegridVarType = (RegridderType.OVERLAP, "mean")
    scale_hydraulic_conductivity: _RegridVarType = (RegridderType.OVERLAP, "mean")
    scale_pressure_head: _RegridVarType = (RegridderType.OVERLAP, "mean")
    depth_perched_water_table: _RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass(config=_CONFIG)
class IdfMappingRegridMethod(RegridMethodType):
    """
    Object containing regridder methods for the
    :class:`imod.msw.IdfMapping` package. This can be provided to the
    ``regrid_like`` method to regrid with custom settings.
    """

    area: _RegridVarType = (RegridderType.RELATIVEOVERLAP, "conductance")
