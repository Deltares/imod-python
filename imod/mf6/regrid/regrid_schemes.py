from pydantic.dataclasses import dataclass

from imod.common.utilities.regrid_method_type import (
    _CONFIG,
    RegridMethodType,
    _RegridVarType,
)
from imod.util.regrid_method_type import (
    RegridderType,
)


@dataclass(config=_CONFIG)
class ConstantHeadRegridMethod(RegridMethodType):
    """
    Object containing regridder methods for the :class:`imod.mf6.ConstantHead`
    package. This can be provided to the ``regrid_like`` method to regrid with
    custom settings.

    Parameters
    ----------
    head: tuple, default (RegridderType.OVERLAP, "mean")
    concentration: tuple, default (RegridderType.OVERLAP, "mean")

    Examples
    --------
    Regrid with custom settings:

    >>> regrid_method = ConstantHeadRegridMethod(head=(RegridderType.BARYCENTRIC,))
    >>> chd.regrid_like(target_grid, RegridderWeightsCache(), regrid_method)

    The RegridderType.OVERLAP and RegridderType.RELATIVEOVERLAP require an extra
    method as string.

    >>> regrid_method = ConstantHeadRegridMethod(head=(RegridderType.OVERLAP, "max",))
    """

    head: _RegridVarType = (
        RegridderType.OVERLAP,
        "mean",
    )  # TODO: should be set to barycentric once supported
    concentration: _RegridVarType = (RegridderType.OVERLAP, "mean")
    ibound: _RegridVarType = (RegridderType.OVERLAP, "mode")


@dataclass(config=_CONFIG)
class DiscretizationRegridMethod(RegridMethodType):
    """
    Object containing regridder methods for the
    :class:`imod.mf6.StructuredDiscretization` and
    :class:`imod.mf6.VerticesDiscretization` packages. This can be provided to
    the ``regrid_like`` method to regrid with custom settings.

    Parameters
    ----------
    top: tuple, default (RegridderType.OVERLAP, "mean")
    bottom: tuple, default (RegridderType.OVERLAP, "mean")
    idomain: tuple, default (RegridderType.OVERLAP, "mode")

    Examples
    --------
    Regrid with custom settings:

    >>> regrid_method = DiscretizationRegridMethod(top=(RegridderType.BARYCENTRIC,))
    >>> dis.regrid_like(target_grid, RegridderWeightsCache(), regrid_method)

    The RegridderType.OVERLAP and RegridderType.RELATIVEOVERLAP require an extra
    method as string.

    >>> regrid_method = DiscretizationRegridMethod(top=(RegridderType.OVERLAP, "max",))
    """

    top: _RegridVarType = (RegridderType.OVERLAP, "mean")
    bottom: _RegridVarType = (RegridderType.OVERLAP, "mean")
    idomain: _RegridVarType = (RegridderType.OVERLAP, "mode")


@dataclass(config=_CONFIG)
class DispersionRegridMethod(RegridMethodType):
    """
    Object containing regridder methods for the :class:`imod.mf6.Dispersion`
    package. This can be provided to the ``regrid_like`` method to regrid with
    custom settings.

    Parameters
    ----------
    diffusion_coefficient: tuple, default (RegridderType.OVERLAP, "mean")
    longitudinal_horizontal: tuple, default (RegridderType.OVERLAP, "mean")
    transversal_horizontal: tuple, default (RegridderType.OVERLAP, "mean")
    longitudinal_vertical: tuple, default (RegridderType.OVERLAP, "mean")
    transversal_horizontal2: tuple, default (RegridderType.OVERLAP, "mean")
    transversal_vertical: tuple, default (RegridderType.OVERLAP, "mean")

    Examples
    --------
    Regrid with custom settings:

    >>> regrid_method = DispersionRegridMethod(longitudinal_horizontal=(RegridderType.BARYCENTRIC,))
    >>> dsp.regrid_like(target_grid, RegridderWeightsCache(), regrid_method)

    The RegridderType.OVERLAP and RegridderType.RELATIVEOVERLAP require an extra
    method as string.

    >>> regrid_method = DispersionRegridMethod(longitudinal_horizontal=(RegridderType.OVERLAP, "max",))
    """

    diffusion_coefficient: _RegridVarType = (RegridderType.OVERLAP, "mean")
    longitudinal_horizontal: _RegridVarType = (RegridderType.OVERLAP, "mean")
    transversal_horizontal1: _RegridVarType = (RegridderType.OVERLAP, "mean")
    longitudinal_vertical: _RegridVarType = (RegridderType.OVERLAP, "mean")
    transversal_horizontal2: _RegridVarType = (RegridderType.OVERLAP, "mean")
    transversal_vertical: _RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass(config=_CONFIG)
class DrainageRegridMethod(RegridMethodType):
    """
    Object containing regridder methods for the :class:`imod.mf6.Drainage`
    package. This can be provided to the ``regrid_like`` method to regrid with
    custom settings.

    Parameters
    ----------
    elevation: tuple, default (RegridderType.OVERLAP, "mean")
    conductance: tuple, default (RegridderType.RELATIVEOVERLAP,"conductance")
    concentration: tuple, default (RegridderType.OVERLAP, "mean")

    Examples
    --------
    Regrid with custom settings:

    >>> regrid_method = DrainageRegridMethod(elevation=(RegridderType.BARYCENTRIC,))
    >>> drn.regrid_like(target_grid, RegridderWeightsCache(), regrid_method)

    The RegridderType.OVERLAP and RegridderType.RELATIVEOVERLAP require an extra
    method as string.

    >>> regrid_method = DrainageRegridMethod(elevation=(RegridderType.OVERLAP, "max",))
    """

    elevation: _RegridVarType = (RegridderType.OVERLAP, "mean")
    conductance: _RegridVarType = (RegridderType.RELATIVEOVERLAP, "conductance")
    concentration: _RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass(config=_CONFIG)
class EvapotranspirationRegridMethod(RegridMethodType):
    """
    Object containing regridder methods for the
    :class:`imod.mf6.Evapotranspiration` package. This can be provided to the
    ``regrid_like`` method to regrid with custom settings.

    Parameters
    ----------
    surface: tuple, default (RegridderType.OVERLAP, "mean")
    rate: tuple, default (RegridderType.OVERLAP, "mean")
    depth: tuple, default (RegridderType.OVERLAP, "mean")
    proportion_rate: tuple, default (RegridderType.OVERLAP, "mean")
    proportion_depth: tuple, default (RegridderType.OVERLAP, "mean")

    Examples
    --------
    Regrid with custom settings:

    >>> regrid_method = EvapotranspirationRegridMethod(surface=(RegridderType.BARYCENTRIC,))
    >>> evt.regrid_like(target_grid, RegridderWeightsCache(), regrid_method)

    The RegridderType.OVERLAP and RegridderType.RELATIVEOVERLAP require an extra
    method as string.

    >>> regrid_method = EvapotranspirationRegridMethod(surface=(RegridderType.OVERLAP, "max",))
    """

    surface: _RegridVarType = (RegridderType.OVERLAP, "mean")
    rate: _RegridVarType = (RegridderType.OVERLAP, "mean")
    depth: _RegridVarType = (RegridderType.OVERLAP, "mean")
    proportion_rate: _RegridVarType = (RegridderType.OVERLAP, "mean")
    proportion_depth: _RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass(config=_CONFIG)
class GeneralHeadBoundaryRegridMethod(RegridMethodType):
    """
    Object containing regridder methods for the
    :class:`imod.mf6.GeneralHeadBoundary` package. This can be provided to the
    ``regrid_like`` method to regrid with custom settings.

    Parameters
    ----------
    head: tuple, default (RegridderType.OVERLAP, "mean")
    conductance: tuple, default (RegridderType.RELATIVEOVERLAP,"conductance")
    concentration: tuple, default (RegridderType.OVERLAP, "mean")

    Examples
    --------
    Regrid with custom settings:

    >>> regrid_method = GeneralHeadBoundaryRegridMethod(head=(RegridderType.BARYCENTRIC,))
    >>> ghb.regrid_like(target_grid, RegridderWeightsCache(), regrid_method)

    The RegridderType.OVERLAP and RegridderType.RELATIVEOVERLAP require an extra
    method as string.

    >>> regrid_method = GeneralHeadBoundaryRegridMethod(head=(RegridderType.OVERLAP, "max",))
    """

    head: _RegridVarType = (
        RegridderType.OVERLAP,
        "mean",
    )  # TODO set to barycentric once supported
    conductance: _RegridVarType = (RegridderType.RELATIVEOVERLAP, "conductance")
    concentration: _RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass(config=_CONFIG)
class InitialConditionsRegridMethod(RegridMethodType):
    """
    Object containing regridder methods for the
    :class:`imod.mf6.InitialConditions` package. This can be provided to the
    ``regrid_like`` method to regrid with custom settings.

    Parameters
    ----------
    start: tuple, default (RegridderType.OVERLAP, "mean")

    Examples
    --------
    Regrid with custom settings:

    >>> regrid_method = InitialConditionsRegridMethod(start=(RegridderType.BARYCENTRIC,))
    >>> ic.regrid_like(target_grid, RegridderWeightsCache(), regrid_method)

    The RegridderType.OVERLAP and RegridderType.RELATIVEOVERLAP require an extra
    method as string.

    >>> regrid_method = InitialConditionsRegridMethod(start=(RegridderType.OVERLAP, "max",))
    """

    start: _RegridVarType = (
        RegridderType.OVERLAP,
        "mean",
    )  # TODO set to barycentric once supported


@dataclass(config=_CONFIG)
class MobileStorageTransferRegridMethod(RegridMethodType):
    """
    Object containing regridder methods for the
    :class:`imod.mf6.MobileStorageTransfer` package. This can be provided to the
    ``regrid_like`` method to regrid with custom settings.

    Parameters
    ----------
    porosity: tuple, default (RegridderType.OVERLAP, "mean")
    decay: tuple, default (RegridderType.OVERLAP, "mean")
    decay_sorbed: tuple, default (RegridderType.OVERLAP, "mean")
    bulk_density: tuple, default (RegridderType.OVERLAP, "mean")
    distcoef: tuple, default (RegridderType.OVERLAP, "mean")
    sp2: tuple, default (RegridderType.OVERLAP, "mean")

    Examples
    --------
    Regrid with custom settings:

    >>> regrid_method = MobileStorageTransferRegridMethod(porosity=(RegridderType.BARYCENTRIC,))
    >>> mst.regrid_like(target_grid, RegridderWeightsCache(), regrid_method)

    The RegridderType.OVERLAP and RegridderType.RELATIVEOVERLAP require an extra
    method as string.

    >>> regrid_method = MobileStorageTransferRegridMethod(porosity=(RegridderType.OVERLAP, "max",))
    """

    porosity: _RegridVarType = (RegridderType.OVERLAP, "mean")
    decay: _RegridVarType = (RegridderType.OVERLAP, "mean")
    decay_sorbed: _RegridVarType = (RegridderType.OVERLAP, "mean")
    bulk_density: _RegridVarType = (RegridderType.OVERLAP, "mean")
    distcoef: _RegridVarType = (RegridderType.OVERLAP, "mean")
    sp2: _RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass(config=_CONFIG)
class NodePropertyFlowRegridMethod(RegridMethodType):
    """
    Object containing regridder methods for the
    :class:`imod.mf6.NodePropertyFlow` package. This can be provided to the
    ``regrid_like`` method to regrid with custom settings.

    Parameters
    ----------
    icelltype: tuple, defaults (RegridderType.OVERLAP, "mean")
    k: tuple, defaults ( RegridderType.OVERLAP,"geometric_mean")
    k22: tuple, defaults (RegridderType.OVERLAP,"geometric_mean")
    k33: tuple, defaults (RegridderType.OVERLAP,"harmonic_mean")
    angle1: tuple, defaults (RegridderType.OVERLAP, "mean")
    angle2: tuple, defaults (RegridderType.OVERLAP, "mean")
    angle3: tuple, defaults (RegridderType.OVERLAP, "mean")
    rewet_layer: tuple, defaults (RegridderType.OVERLAP, "mean")

    Examples
    --------
    Regrid with custom settings:

    >>> regrid_method = NodePropertyFlowRegridMethod(k=(RegridderType.BARYCENTRIC,))
    >>> npf.regrid_like(target_grid, RegridderWeightsCache(), regrid_method)

    The RegridderType.OVERLAP and RegridderType.RELATIVEOVERLAP require an extra
    method as string.

    >>> regrid_method = NodePropertyFlowRegridMethod(k=(RegridderType.OVERLAP, "max",))
    """

    icelltype: _RegridVarType = (RegridderType.OVERLAP, "mode")
    k: _RegridVarType = (
        RegridderType.OVERLAP,
        "geometric_mean",
    )  # horizontal if angle2 = 0
    k22: _RegridVarType = (
        RegridderType.OVERLAP,
        "geometric_mean",
    )  # horizontal if angle2 = 0 & angle3 = 0
    k33: _RegridVarType = (
        RegridderType.OVERLAP,
        "harmonic_mean",
    )  # vertical if angle2 = 0 & angle3 = 0
    angle1: _RegridVarType = (RegridderType.OVERLAP, "mean")
    angle2: _RegridVarType = (RegridderType.OVERLAP, "mean")
    angle3: _RegridVarType = (RegridderType.OVERLAP, "mean")
    rewet_layer: _RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass(config=_CONFIG)
class RechargeRegridMethod(RegridMethodType):
    """
    Object containing regridder methods for the :class:`imod.mf6.Recharge`
    package. This can be provided to the ``regrid_like`` method to regrid with
    custom settings.

    Parameters
    ----------
    rate: tuple, defaults (RegridderType.OVERLAP, "mean")
    concentration: tuple, defaults (RegridderType.OVERLAP, "mean")

    Examples
    --------
    Regrid with custom settings:

    >>> regrid_method = RechargeRegridMethod(rate=(RegridderType.BARYCENTRIC,))
    >>> rch.regrid_like(target_grid, RegridderWeightsCache(), regrid_method)

    The RegridderType.OVERLAP and RegridderType.RELATIVEOVERLAP require an extra
    method as string.

    >>> regrid_method = RechargeRegridMethod(rate=(RegridderType.OVERLAP, "max",))
    """

    rate: _RegridVarType = (RegridderType.OVERLAP, "mean")
    concentration: _RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass(config=_CONFIG)
class RiverRegridMethod(RegridMethodType):
    """
    Object containing regridder methods for the :class:`imod.mf6.River` package.
    This can be provided to the ``regrid_like`` method to regrid with custom
    settings.

    Parameters
    ----------
    stage: tuple, default (RegridderType.OVERLAP, "mean")
    conductance: tuple, default (RegridderType.RELATIVEOVERLAP, "conductance")
    bottom_elevation: tuple, default (RegridderType.OVERLAP, "mean")
    concentration: tuple, default (RegridderType.OVERLAP, "mean")

    Examples
    --------
    Regrid with custom settings:

    >>> regrid_method = RiverRegridMethod(stage=(RegridderType.BARYCENTRIC,))
    >>> riv.regrid_like(target_grid, RegridderWeightsCache(), regrid_method)

    The RegridderType.OVERLAP and RegridderType.RELATIVEOVERLAP require an extra
    method as string.

    >>> regrid_method = RiverRegridMethod(stage=(RegridderType.OVERLAP, "max",))
    """

    stage: _RegridVarType = (RegridderType.OVERLAP, "mean")
    conductance: _RegridVarType = (RegridderType.RELATIVEOVERLAP, "conductance")
    bottom_elevation: _RegridVarType = (RegridderType.OVERLAP, "mean")
    concentration: _RegridVarType = (RegridderType.OVERLAP, "mean")
    infiltration_factor: _RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass(config=_CONFIG)
class SpecificStorageRegridMethod(RegridMethodType):
    """
    Object containing regridder methods for the
    :class:`imod.mf6.SpecificStorage` package. This can be provided to the
    ``regrid_like`` method to regrid with custom settings.

    Parameters
    ----------
    convertible: tuple, default (RegridderType.OVERLAP, "mode")
    specific_storage: tuple, default (RegridderType.OVERLAP, "mean")
    specific_yield: tuple, default (RegridderType.OVERLAP, "mean")

    Examples
    --------
    Regrid with custom settings:

    >>> regrid_method = SpecificStorageRegridMethod(specific_storage=(RegridderType.BARYCENTRIC,))
    >>> sto.regrid_like(target_grid, RegridderWeightsCache(), regrid_method)

    The RegridderType.OVERLAP and RegridderType.RELATIVEOVERLAP require an extra
    method as string.

    >>> regrid_method = SpecificStorageRegridMethod(specific_storage=(RegridderType.OVERLAP, "max",))
    """

    convertible: _RegridVarType = (RegridderType.OVERLAP, "mode")
    specific_storage: _RegridVarType = (RegridderType.OVERLAP, "mean")
    specific_yield: _RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass(config=_CONFIG)
class StorageCoefficientRegridMethod(RegridMethodType):
    """
    Object containing regridder methods for the
    :class:`imod.mf6.StorageCoefficient` package. This can be provided to the
    ``regrid_like`` method to regrid with custom settings.

    Parameters
    ----------
    convertible: tuple, default (RegridderType.OVERLAP, "mode")
    storage_coefficient: tuple, default (RegridderType.OVERLAP, "mean")
    specific_yield: tuple, default (RegridderType.OVERLAP, "mean")

    Examples
    --------
    Regrid with custom settings:

    >>> regrid_method = StorageCoefficientRegridMethod(storage_coefficient=(RegridderType.BARYCENTRIC,))
    >>> sto.regrid_like(target_grid, RegridderWeightsCache(), regrid_method)

    The RegridderType.OVERLAP and RegridderType.RELATIVEOVERLAP require an extra
    method as string.

    >>> regrid_method = StorageCoefficientRegridMethod(storage_coefficient=(RegridderType.OVERLAP, "max",))
    """

    convertible: _RegridVarType = (RegridderType.OVERLAP, "mode")
    storage_coefficient: _RegridVarType = (RegridderType.OVERLAP, "mean")
    specific_yield: _RegridVarType = (RegridderType.OVERLAP, "mean")
