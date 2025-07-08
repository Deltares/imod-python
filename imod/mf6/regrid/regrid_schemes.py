from pydantic.dataclasses import dataclass

from imod.common.utilities.dataclass_type import (
    _CONFIG,
    DataclassType,
)
from imod.util.regrid import RegridderType, RegridVarType


@dataclass(config=_CONFIG)
class ConstantHeadRegridMethod(DataclassType):
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

    head: RegridVarType = (
        RegridderType.OVERLAP,
        "mean",
    )  # TODO: should be set to barycentric once supported
    concentration: RegridVarType = (RegridderType.OVERLAP, "mean")
    ibound: RegridVarType = (RegridderType.OVERLAP, "mode")


@dataclass(config=_CONFIG)
class DiscretizationRegridMethod(DataclassType):
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

    top: RegridVarType = (RegridderType.OVERLAP, "mean")
    bottom: RegridVarType = (RegridderType.OVERLAP, "mean")
    idomain: RegridVarType = (RegridderType.OVERLAP, "mode")


@dataclass(config=_CONFIG)
class DispersionRegridMethod(DataclassType):
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

    diffusion_coefficient: RegridVarType = (RegridderType.OVERLAP, "mean")
    longitudinal_horizontal: RegridVarType = (RegridderType.OVERLAP, "mean")
    transversal_horizontal1: RegridVarType = (RegridderType.OVERLAP, "mean")
    longitudinal_vertical: RegridVarType = (RegridderType.OVERLAP, "mean")
    transversal_horizontal2: RegridVarType = (RegridderType.OVERLAP, "mean")
    transversal_vertical: RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass(config=_CONFIG)
class DrainageRegridMethod(DataclassType):
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

    elevation: RegridVarType = (RegridderType.OVERLAP, "mean")
    conductance: RegridVarType = (RegridderType.RELATIVEOVERLAP, "conductance")
    concentration: RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass(config=_CONFIG)
class EvapotranspirationRegridMethod(DataclassType):
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

    surface: RegridVarType = (RegridderType.OVERLAP, "mean")
    rate: RegridVarType = (RegridderType.OVERLAP, "mean")
    depth: RegridVarType = (RegridderType.OVERLAP, "mean")
    proportion_rate: RegridVarType = (RegridderType.OVERLAP, "mean")
    proportion_depth: RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass(config=_CONFIG)
class GeneralHeadBoundaryRegridMethod(DataclassType):
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

    head: RegridVarType = (
        RegridderType.OVERLAP,
        "mean",
    )  # TODO set to barycentric once supported
    conductance: RegridVarType = (RegridderType.RELATIVEOVERLAP, "conductance")
    concentration: RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass(config=_CONFIG)
class InitialConditionsRegridMethod(DataclassType):
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

    start: RegridVarType = (
        RegridderType.OVERLAP,
        "mean",
    )  # TODO set to barycentric once supported


@dataclass(config=_CONFIG)
class MobileStorageTransferRegridMethod(DataclassType):
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

    porosity: RegridVarType = (RegridderType.OVERLAP, "mean")
    decay: RegridVarType = (RegridderType.OVERLAP, "mean")
    decay_sorbed: RegridVarType = (RegridderType.OVERLAP, "mean")
    bulk_density: RegridVarType = (RegridderType.OVERLAP, "mean")
    distcoef: RegridVarType = (RegridderType.OVERLAP, "mean")
    sp2: RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass(config=_CONFIG)
class NodePropertyFlowRegridMethod(DataclassType):
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

    icelltype: RegridVarType = (RegridderType.OVERLAP, "mode")
    k: RegridVarType = (
        RegridderType.OVERLAP,
        "geometric_mean",
    )  # horizontal if angle2 = 0
    k22: RegridVarType = (
        RegridderType.OVERLAP,
        "geometric_mean",
    )  # horizontal if angle2 = 0 & angle3 = 0
    k33: RegridVarType = (
        RegridderType.OVERLAP,
        "harmonic_mean",
    )  # vertical if angle2 = 0 & angle3 = 0
    angle1: RegridVarType = (RegridderType.OVERLAP, "mean")
    angle2: RegridVarType = (RegridderType.OVERLAP, "mean")
    angle3: RegridVarType = (RegridderType.OVERLAP, "mean")
    rewet_layer: RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass(config=_CONFIG)
class RechargeRegridMethod(DataclassType):
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

    rate: RegridVarType = (RegridderType.OVERLAP, "mean")
    concentration: RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass(config=_CONFIG)
class RiverRegridMethod(DataclassType):
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

    stage: RegridVarType = (RegridderType.OVERLAP, "mean")
    conductance: RegridVarType = (RegridderType.RELATIVEOVERLAP, "conductance")
    bottom_elevation: RegridVarType = (RegridderType.OVERLAP, "mean")
    concentration: RegridVarType = (RegridderType.OVERLAP, "mean")
    infiltration_factor: RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass(config=_CONFIG)
class SpecificStorageRegridMethod(DataclassType):
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

    convertible: RegridVarType = (RegridderType.OVERLAP, "mode")
    specific_storage: RegridVarType = (RegridderType.OVERLAP, "mean")
    specific_yield: RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass(config=_CONFIG)
class StorageCoefficientRegridMethod(DataclassType):
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

    convertible: RegridVarType = (RegridderType.OVERLAP, "mode")
    storage_coefficient: RegridVarType = (RegridderType.OVERLAP, "mean")
    specific_yield: RegridVarType = (RegridderType.OVERLAP, "mean")
