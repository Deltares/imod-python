from dataclasses import dataclass
from typing import ClassVar, Protocol, Tuple

from imod.mf6.utilities.regrid import RegridderType


class RegridMethodType(Protocol):
    # Work around that type annotation is a bit hard on dataclasses, as they
    # don't expose a class interface. 
    # Adapted from: https://stackoverflow.com/a/55240861 
    # "As already noted in comments, checking for this attribute is currently the
    # most reliable way to ascertain that something is a dataclass"
    # See also:
    # https://github.com/python/mypy/issues/6568#issuecomment-1324196557

    __dataclass_fields__: ClassVar[dict] 

@dataclass
class ConstantHeadRegridMethod(RegridMethodType):
    head: Tuple[RegridderType, str] = (
        RegridderType.OVERLAP,
        "mean",
    )  # TODO: should be set to barycentric once supported
    concentration: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")


@dataclass
class DiscretizationRegridMethod(RegridMethodType):
    top: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    bottom: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    idomain: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mode")


@dataclass
class DispersionRegridMethod(RegridMethodType):
    diffusion_coefficient: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    longitudinal_horizontal: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    transversal_horizontal1: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    longitudinal_vertical: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    transversal_horizontal2: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    transversal_vertical: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")


@dataclass
class DrainageRegridMethod(RegridMethodType):
    elevation: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    conductance: Tuple[RegridderType, str] = (
        RegridderType.RELATIVEOVERLAP,
        "conductance",
    )
    concentration: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")


@dataclass
class EmptyRegridderMethod(RegridMethodType):
    pass


@dataclass
class EvapotranspirationRegridMethod(RegridMethodType):
    surface: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    rate: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    depth: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    proportion_rate: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    proportion_depth: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")


@dataclass
class GeneralHeadBoundaryRegridMethod(RegridMethodType):
    head: Tuple[RegridderType, str] = (
        RegridderType.OVERLAP,
        "mean",
    )  # TODO set to barycentric once supported
    conductance: Tuple[RegridderType, str] = (
        RegridderType.RELATIVEOVERLAP,
        "conductance",
    )
    concentration: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")


@dataclass
class InitialConditionsRegridMethod(RegridMethodType):
    start: Tuple[RegridderType, str] = (
        RegridderType.OVERLAP,
        "mean",
    )  # TODO set to barycentric once supported


@dataclass
class MobileStorageTransferRegridMethod(RegridMethodType):
    porosity: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    decay: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    decay_sorbed: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    bulk_density: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    distcoef: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    sp2: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")


@dataclass
class NodePropertyFlowRegridMethod(RegridMethodType):
    icelltype: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    k: Tuple[RegridderType, str] = (
        RegridderType.OVERLAP,
        "geometric_mean",
    )  # horizontal if angle2 = 0
    k22: Tuple[RegridderType, str] = (
        RegridderType.OVERLAP,
        "geometric_mean",
    )  # horizontal if angle2 = 0 & angle3 = 0
    k33: Tuple[RegridderType, str] = (
        RegridderType.OVERLAP,
        "harmonic_mean",
    )  # vertical if angle2 = 0 & angle3 = 0
    angle1: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    angle2: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    angle3: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    rewet_layer: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")


@dataclass
class RechargeRegridMethod(RegridMethodType):
    rate: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    concentration: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")


@dataclass
class RiverRegridMethod(RegridMethodType):
    stage: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    conductance: Tuple[RegridderType, str] = (
        RegridderType.RELATIVEOVERLAP,
        "conductance",
    )
    bottom_elevation: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    concentration: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")


@dataclass
class SpecificStorageRegridMethod(RegridMethodType):
    convertible: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mode")
    specific_storage: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    specific_yield: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")


@dataclass
class StorageCoefficientRegridMethod(RegridMethodType):
    convertible: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mode")
    storage_coefficient: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    specific_yield: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
