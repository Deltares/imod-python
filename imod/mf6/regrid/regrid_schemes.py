from typing import ClassVar, Protocol, Tuple, TypeAlias

from pydantic.dataclasses import dataclass

from imod.mf6.utilities.regrid import RegridderType

_RegridVarType: TypeAlias = Tuple[RegridderType, str] | Tuple[RegridderType]


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
    head: _RegridVarType = (
        RegridderType.OVERLAP,
        "mean",
    )  # TODO: should be set to barycentric once supported
    concentration: _RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass
class DiscretizationRegridMethod(RegridMethodType):
    top: _RegridVarType = (RegridderType.OVERLAP, "mean")
    bottom: _RegridVarType = (RegridderType.OVERLAP, "mean")
    idomain: _RegridVarType = (RegridderType.OVERLAP, "mode")


@dataclass
class DispersionRegridMethod(RegridMethodType):
    diffusion_coefficient: _RegridVarType = (RegridderType.OVERLAP, "mean")
    longitudinal_horizontal: _RegridVarType = (RegridderType.OVERLAP, "mean")
    transversal_horizontal1: _RegridVarType = (RegridderType.OVERLAP, "mean")
    longitudinal_vertical: _RegridVarType = (RegridderType.OVERLAP, "mean")
    transversal_horizontal2: _RegridVarType = (RegridderType.OVERLAP, "mean")
    transversal_vertical: _RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass
class DrainageRegridMethod(RegridMethodType):
    elevation: _RegridVarType = (RegridderType.OVERLAP, "mean")
    conductance: _RegridVarType = (
        RegridderType.RELATIVEOVERLAP,
        "conductance",
    )
    concentration: _RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass
class EmptyRegridderMethod(RegridMethodType):
    pass


@dataclass
class EvapotranspirationRegridMethod(RegridMethodType):
    surface: _RegridVarType = (RegridderType.OVERLAP, "mean")
    rate: _RegridVarType = (RegridderType.OVERLAP, "mean")
    depth: _RegridVarType = (RegridderType.OVERLAP, "mean")
    proportion_rate: _RegridVarType = (RegridderType.OVERLAP, "mean")
    proportion_depth: _RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass
class GeneralHeadBoundaryRegridMethod(RegridMethodType):
    head: _RegridVarType = (
        RegridderType.OVERLAP,
        "mean",
    )  # TODO set to barycentric once supported
    conductance: _RegridVarType = (
        RegridderType.RELATIVEOVERLAP,
        "conductance",
    )
    concentration: _RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass
class InitialConditionsRegridMethod(RegridMethodType):
    start: _RegridVarType = (
        RegridderType.OVERLAP,
        "mean",
    )  # TODO set to barycentric once supported


@dataclass
class MobileStorageTransferRegridMethod(RegridMethodType):
    porosity: _RegridVarType = (RegridderType.OVERLAP, "mean")
    decay: _RegridVarType = (RegridderType.OVERLAP, "mean")
    decay_sorbed: _RegridVarType = (RegridderType.OVERLAP, "mean")
    bulk_density: _RegridVarType = (RegridderType.OVERLAP, "mean")
    distcoef: _RegridVarType = (RegridderType.OVERLAP, "mean")
    sp2: _RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass
class NodePropertyFlowRegridMethod(RegridMethodType):
    icelltype: _RegridVarType = (RegridderType.OVERLAP, "mean")
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


@dataclass
class RechargeRegridMethod(RegridMethodType):
    rate: _RegridVarType = (RegridderType.OVERLAP, "mean")
    concentration: _RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass
class RiverRegridMethod(RegridMethodType):
    stage: _RegridVarType = (RegridderType.OVERLAP, "mean")
    conductance: _RegridVarType = (
        RegridderType.RELATIVEOVERLAP,
        "conductance",
    )
    bottom_elevation: _RegridVarType = (RegridderType.OVERLAP, "mean")
    concentration: _RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass
class SpecificStorageRegridMethod(RegridMethodType):
    convertible: _RegridVarType = (RegridderType.OVERLAP, "mode")
    specific_storage: _RegridVarType = (RegridderType.OVERLAP, "mean")
    specific_yield: _RegridVarType = (RegridderType.OVERLAP, "mean")


@dataclass
class StorageCoefficientRegridMethod(RegridMethodType):
    convertible: _RegridVarType = (RegridderType.OVERLAP, "mode")
    storage_coefficient: _RegridVarType = (RegridderType.OVERLAP, "mean")
    specific_yield: _RegridVarType = (RegridderType.OVERLAP, "mean")
