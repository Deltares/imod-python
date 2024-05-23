from dataclasses import dataclass
from typing import Tuple

from imod.mf6.utilities.regrid import RegridderType


@dataclass
class ConstantHeadRegridMethod:
    head: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")  # TODO: should be set to barycentric once supported
    concentration: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")


@dataclass
class DiscretizationRegridMethod:
    top: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    bottom: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    idomain: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mode")


@dataclass
class DrainageRegridMethod:
    elevation: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    conductance: Tuple[RegridderType, str] = (RegridderType.RELATIVEOVERLAP, "conductance")
    concentration: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")

@dataclass
class EmptyRegridderMethods:
    pass

@dataclass
class GeneralHeadBoundaryRegridMethod:
    head: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")  # TODO set to barycentric once supported
    conductance: Tuple[RegridderType, str] = (RegridderType.RELATIVEOVERLAP, "conductance")
    concentration: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")


@dataclass
class HorizontalFlowBarrierRegridderMethods:
    pass


@dataclass
class InitialConditionsRegridMethod:
    start: Tuple[RegridderType, str] = (
            RegridderType.OVERLAP,
            "mean",
        )  # TODO set to barycentric once supported


@dataclass
class NodePropertyFlowRegridMethod:
    icelltype: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    k: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "geometric_mean")  # horizontal if angle2 = 0
    k22: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "geometric_mean")  # horizontal if angle2 = 0 & angle3 = 0
    k33: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "harmonic_mean")  # vertical if angle2 = 0 & angle3 = 0
    angle1: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    angle2: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    angle3: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    rewet_layer: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")


@dataclass
class OutputControlRegridderMethod:
    pass


@dataclass
class RechargeRegridMethod:
    rate: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    concentration: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")


@dataclass
class RiverRegridMethod:
    stage: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    conductance: Tuple[RegridderType, str] = (RegridderType.RELATIVEOVERLAP, "conductance")
    bottom_elevation: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    concentration: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")


@dataclass
class SpecificStorageRegridMethod:
    convertible: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mode")
    specific_storage: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    specific_yield: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")


@dataclass
class StorageCoefficientRegridMethod:
    convertible: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mode")
    storage_coefficient: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")
    specific_yield: Tuple[RegridderType, str] = (RegridderType.OVERLAP, "mean")