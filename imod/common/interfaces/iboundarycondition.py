from abc import ABC
from typing import ClassVar

from imod.common.interfaces.ipackage import IPackage


class IBoundaryCondition(IPackage, ABC):
    """
    Interface for boundary condition packages with stress-period variables.
    """

    _period_data: ClassVar[tuple[str, ...]]
