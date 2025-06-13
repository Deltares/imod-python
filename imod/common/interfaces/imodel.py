from abc import abstractmethod
from typing import Optional, Tuple

from imod.common.interfaces.idict import IDict
from imod.common.statusinfo import StatusInfoBase
from imod.typing import GridDataArray


class IModel(IDict):
    """
    Interface for imod.mf6.model.Modflow6Model
    """

    @abstractmethod
    def mask_all_packages(self, mask: GridDataArray):
        raise NotImplementedError

    @abstractmethod
    def purge_empty_packages(
        self, model_name: Optional[str] = "", ignore_time: bool = False
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def validate(self, model_name: str = "") -> StatusInfoBase:
        raise NotImplementedError

    @property
    @abstractmethod
    def domain(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def options(self) -> dict:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def is_regridding_supported(self) -> Tuple[bool, str]:
        raise NotImplementedError

    @abstractmethod
    def is_splitting_supported(self) -> Tuple[bool, str]:
        raise NotImplementedError
