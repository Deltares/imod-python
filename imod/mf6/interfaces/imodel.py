from typing import Optional, Tuple

from imod.mf6.interfaces.idict import IDict
from imod.mf6.statusinfo import StatusInfoBase
from imod.typing import GridDataArray


class IModel(IDict):
    """
    Interface for imod.mf6.model.Modflow6Model
    """

    def mask_all_packages(self, mask: GridDataArray):
        raise NotImplementedError

    def purge_empty_packages(self, model_name: Optional[str] = "") -> None:
        raise NotImplementedError

    def validate(self, model_name: str = "") -> StatusInfoBase:
        raise NotImplementedError

    @property
    def domain(self):
        raise NotImplementedError

    @property
    def model_id(self) -> str:
        raise NotImplementedError

    def is_regridding_supported(self) -> Tuple[bool, str]:
        raise NotImplementedError
