from imod.typing import GridDataArray
from typing import Optional
from imod.mf6.statusinfo import  StatusInfoBase
import collections

class IModel(collections.UserDict):   

    def mask_all_packages(  self, mask: GridDataArray    ):
        raise NotImplementedError

    
    def purge_empty_packages(self, model_name: Optional[str] = "") -> None:
        raise NotImplementedError
    
    def _validate(self, model_name: str = "") -> StatusInfoBase:
         raise NotImplementedError
    
    @property
    def domain(self):
         raise NotImplementedError        