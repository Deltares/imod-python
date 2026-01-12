from abc import abstractmethod
from typing import Optional, Self

class INode:
  
    @property
    @abstractmethod
    def parent(self) -> Optional[Self]:
        raise NotImplementedError
      
    @parent.setter
    @abstractmethod
    def parent(self, value: Optional[Self]) -> None:
        raise NotImplementedError