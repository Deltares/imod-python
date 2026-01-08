from abc import abstractmethod

from typing import Optional

from imod.common.interfaces.imodel import IModel
from imod.common.interfaces.isimulation import ISimulation

class IVisitor:
    @abstractmethod
    def visit_simulation(self, simulation: ISimulation, name: str) -> ISimulation:
        pass
      
    @abstractmethod
    def visit_model(self, model: IModel, name: Optional[str]) -> IModel:
        pass
      
class IVisitee:
    @abstractmethod
    def accept(self, visitor: IVisitor, name: str):
        pass
      
      
