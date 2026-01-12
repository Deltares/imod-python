from abc import abstractmethod

from imod.common.interfaces.imodel import IModel
from imod.common.interfaces.ipackage import IPackage
from imod.common.interfaces.isimulation import ISimulation


class IVisitor:
    @abstractmethod
    def visit_simulation(self, simulation: ISimulation) -> ISimulation:
        pass

    @abstractmethod
    def visit_model(self, model: IModel) -> IModel:
        pass
    
    @abstractmethod
    def visit_package(self, package: IPackage) -> IPackage:
        pass


class IVisitee:
    @abstractmethod
    def accept(self, visitor: IVisitor):
        pass
