import abc

from imod.mf6.interfaces.ipackagebase import IPackageBase


class IPackage(IPackageBase, metaclass=abc.ABCMeta):
    """
    The base methods and attributes available in all packages
    """

    @property
    @abc.abstractmethod
    def auxiliary_data_fields(self) -> dict[str, str]:
        raise NotImplementedError