import abc

from imod.common.interfaces.ipackage import IPackage


class IMaskingSettings(IPackage, abc.ABC):
    """
    Interface for packages that support masking
    """

    @property
    @abc.abstractmethod
    def skip_variables(self) -> list[str]:
        raise NotImplementedError
