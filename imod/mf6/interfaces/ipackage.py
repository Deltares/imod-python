import abc

from imod.mf6.interfaces.ipackagebase import IPackageBase


class IPackage(IPackageBase, metaclass=abc.ABCMeta):
    """
    The base methods and attributes available in all packages
    """

    pass