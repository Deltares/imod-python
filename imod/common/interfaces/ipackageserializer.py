import abc
from pathlib import Path
from typing import Any

from imod.common.interfaces.ipackagebase import IPackageBase


class IPackageSerializer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def to_file(
        self, pkg: IPackageBase, directory: Path, file_name: str, **kwargs: Any
    ) -> Path:
        raise NotImplementedError
