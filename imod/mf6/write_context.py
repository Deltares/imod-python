from dataclasses import dataclass
from os.path import relpath
from pathlib import Path
from typing import Union


@dataclass
class WriteContext:
    simulation_directory: Path = "."
    binary: bool = False
    absolute_paths: bool = False

    @property
    def output_directory(self) -> Path:
        return self.__output_directory

    @output_directory.setter
    def output_directory(self, model_directory: Union[Path, str]) -> None:
        self.__output_directory = Path(model_directory)

    def get_adjusted_output_directory(self) -> Path:
        if not self.absolute_paths:
            return Path(relpath(self.output_directory, self.simulation_directory))
        return self.output_directory
