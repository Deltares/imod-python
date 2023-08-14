from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass
class WriteContext:
    simulation_directory: Path = "."
    binary: bool = False
    validate: bool = True
    absolute_paths: bool = False

    @property
    def output_directory(self) -> Path:
        return self.__output_directory

    @output_directory.setter
    def output_directory(self, model_directory: Union[Path, str]) -> None:
        self.__output_directory = Path(model_directory)
