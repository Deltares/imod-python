from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass
class WriteContext:
    sim_directory: Path = "."
    binary: bool = False
    validate: bool = True
    absolute_paths: bool = False

    def get_simulation_directory(self) -> Path:
        return self.sim_directory

    def is_absolute_paths(self) -> bool:
        return self.absolute_paths

    def is_binary(self) -> bool:
        return self.binary

    def is_validate(self) -> bool:
        return self.validate

    @property
    def output_directory(self) -> Path:
        return self.model_directory

    @output_directory.setter
    def output_directory(self, model_directory: Union[Path, str]) -> None:
        self.model_directory = Path(model_directory)
