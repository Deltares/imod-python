from pathlib import Path
from dataclasses import dataclass

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

    def set_output_directory(self, model_directory: Path) -> None:
        self.model_directory = model_directory

    def get_output_directory(self) -> Path:
        return Path(self.model_directory)
