from pathlib import Path


class WriteContext:
    def __init__(
        self,
        sim_directory: Path = ".",
        binary: bool = False,
        validate: bool = True,
        absolute_paths: bool = False,
    ) -> None:
        self.sim_directory = sim_directory
        self.absolute_paths = absolute_paths
        self.validate = validate
        self.binary = binary

    def get_simulation_directory(self) -> Path:
        return self.sim_directory

    def is_absolute_paths(self) -> bool:
        return self.absolute_paths

    def is_binary(self) -> bool:
        return self.binary

    def is_validate(self) -> bool:
        return self.validate
    

    def set_model_directory(self, model_directory: Path) -> None:
       self.model_directory = model_directory

    def get_model_directory(self) -> Path:
       return Path(self.model_directory)