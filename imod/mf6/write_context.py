from dataclasses import dataclass
from os.path import relpath
from pathlib import Path
from typing import Optional, Union


@dataclass
class WriteContext:
    """
    This class is used in the process of writing modflow inputfiles.
    It is a container for options that are used when writing.

    Parameters
    ----------
    simulation_directory: Path
        The directory where the .nam file for the modflow simulation will be written
    binary: bool
        If True, bulk data will be written in a binary format readable by modflow. Regular package input files
        will still be rendered as text.
    absolute_paths: bool
        If True, paths in the modlfow inputfiles will be rendered as absoule paths on your system.
        This makes the modflow input files less portable to other systems but facilitates reading them by Flopy
    _output_directory: Optional[Path] = None
        The directory where the next outputfile will be written. Users do not need to set this parameter
    """

    simulation_directory: Path = "."
    binary: bool = False
    absolute_paths: bool = False
    _output_directory: Optional[Path] = None

    @property
    def current_write_directory(self) -> Optional[Path]:
        return self._output_directory

    @current_write_directory.setter
    def current_write_directory(self, model_directory: Union[Path, str]) -> None:
        self._output_directory = Path(model_directory)

    def get_adjusted_output_directory(self) -> Path:
        if self.absolute_paths:
            return self.current_write_directory
        return Path(
            relpath(self.current_write_directory, self.simulation_directory)
        )