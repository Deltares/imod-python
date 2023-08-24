from __future__ import annotations
from copy import deepcopy
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
    use_binary: bool
        If True, bulk data will be written in a binary format readable by modflow. Regular package input files
        will still be rendered as text.
    use_absolute_paths: bool
        If True, paths in the modlfow inputfiles will be rendered as absoule paths on your system.
        This makes the modflow input files less portable to other systems but facilitates reading them by Flopy
   write_directory: Optional[Path] = None
        The directory where the next outputfile will be written. Users do not need to set this parameter. If not provided
        it will be set to the simulation_directrory.
    """

    def __init__(
        self,
        simulation_directory: Path = ".",
        use_binary: bool = False,
        use_absolute_paths: bool = False,
        write_directory: Optional[Union[str, Path]] = None,
    ):
        self.simulation_directory = Path(simulation_directory)
        self.use_binary = use_binary
        self.use_absolute_paths = use_absolute_paths
        self.write_directory = (
            Path(write_directory)
            if write_directory is not None
            else self.simulation_directory
        )

    def get_formatted_write_directory(self) -> Path:
        if self.use_absolute_paths:
            return self.write_directory
        return Path(relpath(self.write_directory, self.simulation_directory))

    def copy_with_new_write_directory(self, new_write_directory: Path) -> WriteContext:
        new_context = deepcopy(self)
        new_context.write_directory = Path(new_write_directory)
        return new_context

    @property
    def root_directory(self) -> Path:
        """
        returns the simulation directory, or nothing, depending on use_absolute_paths; use this to compose paths
        that are in agreement with the use_absolute_paths setting.
        """
        if self.use_absolute_paths:
            return self.simulation_directory
        else:
            return Path("")
