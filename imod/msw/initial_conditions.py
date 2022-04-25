import pathlib
import shutil

from imod.msw.fixed_format import VariableMetaData
from imod.msw.pkgbase import MetaSwapPackage


class InitialConditionsEquilibrium(MetaSwapPackage):
    """
    Use an equilibrium profile to initialize the model.

    This class is responsible for the file `init_svat.inp`
    """

    _file_name = "init_svat.inp"
    _option = "Equilibrium"
    _metadata_dict = {}

    def __init__(self):
        super().__init__()

    def _render(self, file, *args):
        file.write(self._option + "\n")


class InitialConditionsRootzonePressureHead(MetaSwapPackage):
    """
    Use the pF-value of the root zone pressure head as initial condition.

    This class is responsible for the file `init_svat.inp`

    Parameters
    ----------
    initial_pF: float
        Initial pF value to be used for all soil columns.
    """

    _file_name = "init_svat.inp"
    _option = "Rootzone_pF"
    _metadata_dict = {
        "initial_pF": VariableMetaData(6, 0.0, 6.0, float),
    }

    def __init__(self, initial_pF=2.2):
        super().__init__()
        self.dataset["initial_pF"] = initial_pF

    def _render(self, file, *args):
        file.write(self._option + "\n")

        dataframe = self.dataset.assign_coords(index=[0]).to_dataframe()

        self.write_dataframe_fixed_width(file, dataframe)


class InitialConditionsPercolation(MetaSwapPackage):
    """
    The precipitation intensity at the starting time (iybg, tdbg in
    PARA_SIM.INP) is used for initializing the percolation flux in the profiles.
    This type of initialization is normally done separately from the actual run,
    using a specially prepared meteo-input file. After letting the model reach
    near equilibrium by letting it run for a number of years, the saved state is
    used for the initialization of subsequent runs.

    This class is responsible for the file `init_svat.inp`
    """

    _file_name = "init_svat.inp"
    _option = "MeteoInputP"
    _metadata_dict = {}

    def __init__(self):
        super().__init__()

    def _render(self, file, *args):
        file.write(self._option + "\n")


class InitialConditionsSavedState(MetaSwapPackage):
    """
    Use saved state of a previous MetaSWAP run as initial condition.

    This class is responsible for the file `init_svat.inp`

    Parameters
    ----------
    saved_state: Path or str
        Path to a previously saved state. This file will be copied to
        init_svat.inp.

    """

    _file_name = "init_svat.inp"
    _option = "Saved_State"
    _metadata_dict = {}

    def __init__(self, saved_state):
        super().__init__()
        self.saved_state = saved_state

    def write(self, directory, *args):
        directory = pathlib.Path(directory)
        filename = directory / self._file_name

        shutil.copyfile(self.saved_state, filename)
