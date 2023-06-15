from pathlib import Path
from typing import Optional, Union

import ribasim
import tomli_w

from imod.mf6 import Modflow6Simulation
from imod.mf6.model import Modflow6Model


class RibaMod:
    """
    The RibaMod class creates the necessary input files for coupling Ribasim to
    MODFLOW 6.

    Parameters
    ----------
    ribasim_model : ribasim.model
        The Ribasim model that should be coupled.
    mf6_simulation : Modflow6Simulation
        The Modflow6 simulation that should be coupled.
    """

    _toml_name = "imod_coupler.toml"
    _ribasim_model_dir = "ribasim"
    _modflow6_model_dir = "modflow6"

    def __init__(
        self,
        ribasim_model: ribasim.Model,
        mf6_simulation: Modflow6Simulation,
    ):
        self.ribasim_model = ribasim_model
        self.mf6_simulation = mf6_simulation

    def write(
        self,
        directory: Union[str, Path],
        modflow6_dll: Union[str, Path],
        ribasim_dll: Union[str, Path],
        ribasim_dll_dependency: Union[str, Path],
        modflow6_write_kwargs: Optional[dict] = None,
    ):
        """
        Write Ribasim and Modflow 6 model with exchange files, as well as a
        ``.toml`` file which configures the iMOD Coupler run.

        Parameters
        ----------
        directory: str or Path
            Directory in which to write the coupled models
        modflow6_dll: str or Path
            Path to modflow6 .dll. You can obtain this library by downloading
            `the last iMOD5 release
            <https://oss.deltares.nl/web/imod/download-imod5>`_
        ribasim_dll: str or Path
            Path to ribasim .dll.
        ribasim_dll_dependency: str or Path
            Directory with ribasim .dll dependencies.
        modflow6_write_kwargs: dict
            Optional dictionary with keyword arguments for the writing of
            Modflow6 models. You can use this for example to turn off the
            validation at writing (``validation=False``) or to write text files
            (``binary=False``)
        """

        if modflow6_write_kwargs is None:
            modflow6_write_kwargs = {}

        # force to Path
        directory = Path(directory)
        self.mf6_simulation.write(
            directory / self._modflow6_model_dir,
            **modflow6_write_kwargs,
        )
        self.ribasim_model.write(directory / self._ribasim_model_dir)

        # TODO
        coupling_dict = {}

        # # Write exchange files
        # exchange_dir = directory / "exchanges"
        # exchange_dir.mkdir(mode=755, exist_ok=True)
        # self.write_exchanges(exchange_dir, self.mf6_rch_pkgkey, self.mf6_wel_pkgkey)

        # coupling_dict = self._get_coupling_dict(
        #     exchange_dir, self.mf6_rch_pkgkey, self.mf6_wel_pkgkey
        # )

        self.write_toml(
            directory,
            modflow6_dll,
            ribasim_dll,
            ribasim_dll_dependency,
            coupling_dict,
        )

    def write_toml(
        self,
        directory: Union[str, Path],
        modflow6_dll: Union[str, Path],
        ribasim_dll: Union[str, Path],
        ribasim_dll_dependency: Union[str, Path],
        coupling_dict: dict,
    ):
        """
        Write .toml file which configures the imod coupler run.

        Parameters
        ----------
        directory: str or Path
            Directory in which to write the .toml file.
        modflow6_dll: str or Path
            Path to modflow6 .dll. You can obtain this library by downloading
            `the last iMOD5 release
            <https://oss.deltares.nl/web/imod/download-imod5>`_
        ribasim_dll: str or Path
            Path to ribasim .dll.
        ribasim_dll_dependency: str or Path
            Directory with ribasim .dll dependencies.
        coupling_dict: dict
            Dictionary with names of coupler packages and paths to mappings.
        """
        # force to Path
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        toml_path = directory / self._toml_name
        coupler_toml = {
            "timing": False,
            "log_level": "INFO",
            "driver_type": "ribamod",
            "driver": {
                "kernels": {
                    "modflow6": {
                        "dll": str(modflow6_dll),
                        "work_dir": f".\\{self._modflow6_model_dir}",
                    },
                    "ribasim": {
                        "dll": str(ribasim_dll),
                        "dll_dep_dir": str(ribasim_dll_dependency),
                        "config_file": str(
                            f"{self._ribasim_model_dir}/{self.ribasim_model.modelname}.toml"
                        ),
                    },
                },
                "coupling": [coupling_dict],
            },
        }

        with open(toml_path, "wb") as f:
            tomli_w.dump(coupler_toml, f)

    def _get_gwf_modelnames(self):
        """
        Get names of gwf models in mf6 simulation
        """
        return [
            key
            for key, value in self.mf6_simulation.items()
            if isinstance(value, Modflow6Model)
        ]

    # TODO:
    def _get_coupling_dict(
        self,
        directory: Union[str, Path],
        mf6_rch_pkgkey: str,
        mf6_wel_pkgkey: Optional[str],
    ) -> dict:
        """
        Get dictionary with names of coupler packages and paths to mappings.

        Parameters
        ----------
        directory: str or Path
            Directory where .dxc files are written.
        mf6_rch_pkgkey: str
            Key of Modflow 6 recharge package to which MetaSWAP is coupled.
        mf6_wel_pkgkey: str
            Key of Modflow 6 well package to which MetaSWAP sprinkling is
            coupled.

        Returns
        -------
        coupling_dict: dict
            Dictionary with names of coupler packages and paths to mappings.
        """

        coupling_dict = {}

        gwf_names = self._get_gwf_modelnames()

        # Assume only one groundwater flow model
        # FUTURE: Support multiple groundwater flow models.
        coupling_dict["mf6_model"] = gwf_names[0]

        return coupling_dict

    # TODO
    def write_exchanges(
        self,
        directory: Union[str, Path],
        mf6_riv_pkgkey: str,
    ):
        """
        Write exchange files (.dxc) which map MetaSWAP's svats to Modflow 6 node
        numbers, recharge ids, and well ids.

        Parameters
        ----------
        directory: str or Path
            Directory where .dxc files are written.
        mf6_river_pkgkey: str
            Key of Modflow 6 river package to which Ribasim is coupled.
        """

        # TODO
        # gwf_names = self._get_gwf_modelnames()
        # Assume only one groundwater flow model
        # FUTURE: Support multiple groundwater flow models.
        # gwf_model = self.mf6_simulation[gwf_names[0]]
        # dis = gwf_model[gwf_model._get_pkgkey("dis")]
        # river = gwf_model[mf6_riv_pkgkey]

        return
