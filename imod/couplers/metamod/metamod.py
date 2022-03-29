import textwrap
from pathlib import Path
from typing import Union

import tomli_w

from imod.couplers.metamod.node_svat_mapping import NodeSvatMapping
from imod.couplers.metamod.rch_svat_mapping import RechargeSvatMapping
from imod.couplers.metamod.wel_svat_mapping import WellSvatMapping
from imod.mf6 import Modflow6Simulation
from imod.msw import GridData, MetaSwapModel, Sprinkling


class MetaMod:
    """
    The MetaMod class creates the necessary input files for coupling MetaSWAP to MODFLOW 6.

    Parameters
    ----------
    msw_model : MetaSwapModel
        The MetaSWAP model that should be coupled.
    mf6_simulation : Modflow6Simulation
        The Modflow6 simulation that should be coupled.
    """

    _toml_name = "metamod.toml"
    _modflow6_model_dir = "Modflow6"
    _metaswap_model_dir = "MetaSWAP"

    def __init__(self, msw_model: MetaSwapModel, mf6_simulation: Modflow6Simulation):
        self.msw_model = msw_model
        self.mf6_simulation = mf6_simulation

    def write(
        self,
        directory: Union[str, Path],
        modflow6_dll: Union[str, Path],
        metaswap_dll: Union[str, Path],
        metaswap_dll_dependency: Union[str, Path],
    ):
        """
        Write MetaSWAP and Modflow 6 model with exchange files, as well as a
        ``.toml`` file which configures the imod coupler run.

        Parameters
        ----------
        directory: str or Path
            Directory in which to write the coupled models
        modflow6_dll: str or Path
            Path to modflow6 .dll. You can obtain this library by downloading
            `the last iMOD5 release
            <https://oss.deltares.nl/web/imod/download-imod5>`_
        metaswap_dll: str or Path
            Path to metaswap .dll. You can obtain this library by downloading
            `the last iMOD5 release
            <https://oss.deltares.nl/web/imod/download-imod5>`_
        metaswap_dll_dependency: str or Path
            Directory with metaswap .dll dependencies. Directory should contain:
            [fmpich2.dll, mpich2mpi.dll, mpich2nemesis.dll, TRANSOL.dll]. You
            can obtain these by downloading `the last iMOD5 release
            <https://oss.deltares.nl/web/imod/download-imod5>`_
        """
        # force to Path
        directory = Path(directory)
        # For some reason the Modflow 6 model has to be written first, before
        # writing the MetaSWAP model. Else we get an Access Violation Error when
        # running the coupler.
        self.mf6_simulation.write(directory / self._modflow6_model_dir)
        self.msw_model.write(directory / self._metaswap_model_dir)
        # Exchange files should be in Modflow 6 model directory
        self.write_exchanges(directory / self._modflow6_model_dir)

        self.write_toml(directory, modflow6_dll, metaswap_dll, metaswap_dll_dependency)

    def write_toml(
        self,
        directory: Union[str, Path],
        modflow6_dll: Union[str, Path],
        metaswap_dll: Union[str, Path],
        metaswap_dll_dependency: Union[str, Path],
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
        metaswap_dll: str or Path
            Path to metaswap .dll. You can obtain this library by downloading
            `the last iMOD5 release
            <https://oss.deltares.nl/web/imod/download-imod5>`_
        metaswap_dll_dependency: str or Path
            Directory with metaswap .dll dependencies. Directory should contain:
            [fmpich2.dll, mpich2mpi.dll, mpich2nemesis.dll, TRANSOL.dll]. You
            can obtain these by downloading `the last iMOD5 release
            <https://oss.deltares.nl/web/imod/download-imod5>`_
        """
        # force to Path
        directory = Path(directory)

        toml_path = directory / self._toml_name

        coupler_toml = {
            "timing": False,
            "log_level": "INFO",
            "kernels": {
                "modflow6": {
                    "dll": str(modflow6_dll),
                    "model": f".\\{self._modflow6_model_dir}",
                },
                "metaswap": {
                    "dll": str(metaswap_dll),
                    "model": f".\\{self._metaswap_model_dir}",
                    "dll_dependency": str(metaswap_dll_dependency),
                },
            },
            "exchanges": [{"kernels": ["modflow6", "metaswap"]}],
        }

        with open(toml_path, "wb") as f:
            tomli_w.dump(coupler_toml, f)

    def write_exchanges(self, directory: Union[str, Path]):
        """
        Write exchange files (.dxc) which map MetaSWAP's svats to Modflow 6 node
        numbers, recharge ids, and well ids.

        Parameters
        ----------
        directory: str or Path
            Directory where .dxc files are written.
        """
        gwf_names = [
            key
            for key, value in self.mf6_simulation.items()
            if value._pkg_id == "model"
        ]
        # Assume only one groundwater flow model
        # FUTURE: Support multiple groundwater flow models.
        gwf_model = self.mf6_simulation[gwf_names[0]]

        grid_data_key = [
            pkgname
            for pkgname, pkg in self.msw_model.items()
            if isinstance(pkg, GridData)
        ][0]

        dis = gwf_model[gwf_model._get_pkgkey("dis")]

        index, svat = self.msw_model[grid_data_key].generate_index_array()
        grid_mapping = NodeSvatMapping(svat, dis)
        grid_mapping.write(directory, index, svat)

        # FUTURE: Not necessary after iMOD Coupler refactoring
        if "rch_msw" not in gwf_model.keys():
            raise ValueError(
                textwrap.dedent(
                    "No package named 'rch_msw' detected in Modflow 6 model. "
                    "iMOD_coupler requires a Recharge package with 'rch_msw' as name"
                )
            )

        recharge = gwf_model["rch_msw"]

        rch_mapping = RechargeSvatMapping(svat, recharge)
        rch_mapping.write(directory, index, svat)

        sprinkling_key = self.msw_model._get_pkg_key(Sprinkling, optional_package=True)

        # FUTURE: Not necessary after iMOD Coupler refactoring
        if (sprinkling_key is not None) and not ("wells_msw" in gwf_model.keys()):
            raise ValueError(
                textwrap.dedent(
                    "No package named 'wells_msw' found in Modflow 6 model, "
                    "but Sprinkling package found in MetaSWAP. "
                    "iMOD Coupler requires a Well Package named 'wells_msw' "
                    "to couple wells."
                )
            )
        elif "wells_msw" in gwf_model.keys():
            well = gwf_model["wells_msw"]

            well_mapping = WellSvatMapping(svat, well)
            well_mapping.write(directory, index, svat)
