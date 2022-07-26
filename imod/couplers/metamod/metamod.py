from pathlib import Path
from typing import Optional, Union

import tomli_w

from imod.couplers.metamod.node_svat_mapping import NodeSvatMapping
from imod.couplers.metamod.rch_svat_mapping import RechargeSvatMapping
from imod.couplers.metamod.wel_svat_mapping import WellSvatMapping
from imod.mf6 import Modflow6Simulation
from imod.msw import GridData, MetaSwapModel, Sprinkling


class MetaMod:
    """
    The MetaMod class creates the necessary input files for coupling MetaSWAP to
    MODFLOW 6.

    Parameters
    ----------
    msw_model : MetaSwapModel
        The MetaSWAP model that should be coupled.
    mf6_simulation : Modflow6Simulation
        The Modflow6 simulation that should be coupled.
    mf6_rch_pkgkey: str
        Key of Modflow 6 recharge package to which MetaSWAP is coupled.
    mf6_wel_pkgkey: str or None
        Optional key of Modflow 6 well package to which MetaSWAP sprinkling is
        coupled.
    """

    _toml_name = "imod_coupler.toml"
    _modflow6_model_dir = "Modflow6"
    _metaswap_model_dir = "MetaSWAP"

    def __init__(
        self,
        msw_model: MetaSwapModel,
        mf6_simulation: Modflow6Simulation,
        mf6_rch_pkgkey: str,
        mf6_wel_pkgkey: Optional[str] = None,
    ):
        self.msw_model = msw_model
        self.mf6_simulation = mf6_simulation
        self.mf6_rch_pkgkey = mf6_rch_pkgkey
        self.mf6_wel_pkgkey = mf6_wel_pkgkey

        self.is_sprinkling = self._check_coupler_and_sprinkling()

    def _check_coupler_and_sprinkling(self):
        mf6_rch_pkgkey = self.mf6_rch_pkgkey
        mf6_wel_pkgkey = self.mf6_wel_pkgkey

        gwf_names = self._get_gwf_modelnames()

        # Assume only one groundwater flow model
        # FUTURE: Support multiple groundwater flow models.
        gwf_model = self.mf6_simulation[gwf_names[0]]

        if mf6_rch_pkgkey not in gwf_model.keys():
            raise ValueError(
                f"No package named {mf6_rch_pkgkey} detected in Modflow 6 model. "
                "iMOD_coupler requires a Recharge package."
            )

        sprinkling_key = self.msw_model._get_pkg_key(Sprinkling, optional_package=True)

        sprinkling_in_msw = sprinkling_key is not None
        sprinkling_in_mf6 = mf6_wel_pkgkey in gwf_model.keys()

        if sprinkling_in_msw and not sprinkling_in_mf6:
            raise ValueError(
                f"No package named {mf6_wel_pkgkey} found in Modflow 6 model, "
                "but Sprinkling package found in MetaSWAP. "
                "iMOD Coupler requires a Well Package "
                "to couple wells."
            )
        elif not sprinkling_in_msw and sprinkling_in_mf6:
            raise ValueError(
                f"Modflow 6 Well package {mf6_wel_pkgkey} specified for sprinkling, "
                "but no Sprinkling package found in MetaSWAP model."
            )
        elif sprinkling_in_msw and sprinkling_in_mf6:
            return True
        else:
            return False

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

        # Write exchange files
        exchange_dir = directory / "exchanges"
        exchange_dir.mkdir(mode=755, exist_ok=True)
        self.write_exchanges(exchange_dir, self.mf6_rch_pkgkey, self.mf6_wel_pkgkey)

        coupling_dict = self._get_coupling_dict(
            exchange_dir, self.mf6_rch_pkgkey, self.mf6_wel_pkgkey
        )

        self.write_toml(
            directory,
            modflow6_dll,
            metaswap_dll,
            metaswap_dll_dependency,
            coupling_dict,
        )

    def write_toml(
        self,
        directory: Union[str, Path],
        modflow6_dll: Union[str, Path],
        metaswap_dll: Union[str, Path],
        metaswap_dll_dependency: Union[str, Path],
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
        metaswap_dll: str or Path
            Path to metaswap .dll. You can obtain this library by downloading
            `the last iMOD5 release
            <https://oss.deltares.nl/web/imod/download-imod5>`_
        metaswap_dll_dependency: str or Path
            Directory with metaswap .dll dependencies. Directory should contain:
            [fmpich2.dll, mpich2mpi.dll, mpich2nemesis.dll, TRANSOL.dll]. You
            can obtain these by downloading `the last iMOD5 release
            <https://oss.deltares.nl/web/imod/download-imod5>`_
        coupling_dict: dict
            Dictionary with names of coupler packages and paths to mappings.
        """
        # force to Path
        directory = Path(directory)

        toml_path = directory / self._toml_name

        coupler_toml = {
            "timing": False,
            "log_level": "INFO",
            "log_file": "imod_coupler.log",
            "driver_type": "metamod",
            "driver": {
                "kernels": {
                    "modflow6": {
                        "dll": str(modflow6_dll),
                        "work_dir": f".\\{self._modflow6_model_dir}",
                    },
                    "metaswap": {
                        "dll": str(metaswap_dll),
                        "work_dir": f".\\{self._metaswap_model_dir}",
                        "dll_dep_dir": str(metaswap_dll_dependency),
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
            if value._pkg_id == "model"
        ]

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

        coupling_dict[
            "mf6_msw_node_map"
        ] = f"./{directory.name}/{NodeSvatMapping._file_name}"

        coupling_dict["mf6_msw_recharge_pkg"] = mf6_rch_pkgkey
        coupling_dict[
            "mf6_msw_recharge_map"
        ] = f"./{directory.name}/{RechargeSvatMapping._file_name}"

        coupling_dict["enable_sprinkling"] = self.is_sprinkling

        if self.is_sprinkling:
            coupling_dict["mf6_msw_well_pkg"] = mf6_wel_pkgkey
            coupling_dict[
                "mf6_msw_sprinkling_map"
            ] = f"./{directory.name}/{WellSvatMapping._file_name}"

        return coupling_dict

    def write_exchanges(
        self,
        directory: Union[str, Path],
        mf6_rch_pkgkey: str,
        mf6_wel_pkgkey: Optional[str],
    ):
        """
        Write exchange files (.dxc) which map MetaSWAP's svats to Modflow 6 node
        numbers, recharge ids, and well ids.

        Parameters
        ----------
        directory: str or Path
            Directory where .dxc files are written.
        mf6_rch_pkgkey: str
            Key of Modflow 6 recharge package to which MetaSWAP is coupled.
        mf6_wel_pkgkey: str
            Key of Modflow 6 well package to which MetaSWAP sprinkling is
            coupled.
        """

        gwf_names = self._get_gwf_modelnames()

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

        recharge = gwf_model[mf6_rch_pkgkey]

        rch_mapping = RechargeSvatMapping(svat, recharge)
        rch_mapping.write(directory, index, svat)

        if self.is_sprinkling:
            well = gwf_model[mf6_wel_pkgkey]
            well_mapping = WellSvatMapping(svat, well)
            well_mapping.write(directory, index, svat)
