from pathlib import Path

import imod
from imod.mf6.model import GroundwaterFlowModel
from imod.mf6.simulation import Modflow6Simulation
from imod.typing.grid import GridDataArray, merge, is_unstructured, merge_to_dataset
from typing import List, Dict


def merge_heads(simulation_dir: Path, simulation: Modflow6Simulation) -> GridDataArray:
    """
    This function merges the head output of a split simulation into a single
    head file. Both structured and unstructured grids are supported.
    """
    model_names = list(
        simulation.get_models_of_type(GroundwaterFlowModel._model_id).keys()
    )
    heads = []
    for modelname in model_names:
        modelDirectory = simulation_dir / modelname
        grb_path = _get_grb_file_path(modelDirectory)
        head = imod.mf6.open_hds(
            modelDirectory / f"{modelname}.hds",
            grb_path,
        )
        heads.append(head)

    head = merge(heads)
    return head["head"]


def _get_grb_file_path(model_directory: Path) -> Path:
    """
    Given a directory path, returns the path of the grb file in it. Raises an
    exception if there is not exactly 1 grb file in it.
    """
    return _get_single_file(model_directory, "grb")


def _get_cbc_file_path(model_directory: Path) -> Path:
    """
    Given a directory path, returns the path of the grb file in it. Raises an
    exception if there is not exactly 1 grb file in it.
    """
    return _get_single_file(model_directory, "cbc")

def _get_single_file(model_directory: Path, extension: str) -> Path:
    """
    Given a directory path, and an exttension, it returns a single file in that directory with that extension. 
    It raises an exception if there are multiple files with the same extension.
    """
    candidate_files = list(model_directory.glob(f"*.{extension}"))
    if len(candidate_files) != 1:
        raise RuntimeError(
            f"there should be exactly one {extension} file in directory {model_directory}."
        )    
    return candidate_files[0]

def merge_balances(simulation_dir: Path, simulation:Modflow6Simulation) -> Dict[str,GridDataArray]:
    """
    This function merges the head output of a split simulation into a single
    head file. Both structured and unstructured grids are supported.
    """

    model_names = list(
        simulation.get_models_of_type(GroundwaterFlowModel._model_id).keys()
    )
    unique_balance_keys = set()
    cbc_per_partition = []
    for modelname in model_names:
        modelDirectory = simulation_dir / modelname
        cbc_path = _get_cbc_file_path(modelDirectory)
        grb_path  = _get_grb_file_path(modelDirectory)
        cbc = imod.mf6.open_cbc(cbc_path, grb_path)
        unique_balance_keys.update(list(cbc.keys()))
        cbc_per_partition.append(cbc)
    unique_balance_keys = list(unique_balance_keys)

    merged_keys ={}
    for key in unique_balance_keys:
        balances_of_keys = []
        for balance in cbc_per_partition:
            if key in balance.keys():
                balance[key] = balance[key].rename(key)
                balances_of_keys.append(balance[key])
        x = merge(balances_of_keys)
        darray =  x.to_array(key).drop_vars(key)
        merged_keys[key] =darray.sel({key:0})
    return merged_keys
