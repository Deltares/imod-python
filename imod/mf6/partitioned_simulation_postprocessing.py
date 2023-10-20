from pathlib import Path

import imod
from imod.mf6.model import GroundwaterFlowModel
from imod.mf6.simulation import Modflow6Simulation
from imod.typing.grid import GridDataArray, merge


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

    grb_files = list(model_directory.glob("*.grb"))
    if len(grb_files) != 1:
        raise RuntimeError(
            f"there should be exactly one grb file in directory {model_directory}."
        )
    return grb_files[0]
    """
    Given a directory path, returns the path of the grb file in it. Raises an
    exception if there is not exactly 1 grb file in it.
    """

    cbc_files = list(model_directory.glob("**/*.cbc"))
    if len(cbc_files) != 1:
        raise RuntimeError(
            f"there should be exactly one grb file in directory {model_directory}."
        )
    return cbc_files[0]


def merge_balances(simulation_dir: Path, model_names: List[str]) -> GridDataArray:
    """
    This function merges the head output of a split simulation into a single
    head file. Both structured and unstructured grids are supported.
    """
    fig, ax = plt.subplots()
    allKeys = set()
    cbcs = []
    for modelname in model_names:
        modelDirectory = simulation_dir / modelname
        cbc_path = get_cbc_file_path(modelDirectory)
        grb_path  = get_grb_file_path(modelDirectory)
        cbc = imod.mf6.open_cbc(cbc_path, grb_path)
        allKeys.update(list(cbc.keys()))
        cbcs.append(cbc)


    merged_keys =[]
    for key in allKeys:
        balances_of_keys = []
        for balance in cbcs:
            if key in balance.keys():
                balance[key] = balance[key].rename(key)
                balances_of_keys.append(balance[key])
        x = merge(*balances_of_keys)
        if not is_unstructured(x): 
            x.rename(key)
        merged_keys.append(x) 
    balances =merge(merged_keys)

    return balances
