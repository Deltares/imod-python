from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from imod.mf6.model import Modflow6Model
from imod.mf6.package import Package

if TYPE_CHECKING:
    from imod.mf6 import Modflow6Simulation


def get_models(simulation: Modflow6Simulation) -> Dict[str, Modflow6Model]:
    return {
        model_name: model
        for model_name, model in simulation.items()
        if isinstance(model, Modflow6Model)
    }


def get_packages(simulation: Modflow6Simulation) -> Dict[str, Package]:
    return {
        pkg_name: pkg
        for pkg_name, pkg in simulation.items()
        if isinstance(pkg, Package)
    }
