from imod.mf6.model import Modflow6Model
from imod.mf6.package import Package


def get_models(simulation: "Modflow6Simulation"):
    return {
        model_name: model
        for model_name, model in simulation.items()
        if isinstance(model, Modflow6Model)
    }


def get_packages(simulation: "Modflow6Simulation"):
    return {
        pkg_name: pkg
        for pkg_name, pkg in simulation.items()
        if isinstance(pkg, Package)
    }
