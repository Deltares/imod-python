import collections
import pathlib


import imod


class Modflow6Simulation(collections.UserDict):
    def __init__(self, name):
        super(__class__, self).__init__()
        self.name = name

    def __setitem__(self, key, value):
        # Synchronize names
        if isinstance(imod.mf6.model.Modflow6, value):
            value.modelname = key
        super(__class__, self).__setitem__(key, value)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def render(self):
        """Renders simulation namefile"""
        # includes timing, models, exchanges, solution groups
        return ""

    def write(self, directory="."):
        if isinstance(directory, str):
            directory = pathlib.Path(directory)
        directory.mkdir(exist_ok=True, parents=True)

        # Write simulation namefile
        mfsim_content = self.render()
        mfsim_path = directory / "mfsim.nam"
        with open(mfsim_path, "w") as f:
            f.write(mfsim_content)

        # Write individual models
        for model in self.values():
            model.write(directory)

