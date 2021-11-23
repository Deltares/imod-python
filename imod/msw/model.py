import collections
import pathlib

import jinja2


class Model(collections.UserDict):
    def __setitem__(self, key, value):
        # TODO: Add packagecheck
        super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v


class MetaSwapModel(Model):
    """
    Contains data and writes consistent model input files
    """

    _pkg_id = "model"

    def _initialize_template(self):
        # TODO: adapt for metaswap

        loader = jinja2.PackageLoader("imod", "templates/mf6")
        env = jinja2.Environment(loader=loader, keep_trailing_newline=True)
        self._template = env.get_template("gwf-nam.j2")

    def __init__(self, newton=False, under_relaxation=False):
        super().__init__()
        self._initialize_template()

    def render(self, modelname):
        """Render parameter setting file"""
        # TODO: adapt for metaswap

        dir_for_render = pathlib.Path(modelname)
        d = {"newton": self.newton, "under_relaxation": self.under_relaxation}
        packages = []
        for pkgname, pkg in self.items():
            # Add the six to the package id
            pkg_id = pkg._pkg_id
            key = f"{pkg_id}6"
            path = dir_for_render / f"{pkgname}.{pkg_id}"
            packages.append((key, path.as_posix()))
        d["packages"] = packages
        return self._template.render(d)

    def write(self, wdir, modelname, globaltimes):
        """
        Write parameter setting file
        Write packages
        """
        # TODO: adapt for metaswap

        wdir = pathlib.Path(wdir)

        modeldirectory = wdir / modelname
        modeldirectory.mkdir(exist_ok=True, parents=True)

        # write model namefile
        namefile_content = self.render(modelname)
        namefile_path = modeldirectory / f"{modelname}.nam"
        with open(namefile_path, "w") as f:
            f.write(namefile_content)

        # write package contents
        for pkgname, pkg in self.items():
            pkg.write(modeldirectory, pkgname, globaltimes)
