import pathlib
import shutil
from imod.wq.pkgbase import Package


class HorizontalFlowBarrier(Package):
    """
    Horizontal Flow Barrier package.

    Parameters
    ----------
    hfbfile: str
        is the file location of the imod-wq hfb-file. This file contains cell-
        to-cell resistance values. The hfb file can be constructed from generate
        files using imod-batch. No checks are implemented for this file, user is
        responsible for consistency with model.
    """

    _pkg_id = "hfb6"

    _template = "[hfb6]\n" "    hfbfile = {hfbfile}\n\n"

    def __init__(
        self,
        hfbfile,
    ):
        super().__init__()
        self["hfbfile"] = hfbfile

    def _render(self, modelname, directory, *args, **kwargs):
        self.hfbfile = f"{modelname}.hfb"
        d = {"hfbfile": f"{directory.as_posix()}/{modelname}.hfb"}

        return self._template.format(**d)

    def save(self, directory):
        """Overloads save function.
        Saves hfbfile to directory
        assumes _render() to have run previously"""
        directory.mkdir(exist_ok=True)  # otherwise handled by idf.save

        path_hfb = pathlib.Path(str(self["hfbfile"].values))

        shutil.copyfile(path_hfb, directory / self.hfbfile)

    def _pkgcheck(self, ibound=None):
        pass
