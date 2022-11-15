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

    _pkg_id = "hfb"

    _template = (
        "[hfb]\n"
        "    hfbfile = {hfbfile}\n"
    )

    def __init__(
        self,
        hfbfile,
    ):
        super().__init__()
        self["hfbfile"] = hfbfile
    
    def _render(self, directory, *args, **kwargs):
        path_hfb = pathlib.Path(str(self["hfbfile"].values))
        d = {"hfbfile": f"hfb/{path_hfb.name}"}

        return self._template.format(**d)
            
    def save(self, directory):
        """Overloads save function. 
        Saves hfbfile to directory"""
        directory.mkdir(exist_ok=True)
        
        path_hfb = pathlib.Path(str(self["hfbfile"].values))
        
        shutil.copyfile(path_hfb, directory / path_hfb.name)

    def _pkgcheck(self, ibound=None):
        pass
