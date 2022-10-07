import numpy as np
import xarray as xr

from .pkgbase import Package


class FlowObservationList:
    """
    List-based input for the Observation package.
    
    This class is closest to the MODFLOW6 file structure.
    
    Parameters
    ----------
    head: xr.DataArray, optional
        Must contain head_name coordinate.
    head_file: str, optional
    head_binary: bool, default is False.
    drawdown: xr.DataArray, optional
        Must contain drawdown_name coordinate.
    drawdown_file: str, optional
    drawdown_binary: bool, default is False.
    flow_ja_face: xr.DataArray, optional
        Must contain flow_ja_face_name coordinate.
    flow_ja_face_file: str, optional
    flow_ja_face_binary: bool, default is False
    digits: int, optional
    print_input: bool, default is False
    """
    def __init__(
        self,
        head: xr.DataArray=None,
        head_file: str=None,
        head_binary: bool=False,
        drawdown: xr.DataArray=None,
        drawdown_file: str=None,
        drawdown_binary: bool=False,
        flow_ja_face: xr.DataArray=None,
        flow_ja_face_file: str=None,
        flow_ja_face_binary: bool=False,
        digits: int = None,
        print_input: bool=False,
    ):
        self.dataset = xr.Dataset()
        self.dataset["head"] = head
        self.dataset["head_file"] = head_file
        self.dataset["head_binary"] = head_binary
        self.dataset["drawdown"] = drawdown
        self.dataset["drawdown_file"] = drawdown_file
        self.dataset["drawdown_binary"] = drawdown_binary
        self.dataset["flow_ja_face"] = flow_ja_face
        self.dataset["flow_ja_face_file"] = flow_ja_face_file
        self.dataset["flow_ja_face_binary"] = flow_ja_face_binary
        self.dataset["digits"] = digits
        self.dataset["print_input"] = print_input
    
    def render(self, directory, pkgname, globaltimes, binary):
        d = {}
        for variable in [
            "head_file",
            "head_binary",
            "drawdown_file",
            "drawdown_binary",
            "flow_ja_face_file",
            "flow_ja_face_binary",
            "print_input",   
            "digits",
        ]:
            value = self.dataset[variable][()]
            if self._valid(value):
                d[variable] = value
        
        for variable in ["head", "drawdown"]:
            da = self.dataset[variable]
            if da is not None:
                names = da[f"{variable}_name"].values
                cell_ids = np.atleast_2d(da.values)
                d[variable] = [(name, cell_id) for name, cell_id in zip(names, cell_ids)]
    
        return self._template.render(d)
        

class FlowObservationGrid:
    """
    Parameters
    ----------
    head: xarray.DataArray | xugrid.UgridDataArray of float, optional
        Where to observe the simulated head.
    drawdown: xarray.DataArray | xugrid.UgridDataArray of float, optional
        Where to observe the simulated drawdown.
    digits: int, optional
    print_input: bool, default value is ``False``.
    binary: bool, default value is ``False``.
    """
    def __init__(
        self,
        head=None,
        drawdown=None,
        digits=None,
        print_input=False,
        binary=False,
    ):
        super().__init__(locals())
        self.dataset["head"] = head
        self.dataset["drawdown"] = drawdown
        self.dataset["digits"] = digits
        self.dataset["print_input"] = print_input
        self.dataset["binary"] = binary


class FlowObservationPoints:
    def __init__(
        self,
    )
