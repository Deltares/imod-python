import pathlib

import xarray as xr

from imod.msw.pkgbase import Package, VariableMetaData


class Sprinkling(Package):
    """
    This contains the sprinkling capacities of links between SVAT units and groundwater/ surface water locations.

    This class is responsible for the file `scap_svat.inp`
    """

    _file_name = "scap_svat.inp"
    _metadata_dict = {
        "svat": VariableMetaData(10, 1, 99999999, int),
        "max_abstraction_groundwater_mm_d": VariableMetaData(8, None, None, str),
        "max_abstraction_surfacewater_mm_d": VariableMetaData(8, None, None, str),
        "max_abstraction_groundwater_m3_d": VariableMetaData(8, 0.0, 1e9, float),
        "max_abstraction_surfacewater_m3_d": VariableMetaData(8, 0.0, 1e9, float),
        "svat_groundwater": VariableMetaData(10, None, None, str),
        "layer": VariableMetaData(6, 1, 9999, int),
        "trajectory": VariableMetaData(10, None, None, str),
    }

    def __init__(
        self,
        max_abstraction_groundwater: xr.DataArray,
        min_abstraction_groundwater: xr.DataArray,
        layer: xr.DataArray,
        active: xr.DataArray,
    ):
        super().__init__()
        self.dataset["max_abstraction_groundwater"] = max_abstraction_groundwater
        self.dataset["min_abstraction_groundwater"] = min_abstraction_groundwater
        self.dataset["layer"] = layer
        self.dataset["active"] = active

    def _render(self, file):
        # TODO: Resolve open questions then implement
        # 1) Should it be possible to only add sprinkling for groundwater or surface water?
        # If yes, how to best expose this API?
        # 2) Which of the input parameters depends on `subunit`?
        # 3) `layer` should probably not be given directly, but what would be better?
        raise NotImplementedError("Needs to put actual logic here.")

        # Create DataFrame
        # dataframe = pd.DataFrame(
        #     {
        #         "svat": svat,
        #         "max_abstraction_groundwater_mm_d": max_abstraction_groundwater_mm_d,
        #         "max_abstraction_surfacewater_mm_d": max_abstraction_surfacewater_mm_d,
        #         "max_abstraction_groundwater_m3_d": max_abstraction_groundwater_m3_d,
        #         "max_abstraction_surfacewater_m3_d": max_abstraction_surfacewater_m3_d,
        #         "svat_groundwater": svat_groundwater,
        #         "layer": layer,
        #         "trajectory": trajectory,
        #     }
        # )

        # self._check_range(dataframe)

        # return self.write_dataframe_fixed_width(file, dataframe)

    def write(self, directory):
        directory = pathlib.Path(directory)

        filename = directory / self._file_name
        with open(filename, "w") as f:
            self._render(f)
