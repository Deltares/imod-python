from imod.mf6.pkgbase import BoundaryCondition
from imod.ugrid_utils.snapping import snap_to_grid
import geopandas as gpd
import pandas as pd
import numpy as np
import io


class HorizontalFlowBarrier(BoundaryCondition):
    """
    Horizontal Flow Barrier (HFB) package

    Input to the Horizontal Flow Barrier (HFB) Package is read from the file that has type “HFB6” in the
    Name File.
    Only one HFB Package can be specified for a GWF model.
    https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.2.2.pdf

    Parameters
    ----------
    geometry: object array of shapely LineStrings
        geometry of barriers, should be lines
    resistance: Optional, float or list of floats
        resistance of the barrier [T]. This equals the inverse of the
        hydraulic characteristic which is used traditionally in modflow6.
    multiplier: Optional, float or list of floats
        multiplier to the conductance between the two model cells specified as containing the barrier.
        This is an alternative to resistance, and you cannot specify both.
    layer: Optional, int
        layer where barrier is located. If None, barrier will be broadcasted to all layers.
        Defaults to None.

    """

    _pkg_id = "hfb"
    _keyword_map = {}
    _period_data = ("k1", "i1", "j1", "k2", "i2", "j2", "hc")
    _template = BoundaryCondition._initialize_template(_pkg_id)

    def __init__(self, idomain, geometry, resistance=None, multiplier=None, layer=None):
        super(__class__, self).__init__()

        if not ((resistance is None) ^ (multiplier is None)):
            raise ValueError("Specify either a resistance or a multiplier")

        self.original_dataset = gpd.GeoDataFrame()
        self.original_dataset["resistance"] = resistance
        self.original_dataset["multiplier"] = multiplier
        self.original_dataset["layer"] = layer
        self.original_dataset["geometry"] = geometry

        # Snap lines to cell edges
        self.dataset = self._snap_to_cell_edges(self.original_dataset, idomain)
        self.dataset["hc"] = self._compute_hydraulic_characteristic(self.dataset)
        self.dataset = self._prepare_layers(self.dataset, idomain)
        self.dataset = self.dataset.to_xarray()

        self._pkgcheck()

    def _prepare_layers(self, gdf, idomain):
        gdf = self._expand_layers(gdf, idomain)
        gdf = gdf.rename(columns={"layer": "k1"})
        gdf["k2"] = gdf["k1"]
        return gdf

    def _expand_layers(self, gdf, idomain):
        # TODO: Right naming for method?
        """Add nlay rows for each segment havinng no layer specified."""

        df_no_layer = gdf.loc[gdf["layer"].values == None]
        df_layer = gdf.loc[gdf["layer"].values != None]

        n_segments = len(df_no_layer.index)

        if "layer" in idomain.dims:
            nlay = len(idomain["layer"])
        else:
            nlay = 1

        expanded_gdf = gpd.GeoDataFrame(np.repeat(df_no_layer.values, nlay, axis=0))
        expanded_gdf.columns = df_no_layer.columns
        expanded_gdf["layer"] = (
            np.tile(np.arange(nlay), n_segments) + 1
        )  # Modflow6 is 1-based

        return pd.concat([df_layer, expanded_gdf])

    def _compute_hydraulic_characteristic(self, df):
        # Currently we do not allow users to specify mixed multipliers and resistances
        # So if first element is None, everything should be None
        # (otherwise could not get pass check during initialization)
        use_multiplier = df["multiplier"].iloc[0] is not None

        if use_multiplier:
            hc = df["multiplier"] * -1
        else:
            hc = 1 / df["resistance"]

        return hc.astype(np.float64)

    def _get_rows_and_columns(self, cellids, grid):
        """Compute rows and columns from cellids"""

        width = len(grid.x)

        rows = np.floor_divide(cellids, width) + 1  # Modflow6 is 1-based
        columns = cellids % width + 1  # Modflow6 is 1-based

        return rows, columns

    def _snap_to_cell_edges(self, lines, idomain):
        """
        Snap geometry to cell edges and get edges
        """

        if "layer" in idomain.dims:
            grid = idomain.isel(layer=0)
        else:
            grid = idomain

        cell_to_cell, snapped_gdf = snap_to_grid(lines, grid, return_geometry=True)

        rows, columns = self._get_rows_and_columns(cell_to_cell, grid)

        snapped_gdf["i1"] = rows[:, 0]
        snapped_gdf["i2"] = rows[:, 1]
        snapped_gdf["j1"] = columns[:, 0]
        snapped_gdf["j2"] = columns[:, 1]

        return snapped_gdf

    def render(self, directory, pkgname, globaltimes):
        """Render fills in the template only, doesn't write binary data"""
        d = {}

        # period = {1: f"{directory}/{self._pkg_id}-{i}.bin"}

        bin_ds = self.dataset[list(self._period_data)]

        sparse_data = self.to_sparse(self.dataset.to_dataframe())
        d["periods"] = {1: self.listarr_to_string(sparse_data)}
        # construct the rest (dict for render)
        d = self.get_options(d)

        d["maxbound"] = self._max_active_n()

        return self._template.render(d)

    def _get_field_spec_from_dtype(self, listarr):
        """
        From https://stackoverflow.com/questions/21777125/how-to-output-dtype-to-a-list-or-dict
        """
        return [
            (x, y[0])
            for x, y in sorted(listarr.dtype.fields.items(), key=lambda k: k[1])
        ]

    def listarr_to_string(self, sparse_data):
        """
        based on write_file for Advanced Stress Packages
        """

        s = io.StringIO()

        textformat = self.get_textformat(sparse_data)
        np.savetxt(s, sparse_data, delimiter=" ", fmt=textformat)

        return s.getvalue()

    def get_textformat(self, sparse_data):
        field_spec = self._get_field_spec_from_dtype(sparse_data)
        dtypes = list(zip(*field_spec))[1]
        textformat = []
        for dtype in dtypes:
            if np.issubdtype(dtype, np.integer):  # integer
                textformat.append("%4d")
            elif np.issubdtype(dtype, np.inexact):  # floatish
                textformat.append("%6.3f")
            else:
                raise ValueError(
                    "Data should be a subdatatype of either 'np.integer' or 'np.inexact'"
                )
        textformat = " ".join(textformat)
        return textformat

    def to_sparse(self, gdf):
        """Convert pandas table to numpy sparse table"""
        sparse_dtype = [
            ("k1", np.int32),
            ("i1", np.int32),
            ("j1", np.int32),
            ("k2", np.int32),
            ("i2", np.int32),
            ("j2", np.int32),
            ("hc", np.float64),
        ]
        colnames = list(list(zip(*sparse_dtype))[0])

        nrow = len(gdf.index)
        listarr = np.empty(nrow, dtype=sparse_dtype)
        for colname in colnames:
            listarr[colname] = gdf[colname]

        return listarr

    def write_datafile(self, outpath, ds):
        """
        Writes a modflow6 binary data file
        """
        # sparse_data = self.to_sparse(self.dataset.to_dataframe())
        # outpath.parent.mkdir(exist_ok=True, parents=True)

        # self._write_file(outpath, sparse_data)
        pass  # Do not write file, data already in render

    def _pkgcheck(self):
        # TODO: Check if not multiple HFB packages are specified.
        pass
