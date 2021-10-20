import jinja2
import numpy as np
import scipy.ndimage.morphology
import xarray as xr

from imod import util
from imod.wq.pkgbase import Package


class BasicFlow(Package):
    """
    The Basic package is used to specify certain data used in all models.
    These include:

    1. the locations of acitve, inactive, and specified head in cells,
    2. the head stored in inactive cells,
    3. the initial head in all cells, and
    4. the top and bottom of the aquifer

    The number of layers (NLAY) is automatically calculated using the IBOUND.
    Thickness is calculated using the specified tops en bottoms.
    The Basic package input file is required in all models.

    Parameters
    ----------
    ibound: xr.DataArray of integers
        is the boundary variable.
        If IBOUND(J,I,K) < 0, cell J,I,K has a constant head.
        If IBOUND(J,I,K) = 0, cell J,I,K is inactive.
        If IBOUND(J,I,K) > 0, cell J,I,K is active.
    top: float or xr.DataArray of floats
        is the top elevation of layer 1. For the common situation in which the
        top layer represents a water-table aquifer, it may be reasonable to set
        `top` equal to land-surface elevation.
    bottom: xr.DataArray of floats
        is the bottom elevation of model layers or Quasi-3d confining beds. The
        DataArray should at least include the `layer` dimension.
    starting_head: float or xr.DataArray of floats
        is initial (starting) head—that is, head at the beginning of the
        simulation (STRT). starting_head must be specified for all simulations,
        including steady-state simulations. One value is read for every model
        cell. Usually, these values are read a layer at a time.
    inactive_head: float, optional
        is the value of head to be assigned to all inactive (no flow) cells
        (IBOUND = 0) throughout the simulation (HNOFLO). Because head at
        inactive cells is unused in model calculations, this does not affect
        model results but serves to identify inactive cells when head is
        printed. This value is also used as drawdown at inactive cells if the
        drawdown option is used. Even if the user does not anticipate having
        inactive cells, a value for inactive_head must be entered. Default
        value is 1.0e30.
    """

    __slots__ = ("ibound", "top", "bottom", "starting_head", "inactive_head")
    _pkg_id = "bas6"
    _template = jinja2.Template(
        "[bas6]\n"
        "    {%- for layer, value in ibound.items() %}\n"
        "    ibound_l{{layer}} = {{value}}\n"
        "    {%- endfor %}\n"
        "    hnoflo = {{inactive_head}}\n"
        "    {%- for layer, value in starting_head.items() %}\n"
        "    strt_l{{layer}} = {{value}}\n"
        "    {%- endfor -%}"
    )

    def __init__(self, ibound, top, bottom, starting_head, inactive_head=1.0e30):
        self._check_ibound(ibound)
        super(__class__, self).__init__()
        self["ibound"] = ibound
        self["top"] = top
        self["bottom"] = bottom
        self["starting_head"] = starting_head
        self["inactive_head"] = inactive_head

    def _check_ibound(self, ibound):
        if not isinstance(ibound, xr.DataArray):
            raise TypeError("ibound must be xarray.DataArray")
        dims = ibound.dims
        if not (dims == ("layer", "y", "x") or dims == ("z", "y", "x")):
            raise ValueError(
                f'ibound dimensions must be ("layer", "y", "x") or ("z", "y", "x"),'
                f" got instead {dims}"
            )

    def _render(self, directory, nlayer, *args, **kwargs):
        """
        Renders part of runfile that ends up under [bas] section.
        """
        d = {}
        for varname in ("ibound", "starting_head"):
            d[varname] = self._compose_values_layer(varname, directory, nlayer)
        d["inactive_head"] = self["inactive_head"].values

        return self._template.render(d)

    def _compose_top(self, directory):
        """
        Composes paths to file, or gets the appropriate scalar value for
        a top of model domain.

        Parameters
        ----------
        directory : str
        """
        da = self["top"]
        if "x" in da.coords and "y" in da.coords:
            if not len(da.shape) == 2:
                raise ValueError("Top should either be 2d or a scalar value")
            d = {}
            d["name"] = "top"
            d["directory"] = directory
            d["extension"] = ".idf"
            value = self._compose(d)
        else:
            if not da.shape == ():
                raise ValueError("Top should either be 2d or a scalar value")
            value = float(da)
        return value

    @staticmethod
    def _cellsizes(dx):
        ncell = dx.size
        index_ends = np.argwhere(np.diff(dx) != 0.0) + 1
        index_ends = np.append(index_ends, ncell)
        index_starts = np.insert(index_ends[:-1], 0, 0) + 1

        d = {}
        for s, e in zip(index_starts, index_ends):
            value = abs(float(dx[s - 1]))
            if s == e:
                d[f"{s}"] = value
            else:
                d[f"{s}:{e}"] = value
        return d

    def _render_dis(self, directory, nlayer):
        """
        Renders part of runfile that ends up under [dis] section.
        """
        d = {}
        d["top"] = self._compose_top(directory)
        d["bottom"] = self._compose_values_layer("bottom", directory, nlayer)
        d["nlay"], d["nrow"], d["ncol"] = self["ibound"].shape
        # TODO: check dx > 0, dy < 0?
        if "dx" not in self or "dy" not in self:  # assume equidistant
            dx, _, _ = util.coord_reference(self["x"])
            dy, _, _ = util.coord_reference(self["y"])
        else:
            dx = self.coords["dx"]
            dy = self.coords["dy"]

        if isinstance(dy, (float, int)) or dy.shape in ((), (1,)):
            d["dy"] = {"?": abs(float(dy))}
        else:
            d["dy"] = self._cellsizes(dy)

        if isinstance(dx, (float, int)) or dx.shape in ((), (1,)):
            d["dx"] = {"?": abs(float(dx))}
        else:
            d["dx"] = self._cellsizes(dx)

        # Non-time dependent part of dis
        # Can be inferred from ibound
        _dis_template = jinja2.Template(
            "[dis]\n"
            "    nlay = {{nlay}}\n"
            "    nrow = {{nrow}}\n"
            "    ncol = {{ncol}}\n"
            "    {%- for row, value in dy.items() %}\n"
            "    delc_r{{row}} = {{value}}\n"
            "    {%- endfor %}\n"
            "    {%- for col, value in dx.items() %}\n"
            "    delr_c{{col}} = {{value}}\n"
            "    {%- endfor %}\n"
            "    top = {{top}}\n"
            "    {%- for layer, value in bottom.items() %}\n"
            "    botm_l{{layer}} = {{value}}\n"
            "    {%- endfor %}\n"
            "    laycbd_l? = 0"
        )

        return _dis_template.render(d)

    def thickness(self):
        """
        Computes layer thickness from top and bottom data.

        Returns
        -------
        thickness : xr.DataArray
        """
        th = xr.concat(
            [
                self["top"] - self["bottom"].sel(layer=1),
                -1.0 * self["bottom"].diff("layer"),
            ],
            dim="layer",
        )
        return th

    def _pkgcheck(self, ibound=None):
        if (self["top"] < self["bottom"]).any():
            raise ValueError(f"top should be larger than bottom in {self}")

        active_cells = self["ibound"] != 0
        if (active_cells & np.isnan(self["starting_head"])).any():
            raise ValueError(
                f"Active cells in ibound may not have a nan value in starting_head in {self}"
            )

        _, nlabels = scipy.ndimage.label(active_cells.values)
        if nlabels > 1:
            raise ValueError(
                f"{nlabels} disconnected model domain detected in the ibound in {self}"
            )
