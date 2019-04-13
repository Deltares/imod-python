import jinja2
import xarray as xr

from imod.pkg.pkgbase import Package
from imod.io import util


class BasicFlow(Package):
    _pkg_id = "bas"
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

    # Non-time dependent part of dis
    # Can be inferred from ibound
    _dis_template = jinja2.Template(
        "[dis]\n"
        "    nlay = {{nlay}}\n"
        "    nrow = {{nrow}}\n"
        "    ncol = {{ncol}}\n"
        "    delc_r? = {{dy}}\n"
        "    delr_c? = {{dx}}\n"
        "    top = {{top}}\n"
        "    {%- for layer, value in bottom.items() %}\n"
        "    botm_l{{layer}} = {{value}}\n"
        "    {%- endfor %}\n"
    )

    def __init__(
        self,
        ibound,
        top,
        bottom,
        starting_head,
        inactive_head=1.0e30,
        confining_bed_below=0,
    ):
        self._check_ibound(ibound)
        super(__class__, self).__init__()
        self["ibound"] = ibound
        self["top"] = top
        self["bottom"] = bottom
        self["starting_head"] = starting_head
        self["inactive_head"] = inactive_head
        self["confining_bed_below"] = confining_bed_below
        # TODO: create dx, dy if they don't exist?

    def _check_ibound(self, ibound):
        if not isinstance(ibound, xr.DataArray):
            raise ValueError
        if not len(ibound.shape) == 3:
            raise ValueError

    def _render_bas(self, directory):
        d = {}
        for varname in ("ibound", "starting_head"):
            d[varname] = self._compose_values_layer(varname, directory)
        d["inactive_head"] = self["inactive_head"].values

        return self._template.render(d)

    def _compose_top(self, directory):
        da = self["top"]
        if "x" in da.coords and "y" in da.coords:
            if not len(da.shape) == 2:
                raise ValueError("Top should either be 2d or a scalar value")
            d = {}
            d["name"] = "top"
            d["directory"] = directory
            d["extension"] = ".idf"
            value = util.compose(d)
        else:
            if not da.shape == ():
                raise ValueError("Top should either be 2d or a scalar value")
            value = float(da)
        return value

    def _render_dis(self, directory):
        d = {}
        d["top"] = self._compose_top(directory)
        d["bottom"] = self._compose_values_layer("bottom", directory)
        d["nlay"], d["nrow"], d["ncol"] = self["ibound"].shape
        # TODO: check dx > 0, dy < 0?
        d["dx"] = abs(float(self.coords["dx"]))
        d["dy"] = abs(float(self.coords["dy"]))

        return self._dis_template.render(d)

    def thickness(self):
        """
        Computes layer thickness from top and bottom data.
        """
        th = xr.concat(
            [
                self["top"] - self["bottom"].sel(layer=1),
                -1.0 * self["bottom"].diff("layer"),
            ],
            dim="layer",
        )
        return th


