import jinja2

from imod.wq.pkgbase import Package


class Dispersion(Package):
    """
    Solves the concentration change due to dispersion explicitly or formulates
    the coefficient matrix of the dispersion term for the matrix solver.

    Parameters
    ----------
    longitudinal: float
        is the longitudinal dispersivity (AL), for every cell of the model grid
        (unit: L).
    ratio_horizontal: float
        is a 1D real array defining the ratio of the horizontal transverse
        dispersivity (TRPT), to the longitudinal dispersivity. Each value in the
        array corresponds to one model layer. Some recent field studies suggest
        that ratio_horizontal is generally not greater than 0.1.
    ratio_vertical: float
        (TRPV) is the ratio of the vertical transverse dispersivity to the
        longitudinal dispersivity. Each value in the array corresponds to one
        model layer.
        Some recent field studies suggest that ratio_vertical is generally not
        greater than 0.01.
        Set ratio_vertical equal to ratio_horizontal to use the standard
        isotropic dispersion model. Otherwise, the modified isotropic dispersion
        model is used.
    diffusion_coefficient: float
        is the effective molecular diffusion coefficient (unit: L2T-1). Set
        diffusion_coefficient = 0 if the effect of molecular diffusion is
        considered unimportant. Each value in the array corresponds to one model
        layer.
    """
    _pkg_id = "dsp"

    _mapping = (
        ("al", "longitudinal"),
        ("trpt", "ratio_horizontal"),
        ("trpv", "ratio_vertical"),
        ("dmcoef", "diffusion_coefficient"),
    )

    _template = jinja2.Template(
        "[dsp]\n"
        "    {%- for name, dictname in mapping -%}\n"
        "        {%- for layer, value in dicts[dictname].items() %}\n"
        "    {{name}}_l{{layer}} = {{value}}\n"
        "        {%- endfor -%}\n"
        "    {%- endfor -%}\n"
    )

    def __init__(
        self,
        longitudinal,
        ratio_horizontal=0.1,
        ratio_vertical=0.1,
        diffusion_coefficient=1.0e-9,
    ):
        super(__class__, self).__init__()
        self["longitudinal"] = longitudinal
        self["ratio_horizontal"] = ratio_horizontal
        self["ratio_vertical"] = ratio_vertical
        self["diffusion_coefficient"] = diffusion_coefficient

    def _render(self, directory, *args, **kwargs):
        d = {"mapping": self._mapping}
        dicts = {}

        for varname in self.data_vars.keys():
            dicts[varname] = self._compose_values_layer(varname, directory)
        d["dicts"] = dicts

        return self._template.render(d)
