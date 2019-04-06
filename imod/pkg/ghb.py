import jinja2

from imod.pkg.pkgbase import Package

#class GeneralHeadBoundaryGroup(object):
    # Does a groupby over packages of the same kind when writing
    # Collects total data of all same kind packages
    # adds a system number
    # This one is actually in charge of generating the output from
    # the dictionaries provided by the ._compose_values methods
    # Every system is treated independently


class GeneralHeadBoundary(Package):
    _mapping = (
        ("bhead", "head"),
        ("cond", "cond"),
        ("ghbssmdens", "dens"),
    )

    _template = jinja2.Template(
    """
       {%- for name, dictname in mapping -%}
            {%- for time, layerdict in dicts[dictname].items() %}
                {%- set time_index = loop.index0 %}
                {%- for layer, value in layerdict.items() %}
        {{name}}_p{{time_index + 1}}_s{{system_index}}_l{{layer}} = {{value}}
                {%- endfor -%}
            {%- endfor -%}
        {%- endfor -%}
    """
    )
    def __init__(self, head, cond, conc, dens):
        super(__class__, self).__init__()
        self["head"] = head
        self["cond"] = cond
        self["conc"] = conc
        self["dens"] = dens
