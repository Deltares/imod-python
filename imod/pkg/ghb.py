import jinja2

from imod.pkg.pkgbase import BoundaryCondition

#class GeneralHeadBoundaryGroup(object):
    # Does a groupby over packages of the same kind when writing
    # Collects total data of all same kind packages
    # adds a system number
    # This one is actually in charge of generating the output from
    # the dictionaries provided by the ._compose_values methods
    # Every system is treated independently


class GeneralHeadBoundary(BoundaryCondition):
    _pkg_id = "ghb"
    _mapping = (
        ("bhead", "head"),
        ("cond", "conductance"),
        ("ghbssmdens", "density"),
    )

    _template = jinja2.Template(
    """
       {%- for name, dictname in mapping -%}
            {%- for time, layerdict in dicts[dictname].items() %}
                {%- set time_index = loop.index %}
                {%- for layer, value in layerdict.items() %}
        {{name}}_p{{time_index}}_s{{system_index}}_l{{layer}} = {{value}}
                {%- endfor -%}
            {%- endfor -%}
        {%- endfor -%}
    """
    )
    def __init__(self, head, conductance, concentration, density):
        super(__class__, self).__init__()
        self["head"] = head
        self["conductance"] = conductance
        self["concentration"] = concentration
        self["density"] = density
