"""
Contains an imod model object
"""
import jinja2

# This class allows only imod packages as values
class Model(dict):
    # These templates end up here since they require global information
    # from more than one package
    _gen_template = jinja2.Template(
    """
    [gen]
        modelname = {{modelname}}                          
        writehelp = {{writehelp}} 
        result_dir = {{modelname}} 
        packages = dis, bas6, btn, {{package_list|join(", ")}}
        coord_xll = {{xmin}}
        coord_yll = {{ymin}}
        start_year = {{start_date[:4]}}
        start_month = {{start_date[4:6]}}
        start_day = {{start_date[6:8]}}
        runtype = SEAWAT
    """

    # Create dis as an object with derived values
    _dis_template = jinja2.Template(
    """
        nper = {{nper}}
        {%- for period_duration in time_discretisation.values() %}
        {%- set time_index = loop.index %}
        perlen_p{{time_index}} = {{period_duration}}
        {%- endfor %}
        nstp_p? = {{nstp}}
        sstr_p? = {{sstr}}
        laycbd_l? = {{laycbd}}
    """
    )

    def __init__(self, ibound):
        dict.__init__(self)
        self["ibound"] = ibound

    def __setitem__(self, key, value):
        if not hasattr(value, "_pkgcheck"):
            raise ValueError("not a package")
        dict.__setitem__(self, key, value)
    
    def update(self, *arg, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v
    
    def render(self):
        for pkgname, pkg in self.items():
            if issubclasstype(pkg), imod.pkg.pkgbase.Package):
                pass
        # Create groups for chd, drn, ghb, riv, wel
        # super()
