import jinja2

from imod.pkg.pkgbase import Package

class OutputControl(Package):
    _pkg_id = "occ"
    _template = (
    """
    [oc]
        savehead_p?_l? = {save_head_idf}
        saveconclayer_p?_l? = {save_concentration_idf}
        savebudget_p?_l? = {save_budget_idf}
        saveheadtec_p?_l? = {save_head_tec}
        saveconctec_p?_l? = {save_concentration_tec}
        savevxtec_p?_l? = {save_budget_tec}
        savevytec_p?_l? = {save_budget_tec}
        savevztec_p?_l? = {save_budget_tec}
        saveheadvtk_p?_l? = {save_head_vtk}
        saveconcvtk_p?_l? = {save_concentration_vtk}
        savevelovtk_p?_l? = {save_budget_vtk}
    """
    )

    def __init__(
        self,
        save_head_idf=False,
        save_concentration_idf=False,
        save_budget_idf=False,
        save_head_tec=False,
        save_concentration_tec=False,
        save_budget_tec=False,
        save_head_vtk=False,
        save_concentration_vtk=False,
        save_budget_vtk=False,
    ):
        super(__class__, self).__init__()
        self["save_head_idf"] = save_head_idf
        self["save_concentration_idf"] = save_concentration_idf
        self["save_budget_idf"] = save_budget_idf
        self["save_head_tec"] = save_budget_tec
        self["save_concentration_tec"] = save_concentration_tec
        self["save_budget_tec"] = save_budget_tec
        self["psave_head_vtk"] = save_head_vtk
        self["save_concentration_vtk"] = save_concentration_vtk
        self["save_budget_vtk"] = save_budget_vtk
