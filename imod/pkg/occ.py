import jinja2

class OutputControl(xr.Dataset):
    _pkg_id = "occ"
    _template = (
    """
    [oc]
        savehead_p?_l? = {save_head_idf}
        saveconclayer_p?_l? = {save_conc_idf}
        savebudget_p?_l? = {save_budget_idf}
        saveheadtec_p?_l? = {save_head_tec}
        saveconctec_p?_l? = {save_conc_tec}
        savevxtec_p?_l? = {save_budget_tec}
        savevytec_p?_l? = {save_budget_tec}
        savevztec_p?_l? = {save_budget_tec}
        saveheadvtk_p?_l? = {save_head_vtk}
        saveconcvtk_p?_l? = {save_conc_vtk}
        savevelovtk_p?_l? = {save_budget_vtk}
    """
    )

    def __init__(
        self,
        save_head_idf=False,
        save_conc_idf=False,
        save_budget_idf=False,
        save_head_tec=False,
        save_conc_tec=False,
        save_budget_tec=False,
        save_head_vtk=False,
        save_conc_vtk=False,
        save_budget_vtk=False,
    ):
        super(__class__, self).__init__()
        self["save_head_idf"] = save_head_idf
        self["save_conc_idf"] = save_conc_idf
        self["save_budget_idf"] = save_budget_idf
        self["save_head_tec"] = save_budget_tec
        self["save_conc_tec"] = save_conc_tec
        self["save_budget_tec"] = save_budget_tec
        self["psave_head_vtk"] = save_head_vtk
        self["save_conc_vtk"] = save_conc_vtk
        self["save_budget_vtk"] = save_budget_vtk
