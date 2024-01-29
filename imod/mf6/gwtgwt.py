

class GWTGWT(ExchangeBase):
    """
    This package is for writing an exchange file, used for splitting up a model
    into different submodels (that can be solved in parallel). It (usually)
    is not instantiated by users, but created by the "split" method of the
    simulation class."""

    def __init__(
        self,
        flow_model_id1: str,
        flow_model_id2: str,
        
        cell_id1: xr.DataArray,
        cell_id2: xr.DataArray,
        layer: xr.DataArray,
        cl1: xr.DataArray,
        cl2: xr.DataArray,
        hwva: xr.DataArray,
        angldegx: Optional[xr.DataArray] = None,
        cdist: Optional[xr.DataArray] = None,
    ):
        super().__init__(locals())
        self.dataset["cell_id1"] = cell_id1
        self.dataset["cell_id2"] = cell_id2
        self.dataset["layer"] = layer
        self.dataset["model_name_1"] = flow_model_id1

        self.dataset["model_name_2"] = flow_model_id1

        self.dataset["ihc"] = xr.DataArray(np.ones_like(cl1, dtype=int))
        self.dataset["cl1"] = cl1
        self.dataset["cl2"] = cl2
        self.dataset["hwva"] = hwva

        auxiliary_variables = [var for var in [angldegx, cdist] if var is not None]
        if auxiliary_variables:
            self.dataset["auxiliary_data"] = xr.merge(auxiliary_variables).to_array(
                name="auxiliary_data"
            )
            expand_transient_auxiliary_variables(self)    

    def set_options(
        self,
        print_input: Optional[bool] = None,
        print_flows: Optional[bool] = None,
        save_flows: Optional[bool] = None,
        adv_scheme: Optional[str] = None,
        dsp_xt3d_off: Optional[bool] = None,
        dsp_xt3d_rhs: Optional[bool] = None,
    ):            
        self.dataset["print_input"] = print_input
        self.dataset["print_flows"] = print_flows
        self.dataset["save_flows"] = save_flows
        self.dataset["adv_scheme"] = adv_scheme
        self.dataset["dsp_xt3d_off"] = dsp_xt3d_off
        self.dataset["dsp_xt3d_rhs"] = dsp_xt3d_rhs
