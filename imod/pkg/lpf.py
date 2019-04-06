class LayerPropertyFlow(xr.Dataset):
    def __init__(
        self,
        k_horizontal,
        k_vertical,
        horizontal_anistropy=1.0,
        interblock="harmonic",
        layer_type=0,
        specific_storage,
        specific_yield,
        save_budget=False,
        layer_wet,
        interval_wet,
        method_wet,
        head_dry=1.0e20,
    ):
        super(__class__, self).__init__()