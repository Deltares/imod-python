from imod.typing import DropVarsType, GridDataArray, Imod5DataDict, SelSettingsType

_DROP_LAYER_KWARGS: SelSettingsType = {
    "layer": 0,
    "drop": True,
    "missing_dims": "ignore",
}

_DROP_VARS_KWARGS: DropVarsType = {
    "names": "layer",
    "errors": "ignore",
}


def _drop_sel_and_drop_vars(
    da: GridDataArray,
) -> GridDataArray:
    """
    Drop the layer dimension and the layer coord if present from the given
    DataArray.
    """
    return da.isel(**_DROP_LAYER_KWARGS).compute().drop_vars(**_DROP_VARS_KWARGS)


def drop_layer_dim_cap_data(imod5_data: Imod5DataDict) -> Imod5DataDict:
    cap_data = imod5_data["cap"]
    return {"cap": {key: _drop_sel_and_drop_vars(da) for key, da in cap_data.items()}}
