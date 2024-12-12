from imod.typing import Imod5DataDict, SelSettingsType

_DROP_LAYER_KWARGS: SelSettingsType = {
    "layer": 0,
    "drop": True,
    "missing_dims": "ignore",
}


def drop_layer_dim_cap_data(imod5_data: Imod5DataDict) -> Imod5DataDict:
    cap_data = imod5_data["cap"]
    return {
        "cap": {
            key: da.isel(**_DROP_LAYER_KWARGS).compute() for key, da in cap_data.items()
        }
    }
