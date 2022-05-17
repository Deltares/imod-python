import xarray as xr

from imod.flow import OutputControl


def oc_configuration_template():
    return dict(
        saveshd=0,
        saveflx=0,
        saveghb=0,
        savedrn=0,
        savewel=0,
        saveriv=0,
        saverch=0,
        saveevt=0,
        savehfb=0,
    )


def test_compose_oc_configuration_no_layers():
    nlayer = 3
    oc = OutputControl()

    oc_configuration = oc._compose_oc_configuration(nlayer)

    expected = oc_configuration_template()

    assert oc_configuration == expected


def test_compose_oc_configuration_all_layers():
    nlayer = 3
    oc = OutputControl(save_head=-1)

    oc_configuration = oc._compose_oc_configuration(nlayer)

    expected = oc_configuration_template()
    expected["saveshd"] = -1

    assert oc_configuration == expected


def test_compose_oc_configuration_layers():
    nlayer = 3
    layers = xr.DataArray([1, 3], coords={"layer": [1, 3]}, dims=("layer",))
    oc = OutputControl(save_flux=layers)

    oc_configuration = oc._compose_oc_configuration(nlayer)

    expected = oc_configuration_template()
    expected["saveflx"] = "1,3"

    assert oc_configuration == expected
