from imod.flow import MetaSwap
import pathlib
import pytest
import os
import textwrap
from copy import deepcopy


def test_metaswap_render(metaswap_dict, get_render_dict):
    # See e.g. https://github.com/omarkohl/pytest-datafiles/issues/6
    directory = str(pathlib.Path(".").resolve())

    metaswap = MetaSwap(**metaswap_dict)

    # Force extra paths as well
    metaswap.extra_files = [
        os.path.join(directory, file) for file in metaswap.extra_files
    ]

    nlayer = 1
    to_render = get_render_dict(metaswap, directory, None, nlayer)
    to_render["n_entry"] = nlayer

    compare = textwrap.dedent(
        f"""\
        0001, (cap), 1, MetaSwap, ['boundary', 'landuse', 'rootzone_thickness', 'soil_physical_unit', 'meteo_station_number', 'surface_elevation', 'sprinkling_type', 'sprinkling_layer', 'sprinkling_capacity', 'wetted_area', 'urban_area', 'ponding_depth_urban', 'ponding_depth_rural', 'runoff_resistance_urban', 'runoff_resistance_rural', 'runon_resistance_urban', 'runon_resistance_rural', 'infiltration_capacity_urban', 'infiltration_capacity_rural', 'perched_water_table', 'soil_moisture_factor', 'conductivity_factor']
        022, 001
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}boundary_l1.idf
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}landuse_l1.idf
        1, 1, 001, 1.000, 0.000, 1.2, ""
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}soil_physical_unit_l1.idf
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}meteo_station_number_l1.idf
        1, 1, 001, 1.000, 0.000, 0.0, ""
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}sprinkling_type_l1.idf
        1, 2, 001, 1.000, 0.000, -9999., {directory}{os.sep}sprinkling_layer_l1.idf
        1, 1, 001, 1.000, 0.000, 1000.0, ""
        1, 1, 001, 1.000, 0.000, 30.0, ""
        1, 1, 001, 1.000, 0.000, 30.0, ""
        1, 1, 001, 1.000, 0.000, 0.02, ""
        1, 1, 001, 1.000, 0.000, 0.005, ""
        1, 1, 001, 1.000, 0.000, 1.5, ""
        1, 1, 001, 1.000, 0.000, 1.5, ""
        1, 1, 001, 1.000, 0.000, 1.5, ""
        1, 1, 001, 1.000, 0.000, 1.5, ""
        1, 1, 001, 1.000, 0.000, 10.0, ""
        1, 1, 001, 1.000, 0.000, 2.0, ""
        1, 1, 001, 1.000, 0.000, 0.5, ""
        1, 1, 001, 1.000, 0.000, 1.0, ""
        1, 1, 001, 1.000, 0.000, 1.0, ""
        008,extra files
        {directory}{os.sep}fact_svat.inp
        {directory}{os.sep}luse_svat.inp
        {directory}{os.sep}mete_grid.inp
        {directory}{os.sep}para_sim.inp
        {directory}{os.sep}tiop_sim.inp
        {directory}{os.sep}init_svat.inp
        {directory}{os.sep}comp_post.inp
        {directory}{os.sep}sel_key_svat_per.inp"""
    )
    rendered = metaswap._render_projectfile(**to_render)
    assert rendered == compare


def test_metaswap_pkgcheck(metaswap_dict):
    metaswap = MetaSwap(**metaswap_dict)
    # Test if no ValueError is thrown, source:
    # https://miguendes.me/how-to-check-if-an-exception-is-raised-or-not-with-pytest
    try:
        metaswap._pkgcheck()
    except ValueError as exc:
        assert False, f"'metaswap._pkgcheck()' raised an exception {exc}"


def test_metaswap_pkgcheck_fail(metaswap_dict):
    metaswap_fail = MetaSwap(**metaswap_dict)

    metaswap_fail.dataset = metaswap_fail.dataset.expand_dims("layer")

    with pytest.raises(ValueError):
        metaswap_fail._pkgcheck()


def test_check_extra_files_no_files(metaswap_dict):
    metaswap_fail = MetaSwap(**metaswap_dict)

    metaswap_fail.extra_files = []

    with pytest.raises(ValueError):
        metaswap_fail.check_extra_files()


def test_check_extra_files_missing_files(metaswap_dict):
    metaswap_fail = MetaSwap(**metaswap_dict)

    metaswap_fail.extra_files = [
        "fact_svat.inp",
        "luse_svat.inp",
    ]

    with pytest.raises(ValueError):
        metaswap_fail.check_extra_files()
