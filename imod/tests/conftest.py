import pytest

from .fixtures.flow_basic_fixture import (
    basic_dis,
    get_render_dict,
    horizontal_flow_barrier_gdf,
    metaswap_dict,
    parameterizable_basic_dis,
    three_days,
    two_days,
    well_df,
)
from .fixtures.flow_basic_unstructured_fixture import (
    basic_unstructured_dis,
    circle_dis,
)
from .fixtures.flow_example_fixture import imodflow_model
from .fixtures.mf6_circle_fixture import (
    circle_model,
    circle_model_evt,
    circle_result,
    circle_result_evt,
)
from .fixtures.mf6_flow_with_transport_fixture import (
    bulk_density_fc,
    concentration_fc,
    conductance_fc,
    decay_fc,
    decay_sorbed_fc,
    distcoef_fc,
    elevation_fc,
    flow_model_with_concentration,
    head_fc,
    porosity_fc,
    proportion_depth_fc,
    proportion_rate_fc,
    rate_fc,
    sp2_fc,
)
from .fixtures.mf6_lake_package_fixture import (
    ijsselmeer,
    lake_package,
    lake_table,
    naardermeer,
)
from .fixtures.mf6_regridding_fixture import (
    solution_settings,
    structured_flow_model,
    structured_flow_simulation,
    unstructured_flow_model,
    unstructured_flow_simulation,
)
from .fixtures.mf6_twri_disv_fixture import twri_disv_model
from .fixtures.mf6_twri_fixture import (
    transient_twri_model,
    transient_twri_result,
    twri_model,
    twri_result,
)
from .fixtures.mf6_welltest_fixture import (
    mf6wel_test_data_stationary,
    mf6wel_test_data_transient,
    well_high_lvl_test_data_stationary,
    well_high_lvl_test_data_transient,
    well_test_data_stationary,
    well_test_data_transient,
)
from .fixtures.msw_fixture import fixed_format_parser
from .fixtures.msw_model_fixture import coupled_mf6_model, msw_model
