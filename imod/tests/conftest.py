import pytest

from .fixtures.backward_compatibility_fixture import (
    imod5_dataset,
    imod5_dataset_periods,
    imod5_dataset_transient,
)
from .fixtures.flow_basic_fixture import (
    basic_dis,
    basic_dis__topsystem,
    get_render_dict,
    horizontal_flow_barrier_gdf,
    metaswap_dict,
    parameterizable_basic_dis,
    three_days,
    two_days,
    well_df,
)
from .fixtures.flow_basic_unstructured_fixture import (
    basic_disv__topsystem,
    basic_unstructured_dis,
    circle_dis,
)
from .fixtures.flow_transport_simulation_fixture import flow_transport_simulation
from .fixtures.imod5_cap_data import (
    cap_data_sprinkling_grid,
    cap_data_sprinkling_grid__big,
    cap_data_sprinkling_points,
)
from .fixtures.imod5_well_data import (
    well_duplication_import_prj,
    well_mixed_ipfs,
    well_out_of_bounds_ipfs,
    well_regular_import_prj,
    well_simple_import_prj__steady_state,
    well_simple_import_prj__transient,
)
from .fixtures.mf6_circle_fixture import (
    circle_model,
    circle_model_evt,
    circle_model_transport,
    circle_model_transport_multispecies_variable_density,
    circle_partitioned,
    circle_result,
    circle_result__offset_origins,
    circle_result_evt,
    circle_result_sto,
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
from .fixtures.mf6_rectangle_with_lakes import rectangle_with_lakes
from .fixtures.mf6_small_models_fixture import (
    solution_settings,
    structured_flow_model,
    structured_flow_simulation,
    structured_flow_simulation_2_flow_models,
    unstructured_flow_model,
    unstructured_flow_simulation,
)
from .fixtures.mf6_twri_disv_fixture import twri_disv_model
from .fixtures.mf6_twri_fixture import (
    split_transient_twri_model,
    transient_twri_model,
    transient_twri_result,
    transient_unconfined_twri_model,
    transient_unconfined_twri_result,
    twri_model,
    twri_model_hfb,
    twri_result,
    twri_result_9_drn_in_1_cell,
)
from .fixtures.mf6_welltest_fixture import (
    mf6wel_test_data_stationary,
    mf6wel_test_data_transient,
    well_high_lvl_test_data_stationary,
    well_high_lvl_test_data_transient,
    well_test_data_stationary,
    well_test_data_transient,
)
from .fixtures.msw_fixture import fixed_format_parser, simple_2d_grid_with_subunits
from .fixtures.msw_imod5_cap_fixture import imod5_cap_data
from .fixtures.msw_meteo_fixture import meteo_grids
from .fixtures.msw_model_fixture import coupled_mf6_model, coupled_mf6wel, msw_model
