import pytest

from .fixtures.flow_basic_fixture import (
    basic_dis,
    get_render_dict,
    horizontal_flow_barrier_gdf,
    metaswap_dict,
    three_days,
    two_days,
    well_df,
)
from .fixtures.mf6_circle_fixture import circle_model, circle_result
from .fixtures.mf6_flow_with_tranpsort_fixture import (
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
from .fixtures.mf6_twri_disv_fixture import twri_disv_model
from .fixtures.mf6_twri_fixture import transient_twri_model, twri_model, twri_result
from .fixtures.msw_fixture import fixed_format_parser
from .fixtures.msw_model_fixture import coupled_mf6_model, msw_model
