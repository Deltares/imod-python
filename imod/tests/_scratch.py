# %%
from pathlib import Path

import xarray as xr

import imod
from imod.util.dims import drop_layer_dim_cap_and_bnd_data

# %%
# Path management
wdir = Path(
    r"c:\Users\engelen\projects_wdir\imod-python\imod5_converter\NHI_sprint\Peelvenen"
)

svat_dir = wdir / "CONVERSION_IMOD5_IPF" / "GWF_1" / "MSWAPINPUT"

prj_path = wdir / "prjfiles" / "Peelvenen_absolute_paths_sprinkling_ipf.PRJ"

svat_imod5 = imod.idf.open_dataset(svat_dir / "SVAT*.IDF", pattern="{name}")
svat_imod5 = xr.merge([svat_imod5]).to_dataarray(dim="subunit", name="svat").compute()
svat_imod5 = svat_imod5.assign_coords(subunit=[0, 1])
# %%
# Setup to get example args for _render() of SprinklingPoints
prj_data, period_data = imod.formats.prj.open_projectfile_data(prj_path)

dis_pkg = imod.mf6.StructuredDiscretization.from_imod5_data(prj_data, validate=False)
dis_pkg["idomain"] = dis_pkg["idomain"].clip(min=0)
npf_pkg = imod.mf6.NodePropertyFlow.from_imod5_data(
    prj_data, dis_pkg.dataset["idomain"]
)

prj_data = drop_layer_dim_cap_and_bnd_data(prj_data)
griddata, msw_active = imod.msw.GridData.from_imod5_data(prj_data, dis_pkg)

well = imod.mf6.LayeredWell.from_imod5_cap_data(
    prj_data, target_dis=None, regridder_types=None, regrid_cache=None
)
# Convert to args of Sprinkling._render()
mf6_well = well.to_mf6_pkg(
    dis_pkg["idomain"], dis_pkg["top"], dis_pkg["bottom"], npf_pkg["k"]
)
isactive_1d, svat = griddata.generate_isactive_svat_arrays()

sprinkling_points = imod.msw.SprinklingPoints.from_imod5_data(prj_data)

# %%
directory = r"c:\Users\engelen\projects_wdir\imod-python\imod5_converter\NHI_sprint\Peelvenen\conversion_output_ipf"
sprinkling_points.write(directory, isactive_1d, svat, dis_pkg, mf6_well)

# %%
#
# TODO: Verify if iMOD5 SVAT grid the same as the one derived in this script.
