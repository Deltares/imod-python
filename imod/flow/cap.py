import os
import pathlib

import jinja2
from imod.flow.pkgbase import Package


class MetaSwap(Package):
    """
    The MetaSWAP package (CAP), provides the input to be converted to a
    MetaSWAP model, which is an external model code used to simulate the
    unsaturated zone.

    Note that only two-dimensional DataArrays with dimensions ``("y", "x")``
    should be supplied to this package.  In the current implementation
    time-related files are provided as external files ("extra files"). Similar
    to the iMODFLOW implementation of the projectfile. For now these need to be
    provided as a path.

    MetaSWAP is developed by Alterra, Wageningen as part of the SIMGRO model
    code. The SIMGRO framework is intended for regions with an undulating
    topography and unconsolidated sediments in the (shallow) subsoil. Both
    shallow and deep groundwater levels can be modelled by MetaSWAP. This
    model is based on a simplification of ‘straight Richards’, meaning that no
    special processes like hysteresis, preferential flow and bypass flow are
    modelled. Snow is not modelled, and neither the influence of frost on the
    soil water conductivity. A perched watertable can be present in the SVAT
    column model, but interflow is not modelled. Processes that are typical for
    steep slopes are not included. The code contains several parameterized
    water management schemes, including irrigation and water level management.

    References:

    * Van Walsum, P. E. V., 2017a. SIMGRO V7.3.3.2, Input and Output reference
      manual. Tech. Rep.  Alterra-Report 913.3, Alterra, Wageningen. 98 pp.

    * Van Walsum, P. E. V., 2017b. SIMGRO V7.3.3.2, Users Guide. Tech. Rep.
      Alterra-Report 913.2, Alterra, Wageningen. 111 pp.

    * Van Walsum, P. E. V. and P. Groenendijk, 2008. “Quasi Steady-State
      Simulation on of the Unsaturated Zone in Groundwater Modeling of Lowland
      Regions.” Vadose Zone Journal 7: 769-778.

    * Van Walsum, P. E. V., A. A. Veldhuizen and P. Groenendijk, 2016. SIMGRO
      V7.2.27, Theory and model implementation. Tech. Rep. Alterra-Report 913.1,
      Alterra, Wageningen. 93 pp 491.

    Parameters
    ----------
    boundary : int or xr.DataArray of ints
        2D boundary, used to specify active MetaSWAP elements, similar to
        ibound in the Boundary package

    landuse : int or xr.DataArray of ints
        Landuse codes, referred to in the lookup table file luse_mswp.inp

    rootzone_thickness : float or xr.DataArray of floats
        Rootzone thickness in cm (min. value is 10 centimeter).

    soil_physical_unit : int or xr.DataArray of ints
        Soil Physical Unit, referred to in the lookup table file fact_mswp.inp.

    meteo_station_number : float or xr.DataArray of ints
        Meteo station number, referred to by mete_mswp.inp.

    surface_elevation : float or xr.DataArray of floats
        Surface Elevation (m+MSL)

    sprinkling_type : int or xr.DataArray of ints
        Sprinkling type ("Artificial Recharge Type" in iMOD manual):

        * 0 = no occurrence
        * 1 = from groundwater
        * 2 = from surface water

    sprinkling_layer : int or xr.DataArray of ints
        Number of modellayer from which water is extracted ("Artificial
                Recharge Location" in iMOD manual)

    sprinkling_capacity : float or xr.DataArray of floats
        Sprinkling capacity (mm/d) sets the maximum amount extracted for
        sprinkling ("Artificial Recharge Capacity" in iMOD manual)

    wetted_area : float or xr.DataArray of floats
        Total area (m2) occupied by surface water elements.  Values will be
        truncated by maximum cellsize.

    urban_area : float or xr.DataArray of floats
        Total area (m2) occupied by urban area.  Values will be truncated by
        maximum cellsize.

    ponding_depth_urban : float or xr.DataArray of floats
        Ponding Depth Urban Area (m), specifying the acceptable depth of the
        ponding of water on the surface in the urban area before surface runoff
        occurs.

    ponding_depth_rural : float or xr.DataArray of floats
        Ponding Depth Rural Area (m), specifying the acceptable depth of the
        ponding of water on the surface in the rural area before surface runoff
        occurs.

    runoff_resistance_urban : float or xr.DataArray of floats
        Runoff Resistance Urban Area (day), specifying the resistance surface
        flow encounters in the urban area. The minimum value is equal to the
        model time period.

    runoff_resistance_rural : float or xr.DataArray of floats
        Runoff Resistance Rural Area (day), specifying the resistance surface
        flow encounters in the rural area. The minimum value is equal to the
        model time period.

    runon_resistance_urban : float or xr.DataArray of floats
        Runon Resistance Urban Area (day), specifying the resistance surface
        flow encounters to a model cell from an adjacent cell in the urban
        area. The minimum value is equal to the model time period.

    runon_resistance_rural : float or xr.DataArray of floats
        Runon Resistance Rural Area (day), specifying the resistance surface
        flow encounters to a model cell from an adjacent cell in the rural
        area. The minimum value is equal to the model time period.

    infiltration_capacity_urban : float or xr.DataArray of floats
        the infiltration capacity (m/d) of the soil surface in the urban area.
        The range is 0-1000 m/d. The NoDataValue -9999 indicates unlimited
        infiltration is possible.

    infiltration_capacity_rural : float or xr.DataArray of floats
        the infiltration capacity (m/d) of the soil surface in the urban area.
        The range is 0-1000 m/d. The NoDataValue -9999 indicates unlimited
        infiltration is possible.

    perched_water_table : float or xr.DataArray of floats
        Depth of the perched water table level (m)

    soil_moisture_factor : float
        Soil Moisture Factor to adjust the soil moisture coefficient. This
        factor may be used during calibration. Default value is 1.0.

    conductivity_factor : float
        Conductivity Factor to adjust the vertical conductivity. This factor
        may be used during calibration. Default value is 1.0.

    lookup_and_forcing_files : list of pathlib.Path or str
        List of paths to "extra files" required by MetaSWAP. This a list of
        lookup tables and meteorological information that is required by
        MetaSwap. Note that MetaSwap looks for files with a specific name, so
        calling "luse_svat.inp" something else will result in errors. To view
        the files required, you can call: ``print(MetaSwap()._required_extra)``

    """

    _pkg_id = "cap"
    _variable_order = [
        "boundary",
        "landuse",
        "rootzone_thickness",
        "soil_physical_unit",
        "meteo_station_number",
        "surface_elevation",
        "sprinkling_type",
        "sprinkling_layer",
        "sprinkling_capacity",
        "wetted_area",
        "urban_area",
        "ponding_depth_urban",
        "ponding_depth_rural",
        "runoff_resistance_urban",
        "runoff_resistance_rural",
        "runon_resistance_urban",
        "runon_resistance_rural",
        "infiltration_capacity_urban",
        "infiltration_capacity_rural",
        "perched_water_table",
        "soil_moisture_factor",
        "conductivity_factor",
    ]

    _template_projectfile = jinja2.Template(
        "0001, ({{pkg_id}}), 1, {{name}}, {{variable_order}}\n"
        '{{"{:03d}".format(variable_order|length)}}, {{"{:03d}".format(n_entry)}}\n'
        "{%- for variable in variable_order%}\n"  # Preserve variable order
        "{%-    for layer, value in package_data[variable].items()%}\n"
        "{%-        if value is string %}\n"
        # If string then assume path:
        # 1 indicates the layer is activated
        # 2 indicates the second element of the final two elements should be read
        # 1.000 is the multiplication factor
        # 0.000 is the addition factor
        # -9999 indicates there is no data, following iMOD usual practice
        '1, 2, {{"{:03d}".format(layer)}}, 1.000, 0.000, -9999., {{value}}\n'
        "{%-        else %}\n"
        # Else assume a constant value is provided
        '1, 1, {{"{:03d}".format(layer)}}, 1.000, 0.000, {{value}}, ""\n'
        "{%-        endif %}\n"
        "{%-    endfor %}\n"
        "{%- endfor %}\n"
        # Section for EXTRA FILES comes below
        '{{"{:03d}".format(extra_files|length)}},extra files\n'
        "{%- for file in extra_files %}\n"
        "{{file}}\n"
        "{%- endfor %}\n"
    )

    # TODO: Check which of these actually are required.
    _required_extra = [
        "fact_svat.inp",
        "luse_svat.inp",
        "mete_grid.inp",
        "para_sim.inp",
        "tiop_sim.inp",
        "init_svat.inp",
        "comp_post.inp",
        "sel_key_svat_per.inp",
    ]

    def __init__(
        self,
        boundary,
        landuse,
        rootzone_thickness,
        soil_physical_unit,
        meteo_station_number,
        surface_elevation,
        sprinkling_type,
        sprinkling_layer,
        sprinkling_capacity,
        wetted_area,
        urban_area,
        ponding_depth_urban,
        ponding_depth_rural,
        runoff_resistance_urban,
        runoff_resistance_rural,
        runon_resistance_urban,
        runon_resistance_rural,
        infiltration_capacity_urban,
        infiltration_capacity_rural,
        perched_water_table,
        lookup_and_forcing_files,
        soil_moisture_factor=1.0,
        conductivity_factor=1.0,
    ):
        super(__class__, self).__init__()
        self.dataset["boundary"] = boundary
        self.dataset["landuse"] = landuse
        self.dataset["rootzone_thickness"] = rootzone_thickness
        self.dataset["soil_physical_unit"] = soil_physical_unit
        self.dataset["meteo_station_number"] = meteo_station_number
        self.dataset["surface_elevation"] = surface_elevation
        self.dataset["sprinkling_type"] = sprinkling_type
        self.dataset["sprinkling_layer"] = sprinkling_layer
        self.dataset["sprinkling_capacity"] = sprinkling_capacity
        self.dataset["wetted_area"] = wetted_area
        self.dataset["urban_area"] = urban_area
        self.dataset["ponding_depth_urban"] = ponding_depth_urban
        self.dataset["ponding_depth_rural"] = ponding_depth_rural
        self.dataset["runoff_resistance_urban"] = runoff_resistance_urban
        self.dataset["runoff_resistance_rural"] = runoff_resistance_rural
        self.dataset["runon_resistance_urban"] = runon_resistance_urban
        self.dataset["runon_resistance_rural"] = runon_resistance_rural
        self.dataset["infiltration_capacity_urban"] = infiltration_capacity_urban
        self.dataset["infiltration_capacity_rural"] = infiltration_capacity_rural
        self.dataset["perched_water_table"] = perched_water_table
        self.dataset["soil_moisture_factor"] = soil_moisture_factor
        self.dataset["conductivity_factor"] = conductivity_factor
        self.lookup_and_forcing_files = lookup_and_forcing_files

    def _force_absolute_path(self, f):
        """Force absolute path, because projectfile cannot handle relative paths"""
        return str(pathlib.Path(f).resolve())

    def _render_projectfile(self, **kwargs):
        """
        Returns
        -------
        rendered : str
            The rendered projfectfile part,
            if part of PkgGroup: for a single boundary condition system.
        """
        extra_files = [self._force_absolute_path(file) for file in self.extra_files]
        kwargs["extra_files"] = extra_files
        return self._template_projectfile.render(**kwargs)

    def check_lookup_and_forcing_files(self):
        """Check for presence of required MetaSWAP input files."""
        filenames = set([os.path.basename(f) for f in self.extra_files])
        missing_files = set(self._required_extra) - filenames
        if len(missing_files) > 0:
            raise ValueError(f"Missing MetaSWAP input files {missing_files}")

    def _pkgcheck(self, active_cells=None):
        # Dataset.dims does not return a tuple, like DataArray does.
        # http://xarray.pydata.org/en/stable/generated/xarray.Dataset.dims.html
        dims = tuple(self.dataset.dims.keys())
        # Frozen(SortedKeysDict).keys() does not preserve ordering in keys
        if dims != ("x", "y"):
            raise ValueError(f'Dataset dims not ("y", "x"), instead got {dims}')
        self.check_lookup_and_forcing_files()
