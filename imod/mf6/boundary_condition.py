import abc
import pathlib
from copy import copy, deepcopy
from dataclasses import asdict
from typing import Mapping, Optional, Self, Union

import numpy as np
import xarray as xr
import xugrid as xu

from imod.mf6.aggregate.aggregate_schemes import EmptyAggregationMethod
from imod.mf6.auxiliary_variables import (
    expand_transient_auxiliary_variables,
    get_variable_names,
)
from imod.mf6.dis import StructuredDiscretization
from imod.mf6.disv import VerticesDiscretization
from imod.mf6.npf import NodePropertyFlow
from imod.mf6.package import Package
from imod.mf6.utilities.package import get_repeat_stress
from imod.mf6.write_context import WriteContext
from imod.prepare.topsystem import (
    ALLOCATION_OPTION,
    DISTRIBUTING_OPTION,
    SimulationAllocationOptions,
    SimulationDistributingOptions,
)
from imod.typing import GridDataArray, GridDataDict, GridDataset


def _dis_recarr(arrdict, layer, notnull):
    # Define the numpy structured array dtype
    index_spec = [("layer", np.int32), ("row", np.int32), ("column", np.int32)]
    field_spec = [(key, np.float64) for key in arrdict]
    sparse_dtype = np.dtype(index_spec + field_spec)
    # Initialize the structured array
    nrow = notnull.sum()
    recarr = np.empty(nrow, dtype=sparse_dtype)
    # Fill in the indices
    if notnull.ndim == 2:
        recarr["row"], recarr["column"] = (np.argwhere(notnull) + 1).transpose()
        recarr["layer"] = layer
    else:
        ilayer, irow, icolumn = np.argwhere(notnull).transpose()
        recarr["row"] = irow + 1
        recarr["column"] = icolumn + 1
        recarr["layer"] = layer[ilayer]
    return recarr


def _disv_recarr(arrdict, layer, notnull):
    # Define the numpy structured array dtype
    index_spec = [("layer", np.int32), ("cell2d", np.int32)]
    field_spec = [(key, np.float64) for key in arrdict]
    sparse_dtype = np.dtype(index_spec + field_spec)
    # Initialize the structured array
    nrow = notnull.sum()
    recarr = np.empty(nrow, dtype=sparse_dtype)
    # Fill in the indices
    if notnull.ndim == 1 and layer.size == 1:
        recarr["cell2d"] = (np.argwhere(notnull) + 1).transpose()
        recarr["layer"] = layer
    else:
        ilayer, icell2d = np.argwhere(notnull).transpose()
        recarr["cell2d"] = icell2d + 1
        recarr["layer"] = layer[ilayer]
    return recarr


def _handle_reallocation_options(
    pkg_id: str,
    allocation_option: Optional[ALLOCATION_OPTION],
    distributing_option: Optional[DISTRIBUTING_OPTION],
) -> tuple[ALLOCATION_OPTION, DISTRIBUTING_OPTION]:
    if allocation_option is None:
        allocation_option = asdict(SimulationAllocationOptions())[pkg_id]
    elif allocation_option == ALLOCATION_OPTION.stage_to_riv_bot_drn_above:
        raise ValueError(
            f"Allocation option {allocation_option} is not supported for "
            "reallocation of boundary conditions."
        )
    if distributing_option is None:
        distributing_option = asdict(SimulationDistributingOptions())[pkg_id]
    return allocation_option, distributing_option


class BoundaryCondition(Package, abc.ABC):
    """
    BoundaryCondition is used to share methods for specific stress packages
    with a time component.

    It is not meant to be used directly, only to inherit from, to implement new
    packages.

    This class only supports `list input
    <https://water.usgs.gov/water-resources/software/MODFLOW-6/mf6io_6.0.4.pdf#page=19>`_,
    not the array input which is used in :class:`Package`.
    """

    def __init__(self, allargs: Mapping[str, GridDataArray | float | int | bool | str]):
        # Convert repeat_stress in dict to a xr.DataArray in the right shape if
        # necessary, which is required to merge it into the dataset.
        if "repeat_stress" in allargs.keys() and isinstance(
            allargs["repeat_stress"], dict
        ):
            allargs["repeat_stress"] = get_repeat_stress(allargs["repeat_stress"])  # type: ignore
        # Call the Package constructor, this merges the arguments into a dataset.
        super().__init__(allargs)
        if "concentration" in allargs.keys() and allargs["concentration"] is None:
            # Remove vars inplace
            del self.dataset["concentration"]
            del self.dataset["concentration_boundary_type"]
        else:
            expand_transient_auxiliary_variables(self)

    def _max_active_n(self):
        """
        Determine the maximum active number of cells that are active
        during a stress period.
        """
        da = self.dataset[self.get_period_varnames()[0]]
        if "time" in da.coords:
            nmax = int(da.groupby("time").count(xr.ALL_DIMS).max())
        else:
            nmax = int(da.count())
        return nmax

    def _write_binaryfile(self, outpath, struct_array):
        with open(outpath, "w") as f:
            struct_array.tofile(f)

    def _write_textfile(self, outpath, struct_array):
        fields = struct_array.dtype.fields
        fmt = [self._number_format(field[0]) for field in fields.values()]
        header = " ".join(list(fields.keys()))
        np.savetxt(fname=outpath, X=struct_array, fmt=fmt, header=header)

    def _write_datafile(self, outpath, ds, binary):
        """
        Writes a modflow6 binary data file
        """
        layer = ds["layer"].values if "layer" in ds.coords else None
        arrdict = self._ds_to_arrdict(ds)
        struct_array = self._to_struct_array(arrdict, layer)
        outpath.parent.mkdir(exist_ok=True, parents=True)
        if binary:
            self._write_binaryfile(outpath, struct_array)
        else:
            self._write_textfile(outpath, struct_array)

    def _ds_to_arrdict(self, ds):
        for datavar in ds.data_vars:
            if ds[datavar].shape == ():
                raise ValueError(
                    f"{datavar} in {self._pkg_id} package cannot be a scalar"
                )

        arrdict = {}
        for datavar in ds.data_vars:
            arrdict[datavar] = ds[datavar].values

        return arrdict

    def _to_struct_array(self, arrdict, layer):
        """Convert from dense arrays to list based input"""
        # TODO stream the data per stress period
        # TODO add pkgcheck that period table aligns
        # Get the number of valid values
        if layer is None:
            raise ValueError("Layer should be provided")

        data = next(iter(arrdict.values()))
        notnull = ~np.isnan(data)

        if isinstance(self.dataset, xr.Dataset):
            recarr = _dis_recarr(arrdict, layer, notnull)
        elif isinstance(self.dataset, xu.UgridDataset):
            recarr = _disv_recarr(arrdict, layer, notnull)
        else:
            raise TypeError(
                "self.dataset should be xarray.Dataset or xugrid.UgridDataset,"
                f" is {type(self.dataset)} instead"
            )
        # Fill in the data
        for key, arr in arrdict.items():
            values = arr[notnull].astype(np.float64)
            recarr[key] = values

        return recarr

    def _period_paths(
        self, directory: pathlib.Path | str, pkgname: str, globaltimes, bin_ds, binary
    ):
        directory = pathlib.Path(directory) / pkgname

        if binary:
            ext = "bin"
        else:
            ext = "dat"

        periods: dict[np.int64, str] = {}
        # Force to np.int64 for mypy and numpy >= 2.2.4
        one = np.int64(1)
        if "time" in bin_ds:  # one of bin_ds has time
            package_times = bin_ds.coords["time"].values
            starts = np.searchsorted(globaltimes, package_times) + one
            for i, start in enumerate(starts):
                path = directory / f"{self._pkg_id}-{i}.{ext}"
                periods[start] = path.as_posix()

            repeat_stress = self.dataset.get("repeat_stress")
            if repeat_stress is not None and repeat_stress.values[()] is not None:
                keys = repeat_stress.isel(repeat_items=0).values
                values = repeat_stress.isel(repeat_items=1).values
                repeat_starts = np.searchsorted(globaltimes, keys) + one
                values_index = np.searchsorted(globaltimes, values) + one
                for j, start_repeat in zip(values_index, repeat_starts):
                    periods[start_repeat] = periods[j]
                # Now make sure the periods are sorted by key.
                periods = dict(sorted(periods.items()))
        else:
            path = directory / f"{self._pkg_id}.{ext}"
            periods[one] = path.as_posix()

        return periods

    def _get_unfiltered_pkg_options(
        self, predefined_options: dict, not_options: Optional[list] = None
    ):
        options = copy(predefined_options)

        if not_options is None:
            not_options = self.get_period_varnames()

        for varname in self.dataset.data_vars.keys():  # pylint:disable=no-member
            if varname in not_options:
                continue
            v = self.dataset[varname].values[()]
            options[varname] = v
        return options

    def _get_pkg_options(
        self, predefined_options: dict, not_options: Optional[list] = None
    ):
        unfiltered_options = self._get_unfiltered_pkg_options(
            predefined_options, not_options=not_options
        )
        # Filter out options which are None or False
        options = {
            key: value
            for key, value in unfiltered_options.items()
            if self._valid(value)
        }
        return options

    def _get_bin_ds(self):
        """
        Get binary dataset data for stress periods, this data will be written to
        datafiles. This method can be overriden to do some extra operations on
        this dataset before writing.
        """
        return self[self.get_period_varnames()]

    def render(self, directory, pkgname, globaltimes, binary):
        """Render fills in the template only, doesn't write binary data"""
        d = {"binary": binary}
        bin_ds = self._get_bin_ds()
        d["periods"] = self._period_paths(
            directory, pkgname, globaltimes, bin_ds, binary
        )
        # construct the rest (dict for render)
        d = self._get_pkg_options(d)
        d["maxbound"] = self._max_active_n()

        if (hasattr(self, "_auxiliary_data")) and (names := get_variable_names(self)):
            d["auxiliary"] = names

        return self._template.render(d)

    def _write_perioddata(self, directory, pkgname, binary):
        if len(self.get_period_varnames()) == 0:
            return
        bin_ds = self._get_bin_ds()

        if binary:
            ext = "bin"
        else:
            ext = "dat"

        if "time" in bin_ds:  # one of bin_ds has time
            for i in range(len(self.dataset.time)):
                path = directory / pkgname / f"{self._pkg_id}-{i}.{ext}"
                self._write_datafile(
                    path, bin_ds.isel(time=i), binary=binary
                )  # one timestep
        else:
            path = directory / pkgname / f"{self._pkg_id}.{ext}"
            self._write_datafile(path, bin_ds, binary=binary)

    def _write(
        self,
        pkgname: str,
        globaltimes: Union[list[np.datetime64], np.ndarray],
        write_context: WriteContext,
    ):
        """
        writes the blockfile and binary data

        directory is modelname
        """

        super()._write(pkgname, globaltimes, write_context)
        directory = write_context.write_directory

        self._write_perioddata(
            directory=directory,
            pkgname=pkgname,
            binary=write_context.use_binary,
        )

    def get_period_varnames(self):
        result = []
        if hasattr(self, "_period_data"):
            result.extend(self._period_data)
        if hasattr(self, "_auxiliary_data"):
            result.extend(get_variable_names(self))

        return result

    @classmethod
    def aggregate_layers(cls, dataset: GridDataset) -> GridDataDict:
        """
        Aggregate data over layers into planar dataset.

        Returns
        -------
        dict
            Dict of aggregated data arrays, where the keys are the variable
            names and the values are aggregated across the "layer" dimension.
        """
        aggr_methods = cls.get_aggregate_methods()
        if isinstance(aggr_methods, EmptyAggregationMethod):
            raise TypeError(
                f"Aggregation methods for {cls._pkg_id} package are not defined."
            )
        aggr_methods_dict = asdict(aggr_methods)
        planar_data = {
            key: dataset[key].reduce(func, dim="layer")
            for key, func in aggr_methods_dict.items()
            if key in dataset.data_vars
        }
        return planar_data

    def reallocate(
        self,
        dis: StructuredDiscretization | VerticesDiscretization,
        npf: Optional[NodePropertyFlow] = None,
        allocation_option: Optional[ALLOCATION_OPTION] = None,
        distributing_option: Optional[DISTRIBUTING_OPTION] = None,
    ) -> Self:
        """
        Reallocates topsystem data across layers and create new package with it.
        Aggregate data to planar data first, by taking either the mean for state
        variables (e.g. river stage), or the sum for fluxes and the
        conductance. Consequently allocate and distribute the planar data to the
        provided model layer schematization.

        Parameters
        ----------
        dis : StructuredDiscretization | VerticesDiscretization
            The discretization of the model to which the data should be
            reallocated.
        npf : NodePropertyFlow, optional
            The node property flow package of the model to which the conductance
            should be distributed (if applicable). Required for packages with a
            conductance variable.
        allocation_option : ALLOCATION_OPTION, optional
            The allocation option to use for the reallocation. If None, the
            default allocation option is taken from
            :class:`imod.prepare.SimulationAllocationOptions`.
        distributing_option : DISTRIBUTING_OPTION, optional
            The distributing option to use for the reallocation. Required for
            packages with a conductance variable. If None, the default is taken
            from :class:`imod.prepare.SimulationDistributingOptions`.

        Returns
        -------
        BoundaryCondition
            A new instance of the boundary condition class with the reallocated
            data. The original instance remains unchanged.
        """
        # Handle input arguments
        allocation_option, distributing_option = _handle_reallocation_options(
            self._pkg_id, allocation_option, distributing_option
        )
        has_conductance = "conductance" in self.dataset.data_vars
        if has_conductance and npf is None:
            raise ValueError(
                "NodePropertyFlow must be provided "
                "for packages with conductance variable."
            )
        # Aggregate data to planar data first
        planar_data = self.aggregate_layers(self.dataset)
        # Then allocate and distribute the planar data to the model layers
        if has_conductance:
            grid_dict = self.allocate_and_distribute_planar_data(
                planar_data, dis, npf, allocation_option, distributing_option
            )
        else:
            grid_dict = self.allocate_planar_data(planar_data, dis, allocation_option)
        # River package returns a tuple (second argument can also be Drainage
        # package)
        if isinstance(grid_dict, tuple):
            grid_dict, _ = grid_dict
        options = self._get_unfiltered_pkg_options({})
        data_dict = grid_dict | options
        return self.__class__(**data_dict)

    @classmethod
    def allocate_and_distribute_planar_data(
        cls,
        planar_data: GridDataDict,
        dis: StructuredDiscretization | VerticesDiscretization,
        npf: NodePropertyFlow,
        allocation_option: ALLOCATION_OPTION,
        distributing_option: DISTRIBUTING_OPTION,
    ) -> tuple[GridDataDict, GridDataDict] | GridDataDict:
        raise NotImplementedError(
            "This method should be implemented in the specific boundary condition "
            "class that inherits from BoundaryCondition."
        )

    @classmethod
    def allocate_planar_data(
        cls,
        planar_data: GridDataDict,
        dis: StructuredDiscretization | VerticesDiscretization,
        allocation_option: ALLOCATION_OPTION,
    ) -> tuple[GridDataDict, GridDataDict] | GridDataDict:
        raise NotImplementedError(
            "This method should be implemented in the specific boundary condition "
            "class that inherits from BoundaryCondition."
        )


class AdvancedBoundaryCondition(BoundaryCondition, abc.ABC):
    """
    Class dedicated to advanced boundary conditions, since MF6 does not support
    binary files for Advanced Boundary conditions.

    The advanced boundary condition packages are: "uzf", "lak", "maw", "sfr".

    """

    def _get_field_spec_from_dtype(self, recarr):
        """
        From https://stackoverflow.com/questions/21777125/how-to-output-dtype-to-a-list-or-dict
        """
        return [
            (x, y[0])
            for x, y in sorted(recarr.dtype.fields.items(), key=lambda k: k[1])
        ]

    def _write_file(self, outpath, sparse_data):
        """
        Write to textfile, which is necessary for Advanced Stress Packages
        """
        fields = sparse_data.dtype.fields
        fmt = [self._number_format(field[0]) for field in fields.values()]
        header = " ".join(list(fields.keys()))
        np.savetxt(fname=outpath, X=sparse_data, fmt=fmt, header=header)

    @abc.abstractmethod
    def _package_data_to_sparse(self):
        """
        Get packagedata, override with function for the advanced boundary
        condition in particular
        """
        return

    def write_packagedata(self, directory, pkgname, binary):
        outpath = directory / pkgname / f"{self._pkg_id}-pkgdata.dat"
        outpath.parent.mkdir(exist_ok=True, parents=True)
        package_data = self._package_data_to_sparse()
        self._write_file(outpath, package_data)

    def _write(
        self,
        pkgname: str,
        globaltimes: Union[list[np.datetime64], np.ndarray],
        write_context: WriteContext,
    ):
        boundary_condition_write_context = deepcopy(write_context)
        boundary_condition_write_context.use_binary = False

        self.fill_stress_perioddata()
        super()._write(pkgname, globaltimes, boundary_condition_write_context)

        directory = boundary_condition_write_context.write_directory
        self.write_packagedata(directory, pkgname, binary=False)

    @abc.abstractmethod
    def fill_stress_perioddata(self):
        raise NotImplementedError
