from imod.mf6.simulation import Modflow6Simulation
from imod.mf6.model import Modflow6Model, GroundwaterFlowModel
from imod.typing import GridDataArray
from fastcore.dispatch import typedispatch
import xugrid as xu
import xarray as xa
from imod.util import  to_ugrid2d
from math import sqrt
import copy

def get_label_array(simulation: Modflow6Simulation, npartitions: int) -> GridDataArray:

    gwf_models =simulation.get_models_of_type("gwf6")
    if len(gwf_models) != 1:
        raise ValueError("for partitioning a simulation to work, it must have exactly 1 flow model")
    if npartitions <= 0:
        raise ValueError("You should create at least 1 partition")    

    flowmodel = list(gwf_models.values())[0]
    idomain = flowmodel.domain
    idomain_top = copy.deepcopy( idomain.isel(layer= 0))


    return  partition_idomain( idomain_top, npartitions)

@typedispatch
def partition_idomain(idomain_grid: xu.UgridDataArray, npartitions: int) -> GridDataArray:
    labels = idomain_grid.ugrid.grid.label_partitions(n_part=npartitions)
    labels = labels.rename("idomain")
    return labels


@typedispatch
def partition_idomain(idomain_grid: xa.DataArray, npartitions: int) -> GridDataArray:
    
    gb_x = idomain_grid.shape[0]
    gb_y = idomain_grid.shape[1]

    nx, ny = get_partition_numbers(npartitions)

    if gb_x > gb_y:
        nx, ny = ny, nx

    x_partition = partition_1d( nx, gb_x)
    y_partition = partition_1d( ny, gb_y)

    ipartition = -1
    for ipartx in x_partition:
        start_x, stop_x  = ipartx
        for iparty in y_partition:
            start_y, stop_y  = iparty
            ipartition = ipartition + 1
            idomain_grid.values[start_x: stop_x+1, start_y: stop_y+1] = ipartition
    return  idomain_grid



def partition_1d(nr_partitions, nr_indices):
    npart = int(nr_indices / nr_partitions)
    partitions =[]
    for i in range(nr_partitions-1):
        start = i * npart
        stop = start + npart
        partitions.append( (start, stop))
        start = stop
    partitions.append( (start, nr_indices-1))
    return partitions




def get_partition_numbers( number_partitions: int ) -> (int, int):

    root = sqrt(number_partitions + 1)
    factor = int(root)
    while factor > 0:
        if number_partitions / factor == int( number_partitions / factor):
            return ( factor, int( number_partitions / factor))
        else:
            factor -=1
    
    