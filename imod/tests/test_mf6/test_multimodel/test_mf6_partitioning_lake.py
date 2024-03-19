
import pytest
import copy
from imod.typing.grid import zeros_like

@pytest.mark.usefixtures("rectangle_with_lakes")
def test_mf6_partition_with_lakes(rectangle_with_lakes, tmp_path):
    simulation = rectangle_with_lakes
    simulation.write(tmp_path,False,True,False )
    simulation.run()


    label_array = zeros_like(simulation["GWF_1"].domain.sel(layer=1))
    label_array.values[10:, 10:] = 1

    split_simulation = simulation.split(label_array)
    pass