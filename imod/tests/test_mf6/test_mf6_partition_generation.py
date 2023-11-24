from imod.mf6.partition_generator import _partition_1d, get_label_array
import pytest
import numpy as np

def test_partition_1d_errors():
    
    # test for errors if the mean partition size is less than 3 gridblocks (only acceptable if 1 partition is asked)
    with pytest.raises(ValueError):
        _partition_1d(nr_partitions=4, axis_size=10)

    # test for errors if 0 partitions are requested 
    with pytest.raises(ValueError):
        _partition_1d( nr_partitions=0 , axis_size=10)             

def test_partition_1d_partition_short_axis():
    assert _partition_1d(nr_partitions=1, axis_size=1) ==[(0,0)]   

def test_partition_1d_happy_flow():
    
    assert _partition_1d( nr_partitions=3, axis_size=10) == [(0,2),(3,5),(6,9)]
    assert _partition_1d( nr_partitions=2, axis_size=10) == [(0,4),(5,9)]
    assert _partition_1d( nr_partitions=1, axis_size=10) == [(0,9)]
    assert _partition_1d( nr_partitions=4, axis_size=15) == [(0, 2), (3, 5), (6, 8), (9, 14)]  
    assert _partition_1d( nr_partitions=4, axis_size=16) == [(0, 3), (4, 7), (8, 11), (12, 15)]    

@pytest.mark.usefixtures("circle_model")
def test_partition_2d_unstructured(circle_model):
    for nr_partitions in range(1, 20):
        label_array = get_label_array(circle_model, nr_partitions)

        #check that the labes up to nr_partitions -1 appear in the label array, and not any others
        unique, counts = np.unique(label_array, return_counts=True)
        assert np.all(counts >1)
        assert len(unique) == nr_partitions
        assert max(unique) == nr_partitions -1
        assert min(unique) == 0        


@pytest.mark.usefixtures("twri_model")
def test_partition_2d_structured(twri_model):  
    
    #we skip a few partition numbers which would give an error.
    partition_numbers = [1,2,3, 4,5,6,8,9,10,12,15,16,20] 
    for nr_partitions in partition_numbers:
        label_array = get_label_array(twri_model, nr_partitions)
            
        unique, counts = np.unique(label_array, return_counts=True)
        #check that the labes up to nr_partitions -1 appear in the label array, and not any others
        assert np.all(counts >1)
        assert len(unique) == nr_partitions
        assert max(unique) == nr_partitions -1
        assert min(unique) == 0        

