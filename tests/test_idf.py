import os
import unittest
import imod
import numpy as np
from collections import OrderedDict
import xarray as xr
import pandas as pd


class TestIDF(unittest.TestCase):
    def setUp(self):
        self.idf = 'test.idf'
        arr = np.ones((3, 4), dtype=np.float32)
        cellwidth = 1.0
        cellheight = cellwidth
        xmin = 0.0
        ymax = 3.0
        attrs = OrderedDict()
        attrs['res'] = (cellwidth, cellheight)
        attrs['transform'] = (cellwidth, 0.0, xmin, 0.0, -cellheight, ymax)
        kwargs = {
            'name': 'test',
            'dims': ('y', 'x'),  # only two dimensions in a single IDF
            'attrs': attrs,
        }
        self.da = xr.DataArray(arr, **kwargs)

    def tearDown(self):
        try:
            os.remove(self.idf)
        except FileNotFoundError:
            pass

    def test_idf(self):
        imod.io.writeidf('.', self.da)
        self.assertTrue(os.path.isfile(self.idf))
        # set memmap to False to avoid tearDown PermissionError
        da2 = imod.io.loadarray(self.idf, memmap=False)
        self.assertIsInstance(da2, xr.DataArray)
        self.assertTrue((self.da == da2).all())


if __name__ == '__main__':
    unittest.main()
