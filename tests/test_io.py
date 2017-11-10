import os
import unittest
import imod
import numpy as np
# import xarray as xr
# import pandas as pd


class TestIDF(unittest.TestCase):
    def setUp(self):
        self.idf = 'test.idf'
        self.arr = np.ones((3, 4), dtype=np.float32)
        self.meta = {'xmin': 0.0, 'xmax': 4.0, 'ymin': 0.0, 'ymax': 3.0,
                     'nodata': -9999.0, 'dx': 1.0, 'dy': 1.0}

    def tearDown(self):
        try:
            os.remove(self.idf)
        except FileNotFoundError:
            pass

    # TODO update tests after DataArray rewrite
    def test_idf(self):
        imod.io.writeidf(self.idf, self.arr, self.meta)
        self.assertTrue(os.path.isfile(self.idf))
        d = imod.io.readidf(self.idf, nodata=None, header_only=True)
        self.assertIsInstance(d, dict)
        # check if the returned dict is a superset of self.meta
        self.assertTrue(all(item in d.items() for item in self.meta.items()))
        self.assertIn(('itb', False), d.items())
        self.assertIn(('nrow', 3), d.items())
        self.assertIn(('ncol', 4), d.items())
        a, d = imod.io.readidf(self.idf, nodata=None, header_only=False)
        self.assertIsInstance(a, np.ndarray)
        a, d = imod.io.readidf(self.idf, nodata='mask', header_only=False)
        self.assertIsInstance(a, np.ma.MaskedArray)


if __name__ == '__main__':
    unittest.main()
