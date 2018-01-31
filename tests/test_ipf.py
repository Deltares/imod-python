import os
import unittest
import filecmp
from imod import ipf
import numpy as np
from collections import OrderedDict
import xarray as xr
import pandas as pd


def remove(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


class TestIPF(unittest.TestCase):
    def setUp(self):
        self.path = 'test.ipf'
        self.path_out = 'test-out.ipf'
        # example from iMOD manual
        self.ipfstring = (
            '2\n'
            '4\n'
            'X\n'
            'Y\n'
            'Z\n'
            '"City of Holland"\n'
            '0,TXT\n'
            '100.0,435.0,-32.3,Amsterdam\n'
            '553.0,143.0,-7.3,"Den Bosch"\n'
        )
        with open(self.path, 'w') as f:
            f.write(self.ipfstring)

    def tearDown(self):
        remove(self.path)
        remove(self.path_out)

    def test_load(self):
        df = ipf.load(self.path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(list(df) == ['X', 'Y', 'Z', 'City of Holland'])
        self.assertTrue(len(df) == 2)
        self.assertTrue(df.iloc[0, 2] == -32.3)
        self.assertTrue(df.iloc[1, 3] == 'Den Bosch')
        ipf.save('.', df, 'test-out')
        # cannot do exact comparison since quoting is slightly different
        # manual says: The different data for each field should be delimited by
        # a single (or more) space(s), or a comma.
        # we assume the different delimiters are not mixed inside an IPF
        df2 = ipf.load(self.path_out)
        self.assertTrue(df.equals(df2))


if __name__ == '__main__':
    unittest.main()
