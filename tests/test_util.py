import os
import unittest
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np

from imod import util


class TestUtil(unittest.TestCase):
    def test_compose(self):
        d = {
            "name": "head",
            "directory": Path("path", "to"),
            "extension": ".idf",
            "layer": 5,
            "time": datetime(2018, 2, 22, 9, 6, 57),
        }
        path = util.compose(d)
        self.assertIsInstance(path, Path)
        targetpath = Path(d["directory"], "head_20180222090657_l5.idf")
        self.assertEqual(path, targetpath)

    def test_decompose(self):
        d = util.decompose("path/to/head_20180222090657_l5.idf")
        refd = OrderedDict(
            [
                ("extension", ".idf"),
                ("directory", Path("path", "to")),
                ("name", "head"),
                ("time", np.datetime64('2018-02-22T09:06:57.000000')),
                ("layer", 5),
            ]
        )
        self.assertIsInstance(d, OrderedDict)
        self.assertEqual(d, refd)

    def test_decompose_dateonly(self):
        d = util.decompose("20180222090657.idf")
        print(d)
        refd = OrderedDict(
            [
                ("extension", ".idf"),
                ("directory", Path(".")),
                ("name", "20180222090657"),
                ("time", np.datetime64('2018-02-22T09:06:57.000000')),
            ]
        )
        self.assertIsInstance(d, OrderedDict)
        self.assertEqual(d, refd)



if __name__ == "__main__":
    unittest.main()
