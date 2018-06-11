import os
import unittest
from imod import util
from collections import OrderedDict
from datetime import datetime
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
