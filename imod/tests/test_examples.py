import os
import subprocess
import sys
from glob import glob
from pathlib import Path

import pytest


def get_examples():
    # Where are we? --> __file__
    # Move three up.
    path = Path(__file__).parent.parent.parent
    relpath = Path(os.path.relpath(path, os.getcwd())) / "examples/**/*.py"
    examples = [f for f in glob(str(relpath)) if f.endswith(".py")]
    return examples


@pytest.mark.example
@pytest.mark.parametrize("example", get_examples())
def test_example(example):
    subprocess.run([sys.executable, example], check=True)
