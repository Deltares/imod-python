import contextlib
import os
import pathlib
import warnings
from typing import Union


@contextlib.contextmanager
def ignore_warnings():
    """
    Contextmanager to ignore RuntimeWarnings as they are frequently
    raised by the Dask delayed scheduler.

    Examples
    --------
    >>> with imod.util.context.ignore_warnings():
            function_that_throws_warnings()

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield

@contextlib.contextmanager
def cd(path: Union[str, pathlib.Path]):
    """
    Change directory, and change it back after the with block.

    Examples
    --------
    >>> with imod.util.context.cd("docs"):
            do_something_in_docs()

    """
    curdir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(curdir)
