Automation with tox
===================

This package uses `tox`_ to automate code formatting, linting, and testing.
tox builds an isolated environment for these steps. Because imod has a large
number of dependencies, we rely on the `tox-conda`_ plugin to build these
isolated environments with mamba.

.. note::

    Running the steps below will result in the creation of three conda
    environments in the ``.tox`` directory. The format and linting environments
    are comparatively small (140 MB), but the build environment requires a full
    installation of the imod package and all its dependencies (>2.5 GB).

To format the code with `isort`_ and `black`_, run::

    tox -e format
    
To lint the code with isort, black, and `flake8`_::

    tox -e lint
    
To run the tests and build the documentation, run::

    tox -e build
    
This will run all the tests with pytest, and builds the documentation with
sphinx. Building the documentation is performed in the same step and
environment as running the tests as building the documentation includes running
all the examples in the ``.examples`` directory, which requires a full
installation of the imod package and all its dependencies. Separating the
testing and documentation building requires building two large identical
environments.

.. _tox: https://tox.wiki/en/latest/index.html
.. _tox-conda: https://github.com/tox-dev/tox-conda
.. _isort: https://github.com/PyCQA/isort
.. _black: https://github.com/psf/black
.. _flake8: https://github.com/PyCQA/flake8
