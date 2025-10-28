Installing
==========

TL;DR
-----

This is for people who are in a hurry and just want to get started. We advice
you to read the full documentation below if you are new to Python or the Python
packaging ecosystem.

Regular install
^^^^^^^^^^^^^^^

Install with pixi::

  pixi add imod

Install with conda::

  conda install imod

Install with pip::

  pip install imod

Install beta release
^^^^^^^^^^^^^^^^^^^^

To install a beta release with pixi::

  pixi config prepend default-channels "conda-forge/label/imod_rc"
  pixi add imod=1.0.0rc7

Or with conda::

  conda install -c conda-forge/label/imod_rc -c conda-forge imod=1.0.0rc7

Or with pip::

  pip install --pre imod=1.0.0rc7


Which Python?
-------------

You'll need **Python 3.10, 3.11, 3.12, or 3.13**. 

The recommended way to install Python depends on your experience: Are you new to
the Python packaging ecosystem or already got experience with it? If you already
know what ``conda`` and ``pip`` are, and are able to install tedious packages
GIS packages yourself without issue, you are *experienced*. If you have no idea
what the previous sentence was about, you are *new*.

New
^^^

We recommend new users to install iMOD Python using the `Deltaforge`_ Python
distribution. See :ref:`install_deltaforge`.

Experienced
^^^^^^^^^^^

For experienced users, who want to be in control of packages installed, we
recommend installing using `pixi`_. Pixi supports creating reproducible
projects, where all dependencies' versions in an environment are stored in a
textfile, the "lockfile". On top of that, it is faster than other package
managers.

Alternatively, you can use the `Miniforge`_ Python distribution. This installs
the ``conda`` package manager. This installer differs from the also
commonly-used `Miniconda`_ installer, in a nutshell:

* Miniforge is a community driven installers, installing by
  default from the ``conda-forge`` channel (Pixi also does this).
* Miniconda is a company driven installer, installing by default
  from the ``anaconda`` channel.
* Installing from the ``anaconda`` channel has certain (legal) `limitations`_
  for "commercial use", especially if you work in a large organization.

Installing Pixi/Mambaforge/Miniforge/Miniconda does not require administrative
rights to your computer and doesn't interfere with any other Python
installations in your system.

Ways to install iMOD Python
---------------------------

.. _install_deltaforge:

Installing with Deltaforge
^^^^^^^^^^^^^^^^^^^^^^^^^^

Deltaforge is an installer of Deltares python packages, including iMOD Python,
and their dependencies. This makes it possible to install iMOD Python without
much knowledge about the Python package management system. The download links
are listed `here. <https://deltares.github.io/deltaforge/index.html#where>`__
Users new to the python package ecosystem are recommended to install iMOD Python
using Deltaforge.

Installing with pixi
^^^^^^^^^^^^^^^^^^^^

Pixi supports creating reproducible projects, where all dependencies' versions
in an environment are stored in a textfile, the "lockfile". On top of that, it
is faster than other package managers. Installing with pixi is fast and easy::

  pixi init my_project
  cd my_project
  pixi add imod

These commands create a ``pixi.toml`` and a ``pixi.lock``. The ``pixi.toml``
contains the dependencies of your project. If you followed the commands above,
the only dependency present in the toml should be ``imod``. The ``pixi.lock`` is
a textfile that captures the environment in a specific state, so all
dependencies, with their dependencies and so on, with exact version numbers.
This is very useful, as it allows you to get the exact same environment on a
different machine, by just copying two textfiles. The lockfile will be updated
if you run ``pixi update`` or change the ``pixi.toml`` file. Both the
``pixi.toml`` as well as the ``pixi.lock`` are required to fully reproduce an
environment. 

You can start your favorite editor, e.g. VSCode, in this environment directly by
running::

  pixi run code

Installing with conda
^^^^^^^^^^^^^^^^^^^^^

Alternatively, you can also use the `conda package manager`_. We advice to
install ``imod`` in a seperate ``conda`` environment, as you can simply delete
these in case they break. Not doing so will install imod and its dependencies in
your base environment, which requires a reinstall of Miniforge in case this
environment breaks::

  conda create -n imodenv
  conda install -n imodenv imod --channel conda-forge

``conda`` will automatically find the appropriate versions of the dependencies
and in this case install them in the ``imodenv`` environment. Installing with
conda will automatically download *all* optional dependencies, and
enable all functionality.

To run scripts using ``imod``, you first have to activate the ``imodenv``
environment::

  conda activate imodenv

You can start your favorite editor, e.g. VSCode in this environment::

  code

Installing with pip
^^^^^^^^^^^^^^^^^^^

Finally, you can also use the `pip package manager`_::

  pip install imod
  
Unlike installing with conda, installing with pip will not install
all optional dependencies. This results in a far smaller installation, but
it means that not all functionality is directly available.

Refer to :doc:`../faq/python` in the FAQ section for background
information on ``conda``, and ``pip``.

Installing the latest development version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With pixi you can install the latest development version of imod::

  git clone https://github.com/Deltares/imod-python.git
  cd imod-python
  pixi run install

This will install the same python installation the iMOD Python developers work
with, so it should work (otherwise we couldn't do our work!). This contains an
interactive environment with Jupyter::

  pixi shell -e interactive

Alternatively, you can use ``pip`` to install the latest source from GitHub::

  pip install git+https://github.com/Deltares/imod-python.git

.. _Verde's: https://www.fatiando.org/verde/latest/install.html
.. _Deltaforge: https://deltares.github.io/deltaforge/
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _Miniforge: https://github.com/conda-forge/miniforge
.. _limitations: https://www.anaconda.com/blog/anaconda-commercial-edition-faq
.. _conda package manager: https://docs.conda.io/en/latest/
.. _pip package manager: https://pypi.org/project/pip/
.. _pixi: https://pixi.sh/latest/

Dependencies
------------

The ``imod`` Python package makes extensive use of the modern scientific Python
ecosystem. The most important dependencies are listed here.

Data structures:

* `pandas <https://pandas.pydata.org/>`__
* `numpy <https://www.numpy.org/>`__
* `xarray <https://xarray.pydata.org/>`__
* `xugrid <https://deltares.github.io/xugrid/>`__

Delayed/out-of-core computation, parallellization:

* `dask <https://dask.org/>`__
  
Spatial operations:

* `numba_celltree <https://deltares.github.io/numba_celltree/>`__
* `scipy <https://docs.scipy.org/doc/scipy/reference/>`__

Geospatial libaries (optional):

* `geopandas <https://geopandas.org/en/stable/>`__
* `shapely <https://shapely.readthedocs.io/en/stable/manual.html>`__
* `pyproj <https://pyproj4.github.io/pyproj/stable/>`__
* `rasterio <https://rasterio.readthedocs.io/en/latest/>`__

Data provisioning for examples: 

* `pooch <https://www.fatiando.org/pooch/>`__
  
Visualization:

* `matplotlib <https://matplotlib.org/>`__
* `pyvista <https://docs.pyvista.org/>`__ (Optional)
  
Installing all these dependencies requires around 2.5 gigabyte of space;
Installing only the required dependencies (via pip) requires around 0.5
gigabyte.