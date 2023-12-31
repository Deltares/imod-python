Installing
==========

Which Python?
-------------

You'll need **Python 3.7 or greater**. 

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
recommend using the `Mambaforge`_ Python distribution. This installs Python and
the ``mamba`` package manager. `Miniforge`_ and `Miniconda`_ will only install Python
and the ``conda`` package manager. Differences to note, in a nutshell:

* ``mamba`` is much faster than ``conda``, but has identical commands. 
* Mambaforge and miniforge are community driven installers, installing by
  default from the ``conda-forge`` channel.
* Miniconda is a company driven (Anaconda) installer, installing by default
  from the ``anaconda`` channel.
* Installing from the ``anaconda`` channel has certain (legal) `limitations`_
  for "commercial use".

Installing Mambaforge/Miniforge/Miniconda does not require administrative
rights to your computer and doesn't interfere with any other Python
installations in your system.

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

Installing with mamba
^^^^^^^^^^^^^^^^^^^^^

You can install ``imod`` using the `mamba package manager`_ that comes with the
Mambaforge distribution. We advice to install ``imod`` in a seperate ``conda``
environment, as you can simply delete these in case they break. Not doing so
will install imod and its dependencies in your base environment, which requires
a reinstall of Mambaforge in case this environment breaks::

  mamba create -n imodenv
  mamba install -n imodenv imod --channel conda-forge
  
``mamba`` will automatically find the appropriate versions of the dependencies
and in this case install them in the ``imodenv`` environment. Installing with
mamba or conda will automatically download *all* optional dependencies, and
enable all functionality.

To run scripts using ``imod``, you first have to activate the ``imodenv``
environment::

  conda activate imodenv

Installing with conda
^^^^^^^^^^^^^^^^^^^^^

Alternatively, you can also use the `conda package manager`_. Like mamba, conda
will also infer the appropriate versions of the dependencies and install them.
However, it generally takes around a factor 5 longer to do so, but may be
worthwhile if mamba is unstable or buggy::

  conda create -n imodenv
  conda install -n imodenv imod --channel conda-forge

To run scripts using ``imod``, you first have to activate the ``imodenv``
environment::

  conda activate imodenv

Installing with pip
^^^^^^^^^^^^^^^^^^^

Finally, you can also use the `pip package manager`_::

  pip install imod
  
Unlike installing with conda or mamba, installing with pip will not install
all optional dependencies. This results in a far smaller installation, but
it means that not all functionality is directly available.

Refer to :doc:`../faq/python` in the FAQ section for background
information on ``mamba``, ``conda``, and ``pip``.

Installing the latest development version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use ``pip`` to install the latest source from GitHub::

  pip install git+https://github.com/Deltares/imod-python.git

Alternatively, you can clone the git repository locally and install from there::

  git clone https://github.com/Deltares/imod-python.git
  cd imod-python
  pip install -e .

.. _Verde's: https://www.fatiando.org/verde/latest/install.html
.. _Deltaforge: https://deltares.github.io/deltaforge/
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _Mambaforge: https://github.com/conda-forge/miniforge#mambaforge
.. _Miniforge: https://github.com/conda-forge/miniforge
.. _limitations: https://www.anaconda.com/blog/anaconda-commercial-edition-faq
.. _mamba package manager: https://github.com/mamba-org/mamba
.. _conda package manager: https://docs.conda.io/en/latest/
.. _pip package manager: https://pypi.org/project/pip/
