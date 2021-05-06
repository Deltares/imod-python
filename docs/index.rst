.. imod documentation master file, created by
   sphinx-quickstart on Tue Apr 10 12:38:06 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

iMOD-Python: make massive MODFLOW models
========================================

.. image:: https://img.shields.io/badge/lifecycle-maturing-blue
   :target: https://www.tidyverse.org/lifecycle/
.. image:: https://gitlab.com/deltares/imod/imod-python/badges/master/pipeline.svg
   :target: https://gitlab.com/deltares/imod/imod-python/commits/master
.. image:: https://img.shields.io/pypi/l/imod
   :target: https://choosealicense.com/licenses/mit/
.. image:: https://gitlab.com/deltares/imod/imod-python/badges/master/coverage.svg
   :target: https://gitlab.com/deltares/imod/imod-python/commits/master
.. image:: https://img.shields.io/conda/vn/conda-forge/imod.svg
   :target: https://github.com/conda-forge/imod-feedstock
.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/Deltares/iMOD-DSD-International-2019/master

.. note::

   This package is currently maturing on the way to a stable release. It is being actively used and
   developed at `Deltares <https://www.deltares.nl>`__. To make it easier for others to use this
   package, the documentation still needs significant work. The :doc:`api` is fairly complete, but
   high level overviews and more examples are still lacking. Extending Modflow 6 support is also planned.

The imod Python package is designed to help you in your MODFLOW groundwater modeling efforts.
It makes it easy to go from your raw data to a fully defined MODFLOW model, with the aim to make this process reproducable.
Whether you want to build a simple 2D conceptual model, or a complex 3D regional model with millions of cells,
imod-python scales automatically by making use of `dask <https://dask.org/>`__.

By building on top of popular Python packages like `xarray <http://xarray.pydata.org/>`__, `pandas <http://pandas.pydata.org/>`__,
`rasterio <https://rasterio.readthedocs.io/en/latest/>`__ and `geopandas <http://geopandas.org/>`__, a lot of functionality comes
for free.

Currently we support the creation of the following MODFLOW-based models:

* `USGS MODFLOW 6 <https://www.usgs.gov/software/modflow-6-usgs-modular-hydrologic-model>`__ (:doc:`api/mf6`), structured grids only, and no advanced stress packages yet (LAK, MAW, SFR, UZF)
* `iMODFLOW <https://oss.deltares.nl/web/imod>`__ (:doc:`api/flow`)
* `iMOD-WQ <https://oss.deltares.nl/web/imod>`__ (:doc:`api/wq`), which integrates SEAWAT (density-dependent groundwater flow) and MT3DMS (multi-species reactive transport calculations)

Documentation: https://imod.xyz/

Source code: https://gitlab.com/deltares/imod/imod-python

Interactive notebook examples:

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/Deltares/iMOD-DSD-International-2019/master

.. toctree::
   :titlesonly:
   :hidden:

   getting_started/index
   user_guide/index
   examples/index
   api/index

Getting started
---------------

Install the latest release using ``conda install -c conda-forge imod``, or, when
not using Anaconda, ``pip install imod``. For more detailed installation
information see :doc:`getting_started/installation`.

.. code:: python

   import imod

   # read and write IPF files to pandas DataFrame
   df = imod.ipf.read('wells.ipf')
   imod.ipf.save('wells-out.ipf', df)

   # get all calculated heads in a xarray DataArray
   # with dimensions time, layer, y, x
   da = imod.idf.open('path/to/results/head_*.idf')

   # create a groundwater model
   # abridged example, see examples for the full code
   gwf_model = imod.mf6.GroundwaterFlowModel()
   gwf_model["dis"] = imod.mf6.StructuredDiscretization(
       top=200.0, bottom=bottom, idomain=idomain
   )
   gwf_model["chd"] = imod.mf6.ConstantHead(
       head, print_input=True, print_flows=True, save_flows=True
   )
   simulation = imod.mf6.Modflow6Simulation("ex01-twri")
   simulation["GWF_1"] = gwf_model
   simulation.time_discretization(times=["2000-01-01", "2000-01-02"])
   simulation.write(modeldir)

Authors
-------
This Python package was written primarily by Martijn Visser and Huite Bootsma at `Deltares <https://www.deltares.nl>`__.

.. raw:: html

   <br/>
   <a class="reference external image-reference" href="https://www.deltares.nl">
      <img src="_static/deltares.svg" style="height:60px;"/>
      <br/>
      <img src="_static/enabling-delta-life.svg" style="height:60px;"/>
   </a>

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
