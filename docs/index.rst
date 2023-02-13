iMOD Python: make massive MODFLOW models
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

The ``imod`` Python package is an open source project to make working with
MODFLOW groundwater models in Python easier. It builds on top of popular
packages such as `xarray`_, `pandas`_, `geopandas`_, `dask`_,  and `rasterio`_
to provide a versatile toolset for working with (large) groundwater (modeling)
data:

* Preparing and modifying data from a variety of GIS, scientific, and MODFLOW
  file formats;
* Writing data to MODFLOW-based models;
* Selecting and evaluating for e.g. time series comparison or water budgets;
* Visualizing cross sections, time series, or 3D animations.
  
We currently support the following MODFLOW-based models:

* `USGS MODFLOW 6`_ (:doc:`api/mf6`), structured (DIS) and discretization by
  vertices (DISV) grids only, and not all advanced stress packages yet (LAK,
  MAW, SFR, UZF)
* `iMOD-WQ`_ (:doc:`api/wq`), which integrates SEAWAT (density-dependent
  groundwater flow) and MT3DMS (multi-species reactive transport calculations)
* `iMODFLOW`_ (:doc:`api/flow`)

This Python package is developed primarily by the Groundwater Management Group
at `Deltares`_. It is used together with a broader set of open source tools and
standards for reproducible modeling and data analysis:

* `Git`_: version control of (Python) scripts;
* `DVC`_: version control of data, on top of Git;
* `netCDF`_: open standard of a flexible, self describing data format;
* `Snakemake`_: workflow manager to turn a collection of scripts into a
  workflow.

Documentation: https://deltares.gitlab.io/imod/imod-python/

Source code: https://gitlab.com/deltares/imod/imod-python

Issues: https://gitlab.com/deltares/imod/imod-python/-/issues/

.. raw:: html

   <br/>
   <a class="reference external image-reference" href="https://www.deltares.nl">
      <img src="_static/deltares.svg" style="height:60px;"/>
      <br/>
      <img src="_static/enabling-delta-life.svg" style="height:60px;"/>
   </a>

.. toctree::
   :titlesonly:
   :hidden:

   installation
   user-guide/index
   examples/index
   api/index
   faq/index
   developing

.. _Deltares: https://www.deltares.nl
.. _dask: https://dask.org/
.. _xarray: http://xarray.pydata.org/
.. _pandas: http://pandas.pydata.org/
.. _rasterio: https://rasterio.readthedocs.io/en/latest/
.. _geopandas: http://geopandas.org/
.. _Git: https://git-scm.com/
.. _DVC: https://dvc.org/
.. _netCDF: https://www.unidata.ucar.edu/software/netcdf/
.. _Snakemake: https://snakemake.readthedocs.io/en/stable/
.. _USGS MODFLOW 6: https://www.usgs.gov/software/modflow-6-usgs-modular-hydrologic-model
.. _iMODFLOW: https://oss.deltares.nl/web/imod
.. _iMOD-WQ: https://oss.deltares.nl/web/imod
