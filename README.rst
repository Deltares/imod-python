.. image:: https://img.shields.io/badge/lifecycle-maturing-blue
   :target: https://www.tidyverse.org/lifecycle/
.. image:: https://dpcbuild.deltares.nl/app/rest/builds/buildType:id:iMOD6_IMODPython_Windows_Tests/statusIcon.svg
   :target: https://github.com/Deltares/imod-python/commits/master/
.. image:: https://img.shields.io/pypi/l/imod
   :target: https://choosealicense.com/licenses/mit/
.. image:: https://img.shields.io/conda/vn/conda-forge/imod.svg
   :target: https://github.com/conda-forge/imod-feedstock
.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json
   :target: https://pixi.sh

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

* `USGS MODFLOW 6`_, structured (DIS) and discretization by vertices (DISV)
  grids only, and not all advanced stress packages (only LAK and UZF)
* `iMOD-WQ`_, which integrates SEAWAT (density-dependent
  groundwater flow) and MT3DMS (multi-species reactive transport calculations)

Development currently focuses on supporting more Modflow 6 functionalities. 

This Python package is developed primarily by the Groundwater Management Group
at `Deltares`_. It is used together with a broader set of open source tools and
standards for reproducible modeling and data analysis:

* `Git`_: version control of (Python) scripts;
* `DVC`_: version control of data, on top of Git;
* `netCDF`_: open standard of a flexible, self describing data format;
* `Snakemake`_: workflow manager to turn a collection of scripts into a
  workflow.

Documentation: https://deltares.github.io/imod-python

Source code: https://github.com/Deltares/imod-python

Issues: https://github.com/Deltares/imod-python/issues

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
.. _iMOD-WQ: https://oss.deltares.nl/web/imod
