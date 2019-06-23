.. imod documentation master file, created by
   sphinx-quickstart on Tue Apr 10 12:38:06 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

iMOD-Python: work with iMOD MODFLOW models in Python
====================================================

Work with `iMOD <https://oss.deltares.nl/web/imod>`__ MODFLOW models in
Python.

Documentation: https://deltares.gitlab.io/imod/imod-python/

Source code: https://gitlab.com/deltares/imod/imod-python


Documentation
-------------

**Getting Started**

* :doc:`overview`
* :doc:`installation`
* :doc:`examples`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   overview
   installation
   examples

**User Guide**

* :doc:`data-structures`
* :doc:`coordinates`
* :doc:`regridding`
* :doc:`indexing`
* :doc:`model`
* :doc:`post-processing`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: User Guide

   data-structures
   coordinates
   regridding
   indexing
   model
   post-processing

**Help & reference**

* :doc:`changelog`
* :doc:`api`
* :doc:`internals`
* :doc:`roadmap`
* :doc:`contributing`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Help & reference

   changelog
   api
   internals
   roadmap
   contributing


Getting started
---------------

Install the latest release using ``conda install -c conda-forge imod``, or, when
not using Anaconda, ``pip install imod``. For more detailed installation
information see :doc:`installation`.

.. code:: python

   import imod

   # read and write IPF files to pandas DataFrame
   df = imod.ipf.read('wells.ipf')
   imod.ipf.save('wells-out.ipf', df)

   # get all calculated heads in a xarray DataArray
   # with dimensions time, layer, y, x
   da = imod.idf.open('path/to/results/head_*.idf')

Authors
-------
This Python package was written primarily by Martijn Visser and Huite Bootsma at Deltares.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
