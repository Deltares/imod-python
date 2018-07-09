.. imod documentation master file, created by
   sphinx-quickstart on Tue Apr 10 12:38:06 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

iMOD-Python documentation
=========================

Work with `iMOD <http://oss.deltares.nl/web/imod>`__ MODFLOW models in
Python.

Documentation: https://deltares.gitlab.io/imod-python/

Source code: https://gitlab.com/deltares/imod-python

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   imod

Getting started
===============

.. code:: python

    import imod

    df = imod.ipf.load('wells.ipf')
    imod.ipf.save('wells-out.ipf', df)

    # get all calculated heads in a xarray DataArray
    # with dimensions time, layer, y, x
    da = imod.idf.load('path/to/results/head_*.idf')

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
