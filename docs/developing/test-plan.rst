iMOD Python test plan
=====================

This document describes how and when to perform tests for iMOD Python.

Known shortcomings and issues can be documented `here
<https://deltares.github.io/imod-python/faq/known-issues.html>`_ . Bugs can be
reported on `GitHub <https://github.com/Deltares/imod-python/issues>`_ .

Functional tests
----------------

The functional tests are run as part of the CI pipeline, and consist of
unittests and short examples which will be contained in the documentation. They
can be run locally by running the following commands in the root of the
repository:

.. code-block:: console

  pixi run tests
  pixi run examples

These tests are run automatically on every push to the repository, and on every
pull request. They will therefore presumably work as they are run regularly, but
a final check can't harm anyway.

Some extra edge cases are not covered by the CI, and these should be
checked manually before a release. These are:

- Installing iMOD Python and running tests in a folder with curly braces ``{}``
  in the name, like is unavoidable on the Windows Computational Facilities
  (WCF). 

  .. code-block:: console
  
    mkdir "{WCF}"
    cd "{WCF}"
    git clone https://github.com/Deltares/imod-python.git
    cd imod-python
    pixi run tests

- Installing iMOD Python with the iMODforge distribution on a machine
  disconnected from the internet, like Deltares' LHM server. Examples that use
  the ``imod.data`` module here will not work, as these require an internet
  connection to download the data. Therefore we'll run test an example that
  doesn't require internet connection. Install the `iMODforge distribution
  <https://deltares.github.io/iMOD-Documentation/deltaforge_install.html>`_
  Mext, open the ``iMODforge prompt``. Copy `the TWRI example
  <https://github.com/Deltares/imod-python/blob/master/examples/mf6/ex01_twri.py>`_
  and run the following command to test if the example works:

  .. code-block:: console

    python ``ex01_twri.py``

- Installing iMOD Python on Linux. This is less commonly used by iMOD Python
  users and developers, therefore might lead to missed edge cases. Call the
  following and see if the tests run without errors:

  .. code-block:: console

    git clone https://github.com/Deltares/imod-python.git
    cd imod-python
    pixi run tests

If these edge cases fail, they should be documented in the
:doc:`../faq/known-issues` page with a workaround (if possible), and an issue
should be opened on `GitHub <https://github.com/Deltares/imod-python/issues>`_ .


User acceptance tests
---------------------

The user acceptance tests are not run as part of the CI pipeline, because they
are are either too slow or require too much manual input to be part of our CI.
These therefore require extra attention before a release.

Performance tests
*****************

These are stress tests that tests the capabilities of iMOD Python in dealing
with large groundwater models. The model currently used for the performance
tests is `the LHM model <https://nhi.nu/modellen/lhm/>`_, but more models might
be added in the future.

Run the performance tests locally on a Windows machine by following these steps:

1. First contact imod.support@deltares.nl and ask for an access key to access
   the iMOD Python test data. They will contact you and send you a key. Make
   sure you don't share this key with others!
2. Activate the user acceptance environment by running the following command in the root
   of the repository:
  
  .. code-block:: console
    
    pixi shell -e user-acceptance

3. Add your key to the DVC configuration by running the following command in the root
   of the repository:

  .. code-block:: console

    dvc remote modify --local minio access_key_id <your_access_key>
    dvc remote modify --local minio secret_access_key <your_secret_access_key>

  Don't forget the ``--local`` flag, as this will store the key in the
  ``.dvc/config.local`` file, which is not committed to the repository.

4. Pull the data from the DVC remote by running the following command in the root
   of the repository:

  .. code-block:: console

    pixi run fetch_lhm

  This will unpack the LHM model data, which is used in the user acceptance
  tests.

5. Run the user acceptance tests by running the following command in the root 
   of the repository:

  .. code-block:: console

    pixi run user_acceptance

  This will write the MODFLOW6 input files to the
  ``imod/tests/user_acceptance_data/mf6_imod-python`` folder and the MetaSWAP
  files to ``imod/tests/user_acceptance_data/msp_imod-python``.

6. Run the iMOD5 conversion which is the reference by running the following
   command in the root of the repository. This needs to be run on a Windows
   machine.

   .. code-block:: console

     pixi run run_imod5

  This will write the MODFLOW6 and MetaSWAP input files to the
  ``imod/tests/user_acceptance_data/MF6-MSP_IMOD-5`` folder.

Criteria for user acceptance tests of the 1.0 release are:

* The tests should run without errors.
* The tests should run without warnings from iMOD Python, unless unavoidable.
* The conversion of the transient LHM model run of 1 year on a daily timestep
  (365 stress-periods) should run without memory overflow on a machine with 32
  GB and write a model within 15 minutes.
* The MODFLOW6 and MetaSWAP input files written by iMOD Python should be the
  same as iMOD5 (accounting for differences in row sorting.), unless there was a
  conscious decision to divert from this. These will be mentioned in
  :doc:`../faq/imod5_backwards_compatibility`.
* The conversion of the transient LHM model should not be slower than doing the
  same conversion with iMOD5.

Manual checks
*************

QGIS export
^^^^^^^^^^^

1. Run the pixi task written: 

   .. code-block:: console

     pixi run export_qgis

   This will export a simulation to a TOML file and a set of UGRID netCDFs twice,
   once for a model with a structured grid, once for a model with an unstructured
   grid. The location of the exported files will be printed in the terminal.
2. `Download the latest version of QGIS <https://qgis.org/download/>`_.
3. Open QGIS.
4. Set the coordinate reference system (CRS) of the project to EPSG:28992, the
   same CRS as the exported files.
5. Click ``"Layers" > "Add Layer" > "Add mesh"``. Insert the path printed in the
   terminal in the text box. ``{path_printed_in_terminal}/hondsrug_MDAL/riv.nc``
   This will import the mesh. 
6. Verify if the mesh is rendered in two dimensions, and not as a single
   line of cells. If not, `open an issue on GitHub
   <https://github.com/Deltares/imod-python/issues>`_ . 

Tutorial
^^^^^^^^

1. `Open the tutorial material here
   <https://deltares.github.io/iMOD-Documentation/tutorial_Hondsrug.html/>`_ .
2. Run each jupyter notebook and assure it runs without errors.
3. If there are any errors, open an issue on `iMOD Documentation repository
   Github <https://github.com/Deltares/iMOD-Documentation/issues>`_ .

Documentation
*************

Build the documentation locally by running the following command in the root of
the repository:

.. code-block:: console

  pixi run docs

Check if the documentation builds without errors and warnings. If there are
errors or warnings, fix them before releasing in `a pull request on Github
<https://github.com/Deltares/imod-python/pulls>`_ . Next, check if the
documentation pages are rendered correctly and if the information on them is not
outdated. You can do this by opening the HTML files in the ``docs/_build/html``.
Focus on the following pages for the 1.0 release:

- The `Install documentation <https://deltares.github.io/imod-python/installation/>`_
- The `iMOD Python API documentation
  <https://deltares.github.io/imod-python/api/>`_, focus on whether all classes,
  methods, and functions that are part of the public API are documented.
- The `iMOD5 Backwards compatibility documentation <faq/imod5_backwards_compatibility.html>`_
