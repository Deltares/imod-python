iMOD Python test plan
=====================

This document describes how and when to perform tests for iMOD Python.

Known shortcomings and issues can be documented `here
<https://deltares.github.io/imod-python/faq/known-issues.html>`_ . Bugs can be
reported on `GitHub <https://github.com/Deltares/imod-python/issues>`_ .

Functional tests
----------------

The functional tests are run as part of the CI pipeline, and consist of
unittests and short examples which will be containted in the documentation. They
can be run locally by running the following commands in the root of the
repository:

.. code-block:: console

  pixi run tests
  pixi run examples

These tests are run automatically on every push to the repository, and on every
pull request. They will therefore presumably work as they are run regularly, but
a final check can't harm anyway.

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

Run the performance tests locally by following these steps:

1. Download the test data from <...insert...link...here...> and put it in a
   directory of your liking.
2. Assign the environment variable ``LHM_PRJ`` to the path of the projectfile
   of the test data. You can put this in the ``.env`` file in the root of the
   repository, or set it in your shell.
3. Run the user acceptance tests by running the following command in the root 
   of the repository:

.. code-block:: console

  pixi run user_acceptance

Criteria for user acceptance tests of the 1.0 release are:

* The tests should run without errors.
* The tests should run without warnings from iMOD Python, unless unavoidable.
* The MODFLOW6 and MetaSWAP input files written by iMOD Python should be the
  same as iMOD5 (accounting for differences in row sorting.), unless there was a
  conscious decision to divert from this.
* The conversion from projectfile to MODFLOW6 and MetaSWAP input files should be
  done in a reasonable amount of time and should not be much slower than iMOD5.
  This is subjective and varies per machine, but we aim for less than 5 minutes
  for the LHM model with 1 timestep on a machine with 32 GB RAM on a single
  core.
* The conversion of the transient LHM model run of 40 years on a daily timestep
  (140K timesteps) should be possible without memory overflow.

Manual checks
*************

- Run the pixi task written below. This will export data to a UGRID NetCDF and
  save it under the name <......>. Open QGIS. Click "Layers" > "Add Layer" >
  "Add mesh". Insert the path <......> in the text box. This will import the
  mesh. Verify if the mesh is rendered properly, if not open an issue on `GitHub
  <https://github.com/Deltares/imod-python/issues>`_ .
  
.. code-block:: console

  pixi run export-qgis

- Work through the tutorial material here and verify it is up to date:
  https://deltares.github.io/iMOD-Documentation/

Documentation
*************

Build the documentation locally by running the following command in the root of
the repository:

.. code-block:: console

  pixi run docs

Check if the documentation builds without errors and warnings. If there are
errors or warnings, fix them before releasing in a pull request on `Github
<https://github.com/Deltares/imod-python/pulls>`_ . Next, check if the
documentation pages are rendered correctly and if the information on them is not
outdated. You can do this by opening the HTML files in the ``docs/_build/html``.
Focus on the following pages for the 1.0 release:

- The `Install documentation <https://deltares.github.io/imod-python/installation/>`_
- The `iMOD Python API documentation
  <https://deltares.github.io/imod-python/api/>`_, focus on whether all classes,
  methods, and functions that are part of the public API are documented.
- The `iMOD5 Backwards compatibility documentation <faq/imod5_backwards_compatibility.html>`_
