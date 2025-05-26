iMOD Python test plan
=====================

This document describes how and when to perform tests for iMOD Python.

Known shortcomings and issues can be documented `here
<https://deltares.github.io/imod-python/faq/known-issues.html>`_ . Bugs can be
reported on `GitHub <https://github.com/Deltares/imod-python/issues>`_ .

Functional tests
----------------

The functional tests are run as part of the CI pipeline, and are short models.
They can be run locally by running the following command in the root of the
repository:

```bash
pixi run tests
```

These tests are run on every push to the repository, and on every pull request. T

User acceptance tests
---------------------

The user acceptance tests are not run as part of the CI pipeline. This is a
performance stress test that tests the capabilities of iMOD Python in dealing
with large groundwater models. These tests are either too slow or require to
much manual input to be part of our CI. The model currently used for the user
acceptance tests is `the LHM model <https://nhi.nu/modellen/lhm/>`_, but more
models might be added in the future.

Run the user acceptance tests locally by following these steps:

  1. Download the test data from <...insert...link...here...> and put it in a
     directory of your liking.
  2. Assign the environment variable ``LHM_PRJ`` to the path of the projectfile
     of the test data. You can put this in the ``.env`` file in the root of the
     repository, or set it in your shell.
  3. Run the user acceptance tests by running the following command in the root 
     of the repository:

```bash
pixi run user_acceptance
```

Criteria for user acceptance tests of the 1.0 release are:

* The tests should run without errors.
* The MODFLOW6 and MetaSWAP input files written by iMOD Python should be the
  same as iMOD5 (accounting for differences in sorting.), unless there was a
  conscious decision to divert from this.
* The conversion from projectfile to MODFLOW6 and MetaSWAP input files should be
  done in a reasonable amount of time and should not be much slower than iMOD5.
  This is subjective, but we aim for less than 5 minutes for the LHM model with
  1 timestep.
* Are there large differences in memory usage compared to iMOD5? 

