Contributing Guidelines
=======================

We'd like this to be a community driven project, so all kinds of input
are welcome!

There are numerous way you could contribute:

-  Report bugs by submitting issues
-  Request features by submitting issues
-  Write examples and improve documentation
-  Contribute code: bug fixes, new features

This document is loosely based on the `Contributing to xarray guide`_.
It's worth reading, it covers many of the subjects below in greater
detail.

Reporting bugs
--------------

You can report bugs on the *Issues* `pages`_. Please include a
self-contained Python snippet that reproduces the problem. In the
majority of cases, a Minimal, Complete, and Verifiable Example (MCVE) or
Minimum Working Example (MWE) is the best way to communicate the problem
and provide insight. Have a look at `this stackoverflow article`_ for an
in-depth description.

Contributing Code
-----------------

Version control
~~~~~~~~~~~~~~~

We use Git for version control. Git is excellent software, but it might
take some time to wrap your head around it. There are many excellent
resources online. Have a look at `the extensive manual online`_, a
shorter `handbook`_, searchable `GitHub help`_, a `cheatsheet`_, or try
this `interactive tutorial`_.

Code style
~~~~~~~~~~

We use `Black`_ for automatic code formatting. Like *Black*, we are
uncompromising about formatting. Continuous Integration **will fail** if
running ``black .`` from within the repository root folder would make
any formatting changes.

Integration black into your workflow is easy. Find the instructions
`here`_. If you're using VisualStudioCode (which we heartily recommend),
consider enabling the `Format On Save`_ option -- it'll save a lot of
hassle.

Automated testing
~~~~~~~~~~~~~~~~~

If you add functionality or fix a bug, always add a test. For a new
feature, you're testing anyway to see if it works... you might as well
clean it up and include it in the test suite! In case of a bug, it means
our test coverage is insufficient. Apart from fixing the bug, also
include a test that addresses the bug so it can't happen again in the
future.

We use ``pytest`` to do automated testing. You can run the test suite
locally by simply calling ``pytest`` in the project directory.
``pytest`` will pick up on all tests and run them automatically. Check
the `pytest documentation`_, and have a look at the test suite to figure
out how it works.

Automation: tox
~~~~~~~~~~~~~~~

The manual steps of checking and formatting according to code style as well as
testing can be automated. We use ``tox`` to automate these steps. Tox is
configured for three steps:

* ``tox -e format``: run ``isort`` and ``black`` on all Python modules and 
  format them if needed.
* ``tox -e lint``: check code with ``isort``, ``black``, and ``flake8``.
* ``tox -e build``: run the tests and build the Sphinx documentation.
  
Note that tox builds a new environment for every step. The environments for
formatting and linting are small, but the build environment requires a full
conda installation. The configuration is setup to use mamba, so make sure
tox and mamba are installed before running tox.

Code review
~~~~~~~~~~~

Create a branch, and send a merge or pull request. Your code doesn't have to be
perfect! We'll have a look, and we will probably suggest some modifications or
ask for some clarifications.

How to release a new version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To follow these steps, you need to be one of the maintainers for imod on both
`PyPI <https://pypi.org/project/imod/>`_ and `conda-forge
<https://github.com/conda-forge/imod-feedstock>`_.

1. Update the :doc:`../api/changelog`.

2. Tag in Gitlab UI. `Old tags are here
   <https://gitlab.com/deltares/imod/imod-python/-/tags>`_. `Old releases are
   here <https://gitlab.com/deltares/imod/imod-python/-/releases>`_. To make a
   tag show up under releases, fill in the release notes in the UI. Since we
   keep changes in the :doc:`../api/changelog` only, just put ``See
   https://imod.xyz/changelog.html`` in both the ``Message`` and ``Release
   notes`` box. The tag name should be ``vx.y.z``, where x, y and z are version
   numbers according to `Semantic Versioning <https://semver.org/>`_.

3. Locally, ``git fetch --tags`` and ``git pull``, verify you are on the commit
   you want to release, and that it is clean.

4. Remove the ``build`` and ``dist`` folders if present.

5. Create a source distribution under ``dist/`` with ``python -m build --sdist``

6. Upload the files from step 5 and 6 to PyPI with ``twine upload dist/*``

7. For `conda-forge <https://github.com/conda-forge/imod-feedstock>`_, a PR
   will be created automatically. If the requirements are up to date in
   `meta.yaml
   <https://github.com/conda-forge/imod-feedstock/blob/master/recipe/meta.yaml>`_
   then you can merge it. Otherwise you have to edit them and push this to the
   bot's branch.


.. _Contributing to xarray guide: https://xarray.pydata.org/en/latest/contributing.html
.. _pages: https://gitlab.com/deltares/imod/imod-python/issues
.. _this stackoverflow article: https://stackoverflow.com/help/mcve
.. _the extensive manual online: https://git-scm.com/doc
.. _handbook: https://guides.github.com/introduction/git-handbook/
.. _GitHub help: https://help.github.com/en
.. _cheatsheet: https://github.github.com/training-kit/downloads/github-git-cheat-sheet/
.. _interactive tutorial: https://learngitbranching.js.org/
.. _Black: https://github.com/ambv/black
.. _here: https://github.com/ambv/black#editor-integration
.. _Format On Save: https://code.visualstudio.com/updates/v1_6#_format-on-save
.. _pytest documentation: https://docs.pytest.org/en/latest/
.. _tox: https://tox.wiki/en/latest/index.html


Debugging Continuous Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Continuous Integration runs on an image with a specific operating system, and
Python installation. Due to system idiosyncrasies, CI failing might not
reproduce locally. If an issue requires more than trial-and-error changes,
Docker is likely the easiest way to debug.

On windows, install Docker:
https://docs.docker.com/docker-for-windows/install/

Pull the CI image (at the time of writing), and run it interactively:

.. code-block:: console

  docker pull continuumio/miniconda3:latest
  docker run -it continuumio/miniconda3

This should land you in the docker image. Next, we reproduce the CI setup steps.
Some changes are required, such as installing git and cloning the repository,
which happens automatically within CI.

.. code-block:: console

  apt-get update -q -y
  apt-get install -y build-essential
  conda update -n base conda
  conda install git
  cd /usr/src
  git clone https://gitlab.com/deltares/imod/imod-python.git
  cd imod-python
  conda env create -f environment.yml
  source activate imod
  pip install -e .
  curl -O -L https://gitlab.com/deltares/imod/imod-python/uploads/947a1e194a02ade1376d1111327db34d/mf6.gz
  gunzip mf6.gz
  chmod +x mf6
  mv mf6 /opt/conda/envs/imod/bin

At this point, everything should be ready to run the tests on the Docker image.