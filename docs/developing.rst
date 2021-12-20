Developing
===========

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

We use `black`_ for automatic code formatting. Like *Black*, we are
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

Automation with tox
~~~~~~~~~~~~~~~~~~~

This package uses `tox`_ to automate code formatting, linting, and testing.
tox builds an isolated environment for these steps. Because imod has a large
number of dependencies, we rely on the `tox-conda`_ plugin to build these
isolated environments with mamba.

.. note::

    Running the steps below will result in the creation of three conda
    environments in the ``.tox`` directory. The format and linting environments
    are comparatively small (140 MB), but the build environment requires a full
    installation of the imod package and all its dependencies (>2.5 GB).

To format the code with `isort`_ and `black`_, run::

    tox -e format
    
To lint the code with isort, black, and `flake8`_::

    tox -e lint
    
To run the tests and build the documentation, run::

    tox -e build
    
This will run all the tests with pytest, and builds the documentation with
sphinx. Building the documentation is performed in the same step and
environment as running the tests as building the documentation includes running
all the examples in the ``.examples`` directory, which requires a full
installation of the imod package and all its dependencies. Separating the
testing and documentation building requires building two large identical
environments.

Code review
~~~~~~~~~~~

Create a branch, and send a merge or pull request. Your code doesn't have to be
perfect! We'll have a look, and we will probably suggest some modifications or
ask for some clarifications.

How to release a new version
----------------------------

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

Developing examples
-------------------

All our examples are available as:

* As a rendered HTML gallery online
* As downloadable Python scripts or Jupyter notebooks
* As the original Python scripts in the ``./examples`` directory, which can be
  browsed directly on the online repository.

We use `Sphinx-Gallery`_ to render the Python files as HTML. We could also use
Jupyter notebooks as they are nicely rendered and executable by a user.
However, Sphinx-Gallery has a number of advantages over Jupyter notebooks:

* To render Jupyter notebooks online, cell output has to be stored in the
  notebooks. This is fine for text output, but images are stored as (inline)
  binary blobs. These result in large commits bloating the Git repository.
  Tools such as `nbstripout`_ will remove the cell outputs, but this comes at
  the obvious cost of not having the rendered notebooks available online.
* `Not everybody likes Jupyter notebooks`_ and Jupyter notebooks require
  special software to run. Python scripts can be run with just a Python
  interpreter. Furthermore, Sphinx-Gallery also provides Jupyter notebooks:
  from the Python scripts it will automatically generate them.
* Sphinx-Gallery uses `reStructured Text (rST)`_ rather than Markdown. rST
  syntax is somewhat less straightforward than `Markdown`_, but it also
  provides additional features such as easily linking to the API (including
  other projects, via `intersphinx`_).

For Sphinx-Gallery, rST is `embedded`_ as module docstrings at the start of a
scripts and as comments in between the executable Python code. We use ``# %%``
as the block splitter rather than 79 ``#``'s, as the former is recognized by
editors such as Spyder and VSCode, while the latter is not. The former also
introduces less visual noise into the examples when reading it as an unrendered
Python script.

Note that documentation that includes a large portion of executable code such
as the User Guide has been written as Python scripts with embedded rST as well,
rather than via the use of `IPython Sphinx Directives`_.

Building documentation and examples
-----------------------------------

In the ``docs`` directory, run:

.. code-block:: console

   make html
   
On Windows:

.. code-block:: console

   .\make.bat html

Sphinx will build the documentation in a few steps. This is generally useful,
as it means only part of the documentation needs to be rebuilt after some
changes. However, to start afresh, run:

.. code-block:: console

   python clean.py
   
This will get rid of all files generated by Sphinx.

Building the documentation is also part of the ``tox -e build`` step , see:
`Automation with tox`_.

Debugging Continuous Integration
--------------------------------

Continuous Integration runs on an image with a specific operating system, and
Python installation. Due to system idiosyncrasies, CI failing might not
reproduce locally. If an issue requires more than trial-and-error changes,
Docker may be the easiest way to debug.

On windows, install Docker:
https://docs.docker.com/docker-for-windows/install/

Pull the CI image (at the time of writing), and run it interactively:

.. code-block:: console

  docker pull condaforge/miniforge3:latest
  docker run -it condaforge/miniforge3

This should land you in the docker image. Next, we reproduce the CI setup steps.
Some changes are required, such as installing git and cloning the repository,
which happens automatically within CI.

.. code-block:: console

  conda install mamba tox
  apt-get update -q -y
  apt-get install -y build-essential
  conda install git
  cd /usr/src
  git clone https://gitlab.com/deltares/imod/imod-python.git
  cd imod-python
  conda env create -f imod-environment.yml
  source activate imod
  pip install -e .
  curl -O -L https://gitlab.com/deltares/imod/imod-python/uploads/a8ed27675150689c6acd425239531a5e/mf6.gz
  gunzip mf6.gz
  chmod +x mf6
  mv mf6 /opt/conda/envs/imod/bin

At this point, everything should be ready to run the tests on the Docker image.

.. _Contributing to xarray guide: https://xarray.pydata.org/en/latest/contributing.html
.. _pages: https://gitlab.com/deltares/imod/imod-python/issues
.. _this stackoverflow article: https://stackoverflow.com/help/mcve
.. _the extensive manual online: https://git-scm.com/doc
.. _handbook: https://guides.github.com/introduction/git-handbook/
.. _GitHub help: https://help.github.com/en
.. _cheatsheet: https://github.github.com/training-kit/downloads/github-git-cheat-sheet/
.. _interactive tutorial: https://learngitbranching.js.org/
.. _black: https://github.com/ambv/black
.. _here: https://github.com/ambv/black#editor-integration
.. _Format On Save: https://code.visualstudio.com/updates/v1_6#_format-on-save
.. _pytest documentation: https://docs.pytest.org/en/latest/
.. _tox: https://tox.wiki/en/latest/index.html
.. _Sphinx-Gallery: https://sphinx-gallery.github.io/stable/index.html
.. _nbstripout: https://github.com/kynan/nbstripout
.. _Not everybody likes Jupyter notebooks: https://www.youtube.com/watch?v=7jiPeIFXb6U 
.. _reStructured Text (rST): https://en.wikipedia.org/wiki/ReStructuredText
.. _Markdown: https://en.wikipedia.org/wiki/Markdown
.. _intersphinx: https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
.. _embedded: https://sphinx-gallery.github.io/stable/syntax.html#embedding-rst
.. _IPython Sphinx Directives: https://ipython.readthedocs.io/en/stable/sphinxext.html
.. _tox-conda: https://github.com/tox-dev/tox-conda
.. _isort: https://github.com/PyCQA/isort
.. _flake8: https://github.com/PyCQA/flake8
