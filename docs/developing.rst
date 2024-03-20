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

We use `ruff`_ for automatic code formatting. Like *ruff*, we are
uncompromising about formatting. Continuous Integration **will fail** if
running ``ruff .`` from within the repository root folder would make
any formatting changes.

Integration ruff into your workflow is easy. Find the instructions
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


Code review
~~~~~~~~~~~

Create a branch, and send a merge or pull request. Your code doesn't have to be
perfect! We'll have a look, and we will probably suggest some modifications or
ask for some clarifications.

How to release a new version
----------------------------

To follow these steps, you need to be one of the maintainers for imod on both
`PyPI <https://pypi.org/project/imod/>`_ and `conda-forge
<https://github.com/conda-forge/imod-feedstock>`_, as well as access to the
`Deltares Teamcity build environment <https://dpcbuild.deltares.nl>`_.

1. Update the :doc:`../api/changelog` and the ``__version__`` in ``imod/__init__.py``,
   and the version entry in the ``pixi.toml`` for complenetess.

2. Create a tag on your local machine and push it GitHub. `Old tags are here
   <https://github.com/Deltares/imod-python/tags>`_. `Old releases are
   here <https://github.com/Deltares/imod-python/releases>`_. The tag name should be ``vx.y.z``,
   where x, y and z are version numbers according to `Semantic Versioning
   <https://semver.org/>`_.

3. On Teamcity go to the `Deploy All
   <https://dpcbuild.deltares.nl/buildConfiguration/iMOD6_IMODPython_Windows_DeployAll?mode=builds>`_
   build step in the `Deploy` project.
4. Press the `Run` button and select the `Changes` Tab.
5. Select the branch/tag you want to release and press `Run Build`

The TeamCity pipeline will:

1. Create a release on GitHub
2. Create the imod-python package and upload it to PyPi
3. Build the documentation and deploy it

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

Creating the docs can also be done using pixi. Run the following command to build it:

.. code-block:: console

   pixi run --environment default docs

Debugging Continuous Integration
--------------------------------

The commands run during the Continuous Integration are they same as the tasks defined
in the pixi tasks list. For example if the MyPy step fails you can locally run the command:

.. code-block:: console

   pixi run --environment default mypy_lint

Or if the Unit tests are failing you can run:

.. code-block:: console

   pixi run --environment default unittests

To full lists of tasks  can be found in the pixi.toml file or can be found by running

.. code-block:: console

   pixi task list

.. _Contributing to xarray guide: https://xarray.pydata.org/en/latest/contributing.html
.. _pages: https://github.com/Deltares/imod-python/issues
.. _this stackoverflow article: https://stackoverflow.com/help/mcve
.. _the extensive manual online: https://git-scm.com/doc
.. _handbook: https://guides.github.com/introduction/git-handbook/
.. _GitHub help: https://help.github.com/en
.. _cheatsheet: https://github.github.com/training-kit/downloads/github-git-cheat-sheet/
.. _interactive tutorial: https://learngitbranching.js.org/
.. _here: https://docs.astral.sh/ruff/integrations/
.. _Format On Save: https://code.visualstudio.com/updates/v1_6#_format-on-save
.. _pytest documentation: https://docs.pytest.org/en/latest/
.. _Sphinx-Gallery: https://sphinx-gallery.github.io/stable/index.html
.. _nbstripout: https://github.com/kynan/nbstripout
.. _Not everybody likes Jupyter notebooks: https://www.youtube.com/watch?v=7jiPeIFXb6U 
.. _reStructured Text (rST): https://en.wikipedia.org/wiki/ReStructuredText
.. _Markdown: https://en.wikipedia.org/wiki/Markdown
.. _intersphinx: https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
.. _embedded: https://sphinx-gallery.github.io/stable/syntax.html#embedding-rst
.. _IPython Sphinx Directives: https://ipython.readthedocs.io/en/stable/sphinxext.html
.. _isort: https://github.com/PyCQA/isort
.. _ruff: https://github.com/astral-sh/ruff
