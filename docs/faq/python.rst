.. _python_packages:

Installing Python and packages
==============================

A bit of background
-------------------

Here's a an overview of a few relevant terms:

* Interpreter: pragmatically defined, an interpreter is a computer program that
  executes instructions interactively.

    * The ``python.exe`` you find in your Python installation is an interpreter.
      It will run commands one by one.
    * Interpreted programming languages are often contrasted against compiled
      programming languages.
    * Python is an interpreted language, as is R, for example.

* Compilation: the act of translating a program from source code (human-readable
  text form) to machine code.

    * Machine code is the set of low level instructions that a processor
      understands.
    * A compiled program doesn't run line by line, but rather builds to entire
      program first, and then be called to execute. We're greatly simplifying,
      but the general idea is: building the entire program at once allows for
      optimization. This means that compiled languages are generally (much)
      faster than interpreted languages.
    * C, C++, and Fortran are examples of compiled languages.
    * The most popular Python interpreter, CPython, is a compiled program
      written in C!
    * Depending on operating system (Linux, MacOS, Windows) and processor
      architecture(e.g. x86, ARM; 32 bit versus 64 bit), different machine code
      has to be generated.

And a few notes specific to Python:

* A Python installation includes the Python interpreter, and the Python
  standard library: modules to deal with dates, calendars, mathematics, file
  handling, etc.
* Package: a collection of files that provide a certain set of
  functionality. For example: the ``numpy`` package for working with large,
  multi-dimension arrays and matrices.
* A Python package generally consists of several modules. These are generally
  the invididual ``.py`` files.
* In terms of files: packages can exist out of single module (``.py`` file), or
  a set of directories "marked" with a ``__init__.py`` file.

Generally, we would like to write the entire program in one language. Due to
slowness of interpreted languages, this has not been feasible within Python;
especially for technical computing since we typically want to crunch large
amounts of numbers. The solution people have come up with is to write the
performance critical parts in a compiled language like C or Fortran, and then
intermittently call these programs from Python. This solves the performance
problem, but at a price: while the package previously consisted of simply a
set of Python modules (``.py`` text files), it now includes compiled (binary)
files, that are specific to your operating system and processor architecture.
Not surprisingly, these are more complicated to distribute.


Dependency conflicts
--------------------

Multiple packages often share dependencies, but they might depend on
different, incompatible, versions. For example, package ``B`` and ``C`` might
both rely on package ``A``; but ``B`` relies on version ``A.1``, and ``C``
relies on version ``A.2``.

Typically, you might succeed in installing a version of ``B`` and ``C`` that
use the same version of ``A``, but you'll find that suddenly package ``D``
(which also depends on a version of ``A``) will no longer work. This state of
affairs is colloquially called `"dependency hell"`_.


Installing Python packages -- `without the agonizing pain`_
-----------------------------------------------------------

We highly recommend installing packages using ``pixi`` or alternatively ``conda``

Pixi and conda are package- and environment-managers that install packages from
a remote repository (which is a remote storage location of software packages).
Pip (acronym for "Pip install packages") can also be used for installing Python
packages, but was designed mainly to install pure Python packages, without
binary dependencies; trying to ``pip install`` packages with complex depencies
is therefore a recipe for frustration and disaster.

Pixi and conda, do several things. First and foremost, they solve the dependency
problem when installing a package. Secondly, they also install binary
dependencies. Thirdly, they provides isolated Python installations (termed
environments). You might create a new environment if you have unsatisfiable
version requirements, like two versions of Python (e.g. 2.7 and 3.9).

Additionally, pixi allows you to store an environment in a specific state in a
textfile, allowing you to share envrionments with colleagues. Conda and pip have
similar functionality, but this is by far not as robust as pixi implemented
this.

Some packages cannot be installed by conda/pixi because they are not available
on the conda channels. In that case, with pixi you can install these from PyPI
(Python Package Index) as follows::

  pixi add <package_name> --pypi

With conda you can fall back on ``pip`` to install the package:: 

  pip install <package_name>


Find the articles: `Let's stop dependency hell`_, `Understanding conda and pip`_
and Conda:`Myths and Misconceptions`_ for additional information. See also the
`pixi docs`_.


Setting up an environment
-------------------------

At some point you will run into a dependency issue. Sometimes the dependency
requirements of two packages are straight out unsatisfiable. In other cases,
you'd like to use the latest version, but this would break other packages.
The solution used by imod-python is using Pixi. Pixi creates a complete Python
installation with all necessary dependencies. Creating a new environment will
result in a new Python installation, without sharing of dependencies with other
environments. (This is hardly the most efficient use of your hard disk space
from a theoretical perspective, but it greatly simplifies matters in the
practical sense.)

To create a Pixi environment open a command prompt and ``cd`` to the imod folder.
In the imod folder use the following command::

    pixi run install

This will create a conda environment inside the imod folder. To activate the environment run::

    pixi shell


This essentially temporarily updates your `PATH variable`_, which is the set
of directories where executable programs are located. closing the command prompt,
these directories are removed from PATH again so that the Python installation is
properly isolated. Read more at the full `pixi docs`_.

Use the following command to have a pixi environment with ipython, ipykernel,
and jupyter for interactive scripting (e.g. plotting data with matplotlib)::

    pixi run --environment interactive install

Then start the interactive environment::

    pixi shell --environment interactive


Installing a development version of iMOD Python
-----------------------------------------------

Since we're currently in the process of adding a lot of features, the version
on PyPI or conda-forge doesn't always install the carry the latest updates.
To get the latest version, activate the environment, clone the reposistory to
a repository of choice, and do a "development install"::

    activate imod
    git clone https://github.com/Deltares/imod-python.git
    cd imod-python
    pip install -e .

To get the latest developments at a later point in time, execute within the
`imod-python` directory::

    git pull
    
Past versions can also be found on the iMOD Python `releases page`_.


Anaconda, Miniconda, Miniforge
------------------------------

Ananaconda and Miniconda both provide a Python installation and conda as the
package manager. The difference between them is that Anaconda comes with a
large number of packages pre-installed in the base environment (which is why
the installation is over a gigabyte). Miniconda, on the other hand, comes
bare bones. Since we recommend working from environments to install packages
into (see below), we do not consider the full Anaconda installer attractive.

.. note::

  Since April 2020, Anaconda has `changed`_ their `Terms of Service`_, limiting
  use of the anaconda repository to commercial users. If you worry you fall in
  the category of commercial users, we recommend installing Miniforge instead.
  The change has negligible consequences for imod users, as we've long fully
  relied on the community led `conda-forge`_ channel.

You can find the Miniforge homepage and the Miniforge installers here:

* https://github.com/conda-forge/miniforge

You can find installers for Miniconda here:

* https://conda.io/miniconda.html

.. _"dependency hell": https://en.wikipedia.org/wiki/Dependency_hell
.. _let's stop dependency hell: https://prefix.dev/blog/launching_pixi
.. _without the agonizing pain: https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.110.418>
.. _Understanding conda and pip: https://www.anaconda.com/understanding-conda-and-pip
.. _Myths and Misconceptions: https://jakevdp.github.io/blog/2016/08/25/conda-myths-and-misconceptions/
.. _changed: https://www.anaconda.com/blog/sustaining-our-stewardship-of-the-open-source-data-science-community
.. _Terms of Service: https://www.anaconda.com/terms-of-service
.. _conda-forge: https://conda-forge.org/
.. _PATH variable: https://en.wikipedia.org/wiki/PATH_(variable)
.. _pixi docs: https://pixi.sh/
.. _releases page: https://github.com/Deltares/imod-python/releases
