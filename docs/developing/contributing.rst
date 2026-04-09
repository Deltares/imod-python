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

.. _the extensive manual online: https://git-scm.com/doc
.. _handbook: https://guides.github.com/introduction/git-handbook/
.. _GitHub help: https://help.github.com/en
.. _cheatsheet: https://github.github.com/training-kit/downloads/github-git-cheat-sheet/
.. _interactive tutorial: https://learngitbranching.js.org/
.. _here: https://docs.astral.sh/ruff/integrations/
.. _Format On Save: https://code.visualstudio.com/updates/v1_6#_format-on-save
.. _pytest documentation: https://docs.pytest.org/en/latest/
.. _ruff: https://github.com/astral-sh/ruff