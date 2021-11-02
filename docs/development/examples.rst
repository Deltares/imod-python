Developing examples
===================

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

.. _Sphinx-Gallery: https://sphinx-gallery.github.io/stable/index.html
.. _nbstripout: https://github.com/kynan/nbstripout
.. _Not everybody likes Jupyter notebooks: https://www.youtube.com/watch?v=7jiPeIFXb6U 
.. _reStructured Text (rST): https://en.wikipedia.org/wiki/ReStructuredText
.. _Markdown: https://en.wikipedia.org/wiki/Markdown
.. _intersphinx: https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
.. _embedded: https://sphinx-gallery.github.io/stable/syntax.html#embedding-rst
.. _IPython Sphinx Directives: https://ipython.readthedocs.io/en/stable/sphinxext.html
