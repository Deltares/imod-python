# Contributing Guidelines

We'd like this to be a community driven project, so all kinds of input are welcome!

There are numerous way you could contribute:
* Report bugs by submitting issues
* Request features by submitting issues
* Write examples and improve documentation
* Contribute code: bug fixes, new features

This document is loosely based on the [Contributing to xarray guide](https://xarray.pydata.org/en/latest/contributing.html). It's worth reading, it covers many of the subjects below in greater detail.

## Reporting bugs
You can report bugs on the *Issues*
[pages](https://gitlab.com/deltares/imod/imod-python/issues). Please include
a self-contained Python snippet that reproduces the problem. In the majority
of cases, a Minimal, Complete, and Verifiable Example (MCVE) or Minimum Workin Example (MWE) is the best way
to communicate the problem and provide insight. Have a look at [this stackoverflow
article](https://stackoverflow.com/help/mcve) for an in-depth description.

## Contributing Code
### Version control

We use Git for version control. Git is excellent software, but it might
take some to wrap your head around it. There are many excellent resources
online. Have a look at [the extensive manual
online](https://git-scm.com/doc), a shorter
[handbook](https://guides.github.com/introduction/git-handbook/), searchable
[GitHub help](https://help.github.com/en), a
[cheatsheet](https://github.github.com/training-kit/downloads/github-git-cheat-sheet/),
or try this [interactive tutorial](https://learngitbranching.js.org/).

### Code style
We use [Black](https://github.com/ambv/black) for automatic code formatting.
Like *Black*, we are uncompromising about formatting. Continuous Integration
**will fail** if your code has not been "blackened".

Integration black into your workflow is easy. Find the instructions
[here](https://github.com/ambv/black#editor-integration). If you're using
VisualStudioCode (which we heartily recommend), consider enabling the [Format
On Save](https://code.visualstudio.com/updates/v1_6#_format-on-save) option
-- it'll save a lot of hassle.

### Automated testing
If you add functionality or fix a bug, always add a test. For a new feature,
you're testing anyway to see if it works... you might as well clean it up and
include it in the test suite! In case of a bug, it means our test coverage is
insufficient. Apart from fixing the bug, also include a test that addresses
the bug so it can't happen again in the future.

We use `pytest` to do automated testing. You can run the test suite locally
by simply calling `pytest` in the project directory. `pytest` will pick up on
all tests and run them automatically. Check the [pytest
documentation](https://docs.pytest.org/en/latest/), and have a look at the
test suite to figure out how it works.

### Code review
Create a branch, and send a merge or pull request. Your code doesn't have to be perfect! We'll have a look, and
will probably suggest some modifications or ask for some clarifications.