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