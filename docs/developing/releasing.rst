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

The release is now available on PyPi but not yet on conda-forge. The conda-forge
bot will automatically open a PR to update `the imod feedstock.
<https://github.com/conda-forge/imod-feedstock>`_ It usually takes a few hours
before the bot opens a PR. This PR will be reviewed by the imod-feedstock
maintainers and merged if everything is in order. 

Release a pre-release
^^^^^^^^^^^^^^^^^^^^^

To release a pre-release, follow the same steps as above, but add a ``rc`` to the
version + a build number. For example: ``1.0.0rc0`` for the first release candidate.

PyPI will automatically recognize this as a pre-release, thus will not show it
as a stable build. To get the pre-release on conda-forge, you need to:

1. Fork `the imod-feedstock <https://github.com/conda-forge/imod-feedstock>`_
2. Checkout the ``rc`` branch of the imod-feedstock
3. Update the version in the ``recipe/meta.yaml`` file to the pre-release version, e.g. ``1.0.0rc0``
4. Update the sha256 checksum in the ``recipe/meta.yaml`` file, you can generate one by running:
   ``curl -sL https://pypi.io/packages/source/i/imod/imod-{{version}}.tar.gz | openssl sha256`` 
   (TIP: On Windows, you can install curl and openssl via pixi)
5. Commit and push the changes to your fork
6. Open a PR to the imod-feedstock and make sure it merges to the ``rc`` branch.
7. This will trigger a few CI jobs. When these succeed, the branch can be merged.
