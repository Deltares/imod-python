Debugging Continuous Integration
--------------------------------

When the CI pipeline fails you want to be able to reproduce it locally. There are 2 general approaches to doing this.
- Running the pipeline steps locally
- Running the pipeline steps inside the docker build container

Running the pipeline steps locally
----------------------------------

The commands run during the Continuous Integration are they same as the tasks defined
in the pixi tasks list. Therefor you can directly call these tasks on your local repository.

For example if the MyPy step fails you can locally run the command:

.. code-block:: console

   pixi run --environment default mypy_lint

Or if the Unit tests are failing you can run:

.. code-block:: console

   pixi run --environment default unittests

To full lists of tasks  can be found in the pixi.toml file or can be found by running

.. code-block:: console

   pixi task list

Running the pipeline steps inside the docker build container
------------------------------------------------------------

If you're unable to reproduce a failing build using the method above you can try running it inside the docker build container.
The docker build container is also used on the build server and therefore settings one up locally should provide you with the exact same environment.


Obtain the image
~~~~~~~~~~~~~~~~
To obtain the container you can either pull it from the `docker registry`_ registry or build it yourself.

To pull the latest image use the command below:

.. code-block:: console

   docker pull containers.deltares.nl/hydrology_product_line_imod/windows-pixi:latest

Sometimes you want to use an older version of the docker builder. To get an older version you can run the same command but use a different tag.
To find out which tags are available and which is the latest you can have a look at the `artifacts page`_.
The tag name corresponds with the pixi version inside the docker image e.g. :34.0 means that the image contains pixi v34.0.


To pull a specific version run:

.. code-block:: console

   docker pull containers.deltares.nl/hydrology_product_line_imod/windows-pixi:34.0

To build it yourself see :ref:`How to build`.

Create the container
~~~~~~~~~~~~~~~~~~~~
To spin up the container run the following command:

.. code-block:: console

   docker run --rm -it windows-pixi:latest

This commands spins up an interactive session which will be automatically removed whenever you exit it.
Your next steps should be to clone the imod-python repository, checkout the branch you want to debug and run the pixi tasks corresponding to the failing build step.

.. _docker registry: https://containers.deltares.nl
.. _artifacts page: https://containers.deltares.nl/harbor/projects/32/repositories/windows-pixi/artifacts-tab