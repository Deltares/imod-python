Docker build image
------------------

The CI build steps are run in a docker container. Running the build steps inside a container has several advantages. One of them is the ability to create reproducible builds.
This is done by defining the the docker image and tag to be used by the pipeline inside the pipeline files. If in the future we want to build an old commit then we exactly know which docker image and tag we should use.
Furthermore we can also easily debug a failing build by running the container locally on our machine.


Dockerfile
~~~~~~~~~~
The dockerfile can be found at `.teamcity\\Dockerfile\\Dockerfile` and is displayed below as well.

.. literalinclude:: ../../.teamcity/Dockerfile/Dockerfile
    :language: docker


.. _How to build:

How to build
------------
A prerequisite of building the image is that you have `Docker Desktop`_ installed on your machine.
Once installed open a console and navigate to the folder where the dockerfile resides.

To build the image:

.. code-block:: console

  docker context use desktop-windows  
  docker build -t windows-pixi:v0.34.0 . -m 2GB  

The image tag is not randomly chosen. It matches the Pixi version shipped with the image. So make sure the tag equals the version inside the dockerfile.

Updating the Pixi version
-------------------------
Updating the Pixi version is easy. In the dockerfile there is an argument called ``PIXI_VERSION``. Change this and follow the steps at `How to build`_ to rebuild the image. Don't forget to add the correct tag


Pushing the image
-----------------
Once build you can upload it to the Deltares repository. To do this you do need to have the required credentials.


The first step is to connect to the docker registry. You only have to do this once. The settings thereafter will be stored on your machine.
To connect you need your email address and your cli secret. The secret can be found `here`_  by clicking on your username in the top right, and then selecting your user profile.


To connect:

.. code-block:: console

  docker login -u <<deltares_email>> -p <<cli_secret>> https://containers.deltares.nl

After you connected to the registry you can tag and push your image:

.. code-block:: console

    docker tag windows-pixi:v0.34.0 containers.deltares.nl/hydrology_product_line_imod/windows-pixi:v0.34.0
    docker push containers.deltares.nl/hydrology_product_line_imod/windows-pixi:v0.34.0

Again, make sure the tags match the ``PIXI_VERSION`` inside the dockerfile.


.. _Docker Desktop: https://www.docker.com/products/docker-desktop/
.. _here: https://containers.deltares.nl