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