===============
Getting Started
===============

.. toctree::
    :maxdepth: 1
    :hidden: 

    installation

Install the latest release using ``conda install -c conda-forge imod``, or, when
not using Anaconda, ``pip install imod``. For more detailed installation
information see :doc:`installation`.

.. code:: python

   import imod

   # read and write IPF files to pandas DataFrame
   df = imod.ipf.read('wells.ipf')
   imod.ipf.save('wells-out.ipf', df)

   # get all calculated heads in a xarray DataArray
   # with dimensions time, layer, y, x
   da = imod.idf.open('path/to/results/head_*.idf')

   # create a groundwater model
   # abridged example, see examples for the full code
   gwf_model = imod.mf6.GroundwaterFlowModel()
   gwf_model["dis"] = imod.mf6.StructuredDiscretization(
       top=200.0, bottom=bottom, idomain=idomain
   )
   gwf_model["chd"] = imod.mf6.ConstantHead(
       head, print_input=True, print_flows=True, save_flows=True
   )
   simulation = imod.mf6.Modflow6Simulation("ex01-twri")
   simulation["GWF_1"] = gwf_model
   simulation.time_discretization(times=["2000-01-01", "2000-01-02"])
   simulation.write(modeldir)
