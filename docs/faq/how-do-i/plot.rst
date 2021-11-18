
Plot a timeseries for a single cell
***********************************

.. code-block:: python

    transient_da.sel(x=x, y=y).plot()
    
Plot head of one layer at one time
**********************************

.. code-block:: python

    transient_da.sel(layer=1, time="2020-01-01").plot()
