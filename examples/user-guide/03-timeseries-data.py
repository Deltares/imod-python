"""
Time series data and Pandas
===========================

In groundwatermodelling handeling timeseries is a major part of analysing 
the model output. Timeseries comes in different forms:

* time series linked to a point location, for example a
  measured groundwater level on a specific location. 
* time series of spactial distributed data, for example model output of
  calcuated head for a specific model layer. 

For timeseries imod Pyhton package make use of the pandas data structure.
Pandas is a popular package that makes analysis and processing of tabular 
data easy, and provides many input and output options, which in turn enables 
us to convert for instance existing CSV or Excel files to IPF files.

Pandas contains extensive capabilities and features for working with time 
series data. Using pandas an extensive libary can be used for time series data 
visualisation, manipulating reading and saving data. 

Timeseries on point locations
-----------------------------
In the imod Python package these files are read in as `pandas.DataFrame`_. 
Existing IPF file with time series data can be read using 
:meth:`imod.ipf.read` 

Let's grab a iMOD 5 IPF, store them in a temporarily directory, and
open them:
"""
result_dir = imod.util.temporary_directory()
imod.data.ipf_output(result_dir)
imod.ipf.read('measurements.ipf')

# Select location ts_1
df[df["ID"]=='ts_1']

# Caclute the mean of each time series
df.groupby(['ID']).mean()


"""
Timeseries of spactial data
---------------------------
In the imod Pyhton package timeseries of spatial data are stored in a
'xarray.DataArray'_. Wihtin xarray a lot of the features that make working 
with time series data in pandas easy, are implemented in a similary way in 
xarray. 


"""
