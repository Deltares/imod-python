# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2019-02-26

### Removed
- Reading IDFs with the `chunks` option

### Deprecated
- Reading IDFs with the `memmap` option
- `imod.idf.dataarray`, use `imod.idf.load` instead

### Changed
- Reading IDFs gives delayed objects, which are only read on demand by dask
- IDF: instead of `res` and `transform` attributes, use `dx` and `dy` coordinates (0D or 1D)
- Use `cftime.DatetimeProlepticGregorian` to support time instead of `np.datetime64`, allowing longer timespans
- Repository moved from https://gitlab.com/deltares/imod-python/ to https://gitlab.com/deltares/imod/imod-python/

### Added
- Notebook in `examples` folder for synthetic model example
- Support for nonequidistant IDF files, by adding `dx` and `dy` coordinates

### Fixed
- IPF support implicit `itype`, defaults to timeseries

## [0.4.3] - 2018-11-13
### Fixed
- README.rst formatting only

## [0.4.2] - 2018-11-13
### Fixed
- Lower column names when writing IPF files
### Changed
- Forces dx = dy when nrow or ncol == 1, so iMODFLOW can run 2D models

## [0.4.1] - 2018-11-07
### Fixed
- Include templates as package data for pip install
### Changed
- Accept a time anywhere in the filename

## [0.4.0] - 2018-10-04
### Fixed
- IPF files with associated IPF for itype 3
- IPF quote column names that contain commas
- Fixed saving DataArrays with single timestep to IDF
### Changed
- Nodata values can be set with `nodata` keyword, default is 1.0e20. The xarray default nodata value NaN caused problems too often when used in models.
- IPF column names are no longer lower cased
### Added
- Support timestamps in filenames without hour/minute/second indication
- Support creating iMODFLOW runfiles & models from a set of DataArrays and DataFrames, see `examples/iMODSEAWAT_HenryCase.ipynb`
- `imod.rasterio.resample` function that uses rasterio to resample DataArrays
