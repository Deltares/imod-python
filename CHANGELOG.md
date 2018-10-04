# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
