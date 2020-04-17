from setuptools import find_packages, setup

with open("README.rst") as f:
    long_description = f.read()

setup(
    name="imod",
    description="Make massive MODFLOW models",
    long_description=long_description,
    url="https://gitlab.com/deltares/imod/imod-python",
    author="Martijn Visser",
    author_email="martijn.visser@deltares.nl",
    license="MIT",
    packages=find_packages(),
    package_dir={"imod": "imod"},
    package_data={"imod": ["templates/*.j2", "templates/mf6/*.j2"]},
    test_suite="tests",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    python_requires=">=3.6",
    install_requires=[
        "affine",
        "dask",
        "cftime>=1",
        "cytoolz",  # optional dask dependency we need
        "Jinja2",
        "joblib",
        "matplotlib",
        "numba",
        "numpy",
        "pandas",
        "tqdm",
        "scipy",
        "toolz",  # optional dask dependency we need
        "xarray>=0.11",
    ],
    extras_require={
        "dev": [
            "black",
            "flopy",
            "nbstripout",
            "pytest",
            "pytest-cov",
            "pytest-benchmark",
            "sphinx",
            "sphinx_rtd_theme",
        ],
        "optional": ["geopandas", "pyvista", "rasterio>=1", "zarr", "bottleneck"],
    },
    classifiers=[
        # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Hydrology",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    keywords="imod modflow groundwater modeling",
)
