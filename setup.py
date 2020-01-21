from setuptools import setup, find_packages

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
        "numba",
        "numpy",
        "scipy",
        "matplotlib",
        "xarray>=0.11",
        "cftime>=1",
        "pandas",
        "dask",
        "cytoolz",  # optional dask dependency we need
        "toolz",  # optional dask dependency we need
        "affine",
        "Jinja2",
        "joblib",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "pytest-benchmark",
            "flopy",
            "sphinx",
            "sphinx_rtd_theme",
            "nbstripout",
            "black",
        ],
        "optional": ["rasterio>=1", "geopandas"],
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
