from setuptools import setup
import versioneer

with open("README.rst") as f:
    long_description = f.read()

setup(
    name="imod",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Work with iMOD MODFLOW models",
    long_description=long_description,
    url="https://gitlab.com/deltares/imod/imod-python",
    author="Martijn Visser",
    author_email="martijn.visser@deltares.nl",
    license="MIT",
    packages=["imod"],
    package_dir={"imod": "imod"},
    package_data={"imod": ["templates/*.j2"]},
    test_suite="tests",
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "xarray>=0.11",
        "cftime",
        "pandas",
        "dask",
        "cytoolz",  # optional dask dependency we need
        "toolz",  # optional dask dependency we need
        "affine",
        "Jinja2",
    ],
    extras_require={
        "dev": ["pytest", "sphinx", "sphinx_rtd_theme", "nbstripout"],
        "optional": ["rasterio>=1"],
    },
    classifiers=[
        # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    keywords="imod modflow groundwater modeling",
)
