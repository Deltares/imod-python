from setuptools import setup

with open("README.rst") as f:
    long_description = f.read()

setup(
    name="imod",
    version="0.2.0",
    description="Work with iMOD MODFLOW models",
    long_description=long_description,
    url="https://gitlab.com/deltares/imod-python",
    author="Martijn Visser",
    author_email="mgvisser@gmail.com",
    license="MIT",
    packages=["imod"],
    test_suite="tests",
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "xarray>=0.10",
        "pandas",
        "dask",
        "cytoolz",  # optional dask dependency we need
        "toolz",  # optional dask dependency we need
        "affine",
    ],
    extras_require={"dev": ["sphinx", "sphinx_rtd_theme"], "optional": ["rasterio>=1"]},
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
