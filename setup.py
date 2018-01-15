from setuptools import setup

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except ImportError:
    long_description = ''

setup(
    name='imod',
    version='0.1',
    description='Work with iMOD MODFLOW models',
    long_description=long_description,
    url='https://gitlab.com/visr/imod-python',
    author='Martijn Visser',
    author_email='mgvisser@gmail.com',
    license='MIT',
    packages=['imod'],
    test_suite='tests',
    python_requires='>=3',
    install_requires=[
        'numpy',
        'xarray',
        'pandas',
        'dask',
        'rasterio',
    ],
    classifiers=[
        # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='imod modflow groundwater modeling',
)
