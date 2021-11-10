import logging
import os
import pathlib

import joblib

from imod import util


def hash_filemetadata(path):
    """
    Create a hash based on the path to the file, the size, and the modification
    time. This should be sufficiently robust.
    """
    path = pathlib.Path(path)
    status = os.stat(path)
    return joblib.hash((path.name, status.st_size, status.st_mtime))


def insert_hash(metadatahash):
    """Used only to store hash in cache."""
    pass


def output_metadata_hashes(pkg):
    """
    Generate full list of output files, get metadata of all files, and create
    a single hash.
    """
    # Generate full list of output files, check sta
    paths = pkg._outputfiles
    hashes = []

    # Temporarily change working to directory to one matching the paths as
    # produced by the render methods.
    with util.cd(pkg._reldir):
        for path in paths:
            try:
                hashes.append(hash_filemetadata(path))
            except FileNotFoundError:
                break

    return joblib.hash(hashes)


def hash_exists(func, *args):
    """
    Check whether a hash already exists in the store.

    Parameters
    ----------
    func : joblib.memory.MemorizedFunc
    *args : function arguments
    """
    func_id, args_id = func._get_output_identifiers(*args)
    return func._check_previous_func_code(
        stacklevel=4
    ) and func.store_backend.contains_item([func_id, args_id])


def check_filehashes(filehashes):
    out = []
    for key, value in filehashes.items():
        if value is None:
            raise ValueError(
                'Package "{key}" must be a CachingPackage for caching to work.'
            )
        out.append(value)
    return out


def caching(package, memory):
    """
    This function produces a special version of an input package, which avoids
    expensive computation during rendering and writing of the runfile, and
    writing of the model input if the current model input already suffices.

    Parameters
    ----------
    package : imod.wq Package.
    path : path to the fully stored netCDF with package input.
    memory : joblib memory object.

    Returns
    -------
    caching_package : CachingPackage
        imod.SeawatModel input package with caching mechanisms for the
        computationally expensive methods.
    """
    output_status = memory.cache(insert_hash)

    # Subclass package: overloads _compose, max_active_n, pkgcheck, and save methods.
    class CachingPackage(package):
        __slots__ = (
            "_filehashself",
            "_filehashes",
            "_reldir",
            "_outputfiles",
            "_caching_save",
            "_caching_check",
            "_caching_max_n",
            "_logger",
        )
        # Define methods which take the filehashes. These are the functions that
        # will be cached by joblib. We ignore pkg (self) so non-deterministic
        # dask DAG hashes within DataArrays do not lead to repeat computation.
        #
        # More methods could be added, but these are the ones drawing data into
        # memory / making passes over the data, rather than using just metadata.

        @staticmethod
        def _max_n(pkg, filehashes, varname, nlayer, nrow, ncol):
            return super(type(pkg), pkg)._max_active_n(varname, nlayer, nrow, ncol)

        @staticmethod
        def _check(pkg, filehashes, ibound):
            return super(type(pkg), pkg)._pkgcheck(ibound)

        @staticmethod
        def _save(pkg, filehashes, directory):
            super(type(pkg), pkg).save(directory)

        def __init__(self, path, *args, **kwargs):
            self._caching_save = memory.cache(self._save, ignore=["pkg"])
            self._caching_check = memory.cache(self._check, ignore=["pkg", "ibound"])
            self._caching_max_n = memory.cache(self._max_n, ignore=["pkg"])
            super().__init__(*args, **kwargs)
            self._filehashself = hash_filemetadata(path)
            self._filehashes = {}
            self._outputfiles = []
            self._logger = logging.getLogger(__class__.__name__)
            self._logger.setLevel(logging.INFO)
            # TODO: ensure some immutability somehow?

        def _compose(self, d, pattern=None):
            # d : dict
            # pattern : string or re.pattern
            path = util.compose(d, pattern).as_posix()
            self._outputfiles.append(path)
            return path

        def _max_active_n(self, varname, nlayer, nrow, ncol):
            filehashes = check_filehashes(self._filehashes)
            if hash_exists(
                self._caching_max_n, self, filehashes, varname, nlayer, nrow, ncol
            ):
                self._logger.info("MAX_N: Input recognized. Skipping.")
            else:
                self._logger.info("MAX_N: Input is new. Counting anew.")
            return self._caching_max_n(self, filehashes, varname, nlayer, nrow, ncol)

        def _pkgcheck(self, ibound=None):
            filehashes = check_filehashes(self._filehashes)
            if hash_exists(self._caching_check, self, filehashes, ibound):
                self._logger.info("CHECK: Input recognized. Skipping.")
            else:
                self._logger.info("CHECK: Input is new. Checking anew.")
            self._caching_check(self, filehashes, ibound)

        def save(self, directory):
            filehashes = check_filehashes(self._filehashes)
            # Check if the output has already been written once with the current
            # input files.
            if hash_exists(self._caching_save, self, filehashes, directory):
                self._logger.info("SAVING: Input recognized.")
                # Hash exists within store, so files should exist already.
                # But check whether the output looks good.
                output_hashes = output_metadata_hashes(self)
                # If it doesn't exist in store, something's wrong:
                if not hash_exists(output_status, output_hashes):
                    self._logger.info("SAVING: Output has been changed. Saving anew.")
                    # Retry: write it again, collect hash for the output data,
                    # and store it.
                    # By-pass caching: filehash already exists
                    self._save(self, filehashes, directory)
                    output_hashes = output_metadata_hashes(self)
                    output_status(output_hashes)
                else:
                    self._logger.info("SAVING: Output recognized. Skipping.")
            else:  # The files haven't been written yet with the current input.
                # Write it, collect hash for the output data,
                # and store it.
                self._logger.info("SAVING: Input is new. Saving anew.")
                self._caching_save(self, filehashes, directory)
                output_hashes = output_metadata_hashes(self)
                output_status(output_hashes)

    # Update name of CachingPackage
    CachingPackage.__name__ = f"Caching{package.__name__}"
    CachingPackage.__qualname__ = f"Caching{package.__qualname__}"

    return CachingPackage
