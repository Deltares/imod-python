import warnings

import numpy as np
from scipy import sparse
from xugrid.ugrid.interpolate import ILU0Preconditioner


def _build_connectivity(shape):
    # Get the Cartesian neighbors for a finite difference approximation.
    size = np.prod(shape)
    index = np.arange(size).reshape(shape)

    # Build nD connectivity
    ii = []
    jj = []
    for d in range(len(shape)):
        slices = [slice(None)] * len(shape)

        slices[d] = slice(None, -1)
        left = index[tuple(slices)].ravel()
        slices[d] = slice(1, None)
        right = index[tuple(slices)].ravel()
        ii.extend([left, right])
        jj.extend([right, left])

    i = np.concatenate(ii)
    j = np.concatenate(jj)
    return sparse.coo_matrix((np.ones(len(i)), (i, j)), shape=(size, size)).tocsr()


def _interpolate(
    arr: np.ndarray,
    connectivity: sparse.csr_matrix,
    direct: bool,
    delta: float,
    relax: float,
    rtol: float,
    atol: float,
    maxiter: int,
):
    ar1d = arr.ravel()
    unknown = np.isnan(ar1d)
    known = ~unknown

    # Set up system of equations.
    matrix = connectivity.copy()
    matrix.setdiag(-matrix.sum(axis=1).A[:, 0])
    rhs = -matrix[:, known].dot(ar1d[known])

    # Linear solve for the unknowns.
    A = matrix[unknown][:, unknown]
    b = rhs[unknown]
    if direct:
        x = sparse.linalg.spsolve(A, b)
    else:  # Preconditioned conjugate-gradient linear solve.
        # Create preconditioner M
        M = ILU0Preconditioner.from_csr_matrix(A, delta=delta, relax=relax)
        # Call conjugate gradient solver
        x, info = sparse.linalg.cg(A, b, rtol=rtol, atol=atol, maxiter=maxiter, M=M)
        if info < 0:
            raise ValueError("scipy.sparse.linalg.cg: illegal input or breakdown")
        elif info > 0:
            warnings.warn(f"Failed to converge after {maxiter} iterations")

    out = ar1d.copy()
    out[unknown] = x
    return out.reshape(arr.shape)
