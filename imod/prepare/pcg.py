"""Preconditioned Conjugate Gradient Solver Package.

This is a translation of the fortran77 modflow2000 pcg2.f module.
"""
import numba


class PreconditionedConjugateGradientSolver(object):
    __slots__ = ("hclose", "rclose", "mxiter", "iter1", "relax")

    def __init__(self, hclose, rclose, mxiter, iter1, relax=1.0):
        """
        hclose : float
            Head tolerance for stopping condition
        rclose : float
            Residual tolerance for stopping condition
        iter1 : integer
            Maximum number of inner iterations
        mxiter : integer
            Maximum number of outer iterations
        relax : float
            Relaxation parameter for incomplete Cholesky preconditioning
        damp : float
            Dampening parameter for head change of outer iterations
        """
        self.hclose = hclose
        self.rclose = rclose
        self.mxiter = mxiter
        self.iter1 = iter1
        self.relax = relax

    def solve(self, hnew, cc, cr, cv, ibound, rhs, hcof, res, cd, v, ss, p):
        """
        Solves the groundwater equations via the method of conjugated gradients,
        with incomplete Cholesky preconditioning.

        Mutates hnew, converged, cr, cc, cv, ibound, and the work arrays.

        Parameters
        ----------
        hnew : ndarray (3D) of floats
            Initial guess, will be updated in place.

            The linear operator exists only virtually, and is formed by cr, cc, cv,
            ibound.

        cr : ndarray (3D) of floats
            Row conductance.
        cc : ndarray (3D) of floats
            Column conductance
        cv : ndarray (3D) of floats
            Vertical conductance (layer)
        ibound : ndarray (3D) of integers
            Active (1), inactive (0), fixed head (-1)
        rhs : ndarray (3D) of floats
            Right-hand side
        hcof : ndarray (3D)
            Work array, coefficient of head
        res : ndarray (1D)
            Work array, residual
        cd : ndarray (1D)
            Work array, Incomplete Cholesky diagonal
        v : ndarray (1D)
            Work array, intermediate solution
        ss : ndarray (1D)
            Work array
        p : ndarray (1D)
            Work array

        Returns
        -------
        converged : boolean
            convergence flag
        """

        # Initialize
        nlay, nrow, ncol = ibound.shape
        nrc = nrow * ncol
        nodes = nrc * nlay
        srnew = 0.0
        iiter = 0
        converged = False
        pcg_converged = False

        # Ravel 3D arrays
        hnew = hnew.ravel()
        cc = cc.ravel()
        cr = cr.ravel()
        cv = cv.ravel()
        ibound = ibound.ravel()
        rhs = rhs.ravel()
        hcof = hcof.ravel()

        # Clear work arrays
        res[:] = 0.0
        cd[:] = 0.0
        v[:] = 0.0
        ss[:] = 0.0
        p[:] = 0.0

        # Calculate residual
        # Will also get rid of dry cells in ibound, cc, cr, cv
        ibound, cc, cr, cv, hcof, res = calculate_residual(
            ibound, nlay, nrow, ncol, nrc, nodes, cr, cc, cv, hnew, rhs, hcof, res
        )

        while not pcg_converged and iiter < self.iter1:
            # Start internal iterations
            iiter = iiter + 1
            # Initialize variables that track maximum head change
            # and residual value during each iteration
            bigh = 0.0
            bigr = 0.0

            cd, v = precondition_incomplete_cholesky(
                ibound,
                nlay,
                nrow,
                ncol,
                nrc,
                cc,
                cr,
                cv,
                iiter,
                hcof,
                self.relax,
                res,
                cd,
                v,
            )

            ss = back_substition(ibound, nlay, nrow, ncol, nrc, cr, cc, cv, v, cd, ss)

            p, srnew = calculate_cg_p(ibound, nodes, ss, res, srnew, iiter, p)

            v, alpha = calculate_cg_alpha(
                ibound,
                nlay,
                nrow,
                ncol,
                nrc,
                nodes,
                v,
                cc,
                cr,
                cv,
                p,
                hcof,
                srnew,
                self.mxiter,
            )

            hnew, res, bigh, bigr, indices_bigh, indices_bigr = calculate_heads(
                ibound, nlay, nrow, ncol, nrc, alpha, p, bigh, bigr, hnew, res, v
            )

            # check the convergence criterion
            if abs(bigh) <= self.hclose and abs(bigr) <= self.rclose:
                pcg_converged = True
            if iiter == 1 and pcg_converged:
                converged = True

        return converged


@numba.njit
def calculate_residual(
    ibound, nlay, nrow, ncol, nrc, nodes, cr, cc, cv, hnew, rhs, hcof, res
):
    """
    Calculate the residual.

    For a dense form:
    Residual = np.dot(A, x) - b

    Parameters
    ----------
    ibound
    nlay
    nrow
    ncol
    nrc
    nodes
    cr
    cc
    cv
    hnew
    rhs
    hcof : array[nodes]

    Returns
    -------
    ibound
        ibound with cells remove that are surrounded by inactive cells
    hcof
        coefficient of head
    res
        residuals
    """
    for k in range(nlay):
        for i in range(nrow):
            for j in range(ncol):
                # Calculate 1 dimensional subscript of current cell
                # and skip calculcations if cell is inactive
                n = j + i * ncol + k * nrc
                if ibound[n] == 0:
                    cc[n] = 0.0
                    cr[n] = 0.0
                    if n <= (nodes - nrc - 1):
                        cv[n] = 0.0
                    if n >= 1:
                        cr[n - 1] = 0.0
                    if n >= ncol - 1:
                        cc[n - ncol] = 0.0
                    if n <= (nodes - nrc - 1) and n >= nrc:
                        cv[n - nrc] = 0.0
                    continue
                # Calculate 1 dimensional subscript for locating
                # the 6 surrounding cells
                nrn = n + ncol
                nrl = n - ncol
                ncn = n + 1
                ncl = n - 1
                nln = n + nrc
                nll = n - nrc
                # Calculate 1 dimensional subscripts for conductance to
                # the 6 surrounding cells
                ncf = n
                ncd = n - 1
                nrb = n - ncol
                nrh = n
                nls = n
                nlz = n - nrc
                # Get conductances to neighboring cells
                # neighbor is 1 row back
                b = 0.0
                bhnew = 0.0
                if i != 0:
                    b = cc[nrb]
                    bhnew = b * (hnew[nrl] - hnew[n])
                # neighbour is 1 row ahead
                h = 0.0
                hhnew = 0.0
                if i != nrow - 1:
                    h = cc[nrh]
                    hhnew = h * (hnew[nrn] - hnew[n])
                # neighbour is 1 column back
                d = 0.0
                dhnew = 0.0
                if j != 0:
                    d = cr[ncd]
                    dhnew = d * (hnew[ncl] - hnew[n])
                # neighbour is 1 column ahead
                f = 0.0
                fhnew = 0.0
                if j != ncol - 1:
                    f = cr[ncf]
                    fhnew = f * (hnew[ncn] - hnew[n])
                # neighbour is 1 layer behind
                z = 0.0
                zhnew = 0.0
                if k != 0:
                    z = cv[nlz]
                    zhnew = z * (hnew[nll] - hnew[n])
                # neighbour is 1 layer ahead
                s = 0.0
                shnew = 0.0
                if k != nlay - 1:
                    s = cv[nls]
                    shnew = s * (hnew[nln] - hnew[n])

                if i == nrow - 1:
                    cc[n] = 0.0
                if j == ncol - 1:
                    cr[n] = 0.0

                # Skip calculations and make cell inactive if all
                # surrounding cells are inactive
                if ibound[n] == 1:
                    if (b + h + d + f + z + s) == 0.0:
                        ibound[n] = 0.0
                        hcof[n] = 0.0
                        rhs[n] = 0.0
                        continue

                # Calculate the residual and store it in res.
                rrhs = rhs[n]
                hhcof = hnew[n] * hcof[n]
                res[n] = rrhs - zhnew - bhnew - dhnew - hhcof - fhnew - hhnew - shnew
                if ibound[n] < 0:
                    res[n] = 0.0
    return ibound, cc, cr, cv, hcof, res


@numba.njit
def cholesky_diagonal(
    ir,
    ic,
    il,
    i,
    j,
    k,
    n,
    f,
    h,
    s,
    ncol,
    nrow,
    nlay,
    cc,
    cr,
    cv,
    hcof,
    relax,
    delta,
    cd1,
    cd,
):
    """
    Calculate one value of cd
    first interal iteration only

    Returns
    -------
    delta
    cd1
    cd
    """

    cdcr = 0.0
    cdcc = 0.0
    cdcv = 0.0
    fcc = 0.0
    fcr = 0.0
    fcv = 0.0
    if ir >= 0 and cd[ir] != 0.0:
        cdcr = (f ** 2.0) / cd[ir]
    if ic >= 0 and cd[ic] != 0.0:
        cdcc = (h ** 2.0) / cd[ic]
    if il >= 0 and cd[il] != 0.0:
        cdcv = (s ** 2.0) / cd[il]

    if ir >= 0:
        fv = cv[ir]
        if k == nlay - 1:
            fv = 0.0
        if cd[ir] != 0.0:
            fcr = (f / cd[ir]) * (cc[ir] + fv)

    if ic >= 0:
        fv = cv[ic]
        if k == nlay - 1 and i > 0:
            fv = 0.0
        if cd[ic] != 0.0:
            fcc = (h / cd[ic]) * (cr[ic] + fv)
    if il >= 0:
        if cd[il] != 0.0:
            fcv = (s / cd[il]) * (cr[il] + cc[il])

    b = 0.0
    h = 0.0
    d = 0.0
    f = 0.0
    z = 0.0
    s = 0.0
    if i != 0:
        b = cc[ic]
    if i != nrow:
        h = cc[n]
    if j != 0:
        d = cr[ir]
    if j != ncol:
        f = cr[n]
    if k != 0:
        z = cv[il]
    if k != nlay:
        s = cv[n]

    hhcof = hcof[n] - z - b - d - f - h - s
    cd[n] = (1.0 + delta) * hhcof - cdcr - cdcc - cdcv - relax * (fcr + fcc + fcv)

    if cd1 == 0.0 and cd[n] != 0.0:
        cd1 = cd[n]
    if (cd[n] * cd1) < 0:
        delta = 1.5 * delta + 0.001
        raise RuntimeError(
            "Matrix is severely non-diagonally dominant. Check input. Stopping execution."
        )
    return cd, cd1, delta


@numba.njit
def precondition_incomplete_cholesky(
    ibound, nlay, nrow, ncol, nrc, cc, cr, cv, iiter, hcof, relax, res, cd, v
):
    """
    Incomplete Cholesky preconditioning
    Step through cells to calculate the diagonal of the cholesky
    matrix (first internal iteration only) and the intermediate
    solution. Store them in cd and v, respectively.

    Returns
    -------
    cd
        Cholesky diagonal
    v
        intermediate solution
    """
    delta = 0.0
    cd1 = 0.0
    for k in range(nlay):
        for i in range(nrow):
            for j in range(ncol):
                n = j + i * ncol + k * nrc
                if ibound[n] < 1:
                    continue

                # Calculate v
                h = 0.0
                vcc = 0.0
                ic = n - ncol  # won't be used if i == 0
                if i != 0:
                    h = cc[ic]
                    if cd[ic] != 0.0:
                        vcc = h * v[ic] / cd[ic]

                f = 0.0
                vcr = 0.0
                ir = n - 1  # wont'be used if j == 0
                if j != 0:
                    f = cr[ir]
                    if cd[ir] != 0.0:
                        vcr = f * v[ir] / cd[ir]

                s = 0.0
                vcv = 0.0
                il = n - nrc  # won't be used if k == 0
                if k != 0:
                    s = cv[il]
                    if cd[il] != 0.0:
                        vcv = s * v[il] / cd[il]

                v[n] = res[n] - vcr - vcc - vcv

                # Calculate Cholesky diagonal for the first internal iteration.
                if iiter == 1:
                    cd, cd1, delta = cholesky_diagonal(
                        ir,
                        ic,
                        il,
                        i,
                        j,
                        k,
                        n,
                        f,
                        h,
                        s,
                        ncol,
                        nrow,
                        nlay,
                        cc,
                        cr,
                        cv,
                        hcof,
                        relax,
                        delta,
                        cd1,
                        cd,
                    )

    return cd, v


@numba.njit
def back_substition(ibound, nlay, nrow, ncol, nrc, cr, cc, cv, v, cd, ss):
    """
    Step through each cell and solve for s of the conjugate
    gradient algorithm by back substition. Store the result in ss.

    Returns
    -------
    ss
    """
    for kk in range(nlay - 1, -1, -1):
        for ii in range(nrow - 1, -1, -1):
            for jj in range(ncol - 1, -1, -1):
                n = jj + ii * ncol + kk * nrc
                if ibound[n] <= 0:
                    continue
                nc = n + 1
                nr = n + ncol
                nl = n + nrc
                sscr = 0.0
                sscc = 0.0
                sscv = 0.0
                if jj != ncol - 1:
                    sscr = cr[n] * ss[nc] / cd[n]
                if ii != nrow - 1:
                    sscc = cc[n] * ss[nr] / cd[n]
                if kk != nlay - 1:
                    sscv = cv[n] * ss[nl] / cd[n]
                vn = v[n] / cd[n]
                ss[n] = vn - sscr - sscc - sscv
    return ss


@numba.njit
def calculate_cg_p(ibound, nodes, ss, res, srnew, iiter, p):
    """
    Calculate p of the conjugate gradient algorithm

    Returns
    -------
    p
    srnew
    """
    srold = srnew
    srnew = 0.0
    for n in range(nodes):
        if ibound[n] <= 0:
            continue
        srnew = srnew + ss[n] * res[n]
    if iiter == 1:
        for n in range(nodes):
            p[n] = ss[n]
    else:
        for n in range(nodes):
            p[n] = ss[n] + (srnew / srold) * p[n]
    return p, srnew


@numba.njit
def calculate_cg_alpha(
    ibound, nlay, nrow, ncol, nrc, nodes, v, cc, cr, cv, p, hcof, srnew, mxiter
):
    """
    Calculate alpha of the conjugate routine.
    For the denominator of alpha, multiply the matrix a by the
    vector p, and store in v; then multiply p by v. Store in pap.

    Returns
    -------
    v
        intermediate solution
    alpha
        conjugate gradient alpha
    """
    pap = 0.0
    for k in range(nlay):
        for i in range(nrow):
            for j in range(ncol):
                n = j + i * ncol + k * nrc
                v[n] = 0.0
                if ibound[n] < 1:
                    continue

                nrn = n + ncol
                nrl = n - ncol
                ncn = n + 1
                ncl = n - 1
                nln = n + nrc
                nll = n - nrc

                ncf = n
                ncd = ncl
                nrb = nrl
                nrh = n
                nls = n
                nlz = nll

                b = 0.0
                if i != 0:
                    b = cc[nrb]
                h = 0.0
                if i != nrow - 1:
                    h = cc[nrh]
                d = 0.0
                if j != 0:
                    d = cr[ncd]
                f = 0.0
                if j != ncol - 1:
                    f = cr[ncf]
                z = 0.0
                if k != 0:
                    z = cv[nlz]
                s = 0.0
                if k != nlay - 1:
                    s = cv[nls]

                pn = p[n]

                bhnew = 0.0
                hhnew = 0.0
                dhnew = 0.0
                fhnew = 0.0
                zhnew = 0.0
                shnew = 0.0
                if nrl >= 0:
                    bhnew = b * (p[nrl] - pn)
                if nrn <= nodes - 1:
                    hhnew = h * (p[nrn] - pn)
                if ncl >= 0:
                    dhnew = d * (p[ncl] - pn)
                if ncn <= nodes - 1:
                    fhnew = f * (p[ncn] - pn)
                if nll >= 0:
                    zhnew = z * (p[nll] - pn)
                if nln <= nodes - 1:
                    shnew = s * (p[nln] - pn)

                # Calculate the product of a matrix and vector p and store
                # result in v.
                pn = hcof[n] * p[n]
                vn = zhnew + bhnew + dhnew + pn + fhnew + hhnew + shnew
                v[n] = vn
                pap = pap + p[n] * vn

    # Calculate alpha
    alpha = 1.0
    if pap == 0.0 and mxiter == 1:
        raise RuntimeError(
            "Conjugate gradient method failed. Set mxiter greater than one and try again. Stopping execution."
        )
    if pap != 0.0:
        alpha = srnew / pap
    return v, alpha


@numba.njit
def calculate_heads(ibound, nlay, nrow, ncol, nrc, alpha, p, bigh, bigr, hnew, res, v):
    """
    Calculate new heads and residuals, and save the largest
    change in head and the largest value of the residual.
    """
    ih = jh = kh = -1
    ir = jr = kr = -1
    for k in range(nlay):
        for i in range(nrow):
            for j in range(ncol):
                n = j + i * ncol + k * nrc
                if ibound[n] <= 0:
                    continue
                hchgn = alpha * p[n]
                if abs(hchgn) > abs(bigh):
                    bigh = hchgn
                    ih = i
                    jh = j
                    kh = k
                hnew[n] = hnew[n] + hchgn

                # residual (v is the product of matrix a and vector p)
                rchgn = -alpha * v[n]
                res[n] = res[n] + rchgn
                if abs(res[n]) > abs(bigr):
                    bigr = res[n]
                    ir = i
                    jr = j
                    kr = k
    indices_bigh = (kh, ih, jh)
    indices_bigr = (kr, ir, jr)
    return hnew, res, bigh, bigr, indices_bigh, indices_bigr
