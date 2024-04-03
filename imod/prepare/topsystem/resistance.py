import numpy as np


def c_radial(L, kh, kv, B, D):
    """
    Ernst's radial resistance term to a drain.

    Parameters
    ----------
    L : xr.DataArray of floats
        distance between water features
    kh : xr.DataArray of floats
        horizontal conductivity
    kv : xr.DataArray of floats
        vertical conductivity
    B : xr.DataArray of floats
        water feature wetted perimeter
    D : xr.DataArray of floats
        saturated thickness of the top system

    Returns
    -------
    radial_c : xr.DataArray
        Ernst's radial resistance for a drain
    """
    # Take anisotropy into account fully
    c = (
        L
        / (np.pi * np.sqrt(kh * kv))
        * np.log((4.0 * D) / (np.pi * B) * np.sqrt(kh / kv))
    )
    c = c.where(~(c < 0.0), other=0.0)
    return c


def c_leakage(kh, kv, D, c0, c1, B, length, dx, dy):
    """
    Computes the phreatic leakage resistance.

    Parameters
    ----------
    kh : xr.DataArray of floats
        horizontal conductivity of phreatic aquifer
    kv : xr.DataArray of floats
        vertical conductivity of phreatic aquifer
    c0 : xr.DataArray of floats
        hydraulic bed resistance of water feature
    c1 : xr.DataArray of floats
        hydraulic resistance of the first aquitard
    D : xr.DataArray of floats
        saturated thickness of the top system
    B : xr.DataArray of floats
        water feature wetted perimeter
    length : xr.DataArray of floats
        water feature length per cell
    dx : xr.DataArray of floats
        cellsize in x
    dy : xr.DataArray of floats
        cellsize in y

    Returns
    -------
    c_leakage: xr.DataArray of floats
        Hydraulic resistance of water features corrected for intra-cell
        hydraulic resistance and surface water interaction.
    """

    def f(x):
        """
        two x times cotangens hyperbolicus of x
        """
        # Avoid overflow for large x values
        # 20 is safe for 32 bit floats
        x = x.where(~(x > 20.0), other=20.0)
        exp2x = np.exp(2.0 * x)
        return x * (exp2x + 1) / (exp2x - 1.0)

    # TODO: raise ValueError when cells are far from square?
    # Since method of average ditch distance will not work well
    # Compute geometry of ditches within cells
    cell_area = abs(dy * dx)
    wetted_area = length * B
    fraction_wetted = wetted_area / cell_area
    # Compute average distance between ditches
    L = (cell_area - wetted_area) / length

    # Compute total resistance to aquifer
    c1_prime = c1 + (D / kv)
    # Compute influence for the part below the ditch
    labda_B = np.sqrt((kh * D * c1_prime * c0) / (c1_prime + c0))
    # ... and the field
    labda_L = np.sqrt(c1_prime * kh * D)

    x_B = B / (2.0 * labda_B)
    x_L = L / (2.0 * labda_L)

    # Compute feeding resistance
    c_rad = c_radial(L, kv, kh, B, D)
    c_L = (c0 + c1_prime) * f(x_L) + (c0 * L / B) * f(x_B)
    c_B = (c1_prime + c0) / (c_L - c0 * L / B) * c_L
    # total feeding resistance equals the harmonic mean of resistances below
    # ditch (B) and field (L) plus the radical resistance
    # Can also be regarded as area weighted sum of conductances of B and L
    c_total = 1.0 / (fraction_wetted / c_B + (1.0 - fraction_wetted) / c_L) + c_rad
    # Subtract aquifer and aquitard resistance from feeding resistance
    c = c_total - c1_prime
    # Replace areas where cells are fully covered by water
    c = c.where(~(L == 0.0), other=c0)
    return c
