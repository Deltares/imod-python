import xarray as xr


def convert_pointwaterhead_freshwaterhead(
    pointwaterhead, density, elevation, density_fresh=1000.0
):
    r"""Function to convert point water head (as outputted by seawat)
    into freshwater head, using Eq.3 from Guo, W., & Langevin, C. D. (2002):

    .. math:: h_{f}=\frac{\rho}{\rho_{f}}h-\frac{\rho-\rho_{f}}{\rho_{f}}Z

    An edge case arises when the head is below the cell centre, or entirely below
    the cell. Strictly applying Eq.3 would result in freshwater heads that are
    lower than the original point water head, which is physically impossible. This
    function then outputs the freshwaterhead for the uppermost underlying cell where
    the original point water head exceeds the cell centre.

    *Requires bottleneck.*

    Parameters
    ----------
    pointwaterhead : float or xr.DataArray of floats
    `pointwaterhead` is the point water head as outputted by SEAWAT, in m.
    density : float or xr.DataArray of floats
    `density` is the water density on the same locations as `pointwaterhead`. 
    elevation : float or xr.DataArray of floats
    `elevation` is elevation on the same locations as `pointwaterhead`. 
    density_fresh : float, optional
    `density_fresh` is the density of freshwater (1000 kg/m3), or a different value 
    if different units are used, or a different density reference is required.

    Returns
    -------
    freshwaterhead : float or xr.DataArray of floats
    """

    freshwaterhead = (
        density / density_fresh * pointwaterhead
        - (density - density_fresh) / density_fresh * elevation
    )

    # edge case: point water head below z
    # return freshwater head of top underlying cell where elevation < pointwaterhead
    # only for xr.DataArrays
    if isinstance(pointwaterhead, xr.DataArray) and "layer" in pointwaterhead.dims:
        freshwaterhead = freshwaterhead.where(pointwaterhead > elevation).compute()
        freshwaterhead = freshwaterhead.bfill(dim="layer")

    return freshwaterhead
