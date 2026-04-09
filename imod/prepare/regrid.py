import textwrap

ERROR_MSG = textwrap.dedent(
    """{name} is removed. Use the
    the regridder in ``xugrid`` instead, which is around 10 times faster. To
    regrid a single array, see:
    https://deltares.github.io/xugrid/examples/regridder_overview.html. To
    regrid Modflow6 packages or entire simulations, see the iMOD Python user
    guide:
    https://deltares.github.io/imod-python/user-guide/08-regridding.html.
    """
)


class Regridder(object):
    """
    Placeholder to preserve removed Regridder class namespace.

    .. attention::

        ``imod.prepare.Regridder`` is removed. Use the regridder in ``xugrid``
        instead, which is around 10 times faster. It is as simple as:

        >>> import xugrid as xu
        >>> regridder = xu.OverlapRegridder(source=source, target=like, method="mean")
        >>> result = regridder.regrid(source)

        For more information, see:
        https://deltares.github.io/xugrid/examples/regridder_overview.html. To
        regrid MODFLOW6 packages or entire MODFLOW6 simulations, see the iMOD
        Python user guide:
        https://deltares.github.io/imod-python/user-guide/08-regridding.html.

    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(ERROR_MSG.format(name=self.__class__.__name__))
