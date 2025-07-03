def test_public_api():
    """
    Test if functions previously in imod.util.py are still available under same
    namespace
    """
    from imod.util import (
        cd,  # noqa: F401
        empty_2d,  # noqa: F401
        empty_2d_transient,  # noqa: F401
        empty_3d,  # noqa: F401
        empty_3d_transient,  # noqa: F401
        from_mdal_compliant_ugrid2d,  # noqa: F401
        ignore_warnings,  # noqa: F401
        mdal_compliant_ugrid2d,  # noqa: F401
        replace,  # noqa: F401
        spatial_reference,  # noqa: F401
        temporary_directory,  # noqa: F401
        to_datetime,  # noqa: F401
        to_ugrid2d,  # noqa: F401
        transform,  # noqa: F401
        ugrid2d_data,  # noqa: F401
        values_within_range,  # noqa: F401
        where,  # noqa: F401
    )
