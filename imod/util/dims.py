from functools import wraps

from imod.typing.grid import enforce_dim_order


def enforced_dim_order(func):
    """Decorator to enforce dimension order after function call"""

    @wraps(func)
    def decorator(*args, **kwargs):
        x = func(*args, **kwargs)
        # Multiple grids returned
        if isinstance(x, tuple):
            return tuple(enforce_dim_order(i) for i in x)
        return enforce_dim_order(x)

    return decorator
