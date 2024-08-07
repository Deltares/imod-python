from functools import wraps
from typing import Callable, ParamSpec, TypeVar

from imod.typing.grid import enforce_dim_order

T = TypeVar("T")
P = ParamSpec("P")


def enforced_dim_order(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator to enforce dimension order after function call"""

    @wraps(func)
    def decorator(*args: P.args, **kwargs: P.kwargs):
        x = func(*args, **kwargs)
        # Multiple grids returned
        if isinstance(x, tuple):
            return tuple(enforce_dim_order(i) for i in x)
        return enforce_dim_order(x)

    return decorator
