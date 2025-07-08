from typing import ClassVar, Protocol, runtime_checkable

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@runtime_checkable
class RegridMethodType(Protocol):
    # Work around that type annotation is a bit hard on dataclasses, as they
    # don't expose a class interface.
    # Adapted from: https://stackoverflow.com/a/55240861
    # "As already noted in comments, checking for this attribute is currently the
    # most reliable way to ascertain that something is a dataclass"
    # See also:
    # https://github.com/python/mypy/issues/6568#issuecomment-1324196557

    __dataclass_fields__: ClassVar[dict]

    def asdict(self) -> dict:
        return vars(self)


_CONFIG = ConfigDict(extra="forbid")


@dataclass(config=_CONFIG)
class EmptyRegridMethod(RegridMethodType):
    pass

