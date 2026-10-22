import abc
from collections.abc import ItemsView, Iterator, KeysView, ValuesView
from typing import Any, Self


class IDict(abc.ABC):
    """
    Interface for collections.UserDict
    """

    def __setitem__(self, key: Any, item: Any) -> None:
        raise NotImplementedError

    def __getitem__(self, key: Any) -> Any:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __delitem__(self, key: Any) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def copy(self) -> Self:
        raise NotImplementedError

    def has_key(self, k: Any) -> bool:
        raise NotImplementedError

    def update(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def keys(self) -> KeysView[Any]:
        raise NotImplementedError

    def values(self) -> ValuesView[Any]:
        raise NotImplementedError

    def items(self) -> ItemsView[Any, Any]:
        raise NotImplementedError

    def pop(self, *args: Any) -> Any:
        raise NotImplementedError

    def __cmp__(self, dict_: Any) -> int:
        raise NotImplementedError

    def __contains__(self, item: Any) -> bool:
        raise NotImplementedError

    def __iter__(self) -> Iterator[Any]:
        raise NotImplementedError

    def __unicode__(self) -> str:
        raise NotImplementedError
