import abc


class IDict(abc.ABC):
    """
    Interface for collections.UserDict
    """

    def __setitem__(self, key, item):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __delitem__(self, key):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def has_key(self, k):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def keys(self):
        raise NotImplementedError

    def values(self):
        raise NotImplementedError

    def items(self):
        raise NotImplementedError

    def pop(self, *args):
        raise NotImplementedError

    def __cmp__(self, dict_):
        raise NotImplementedError

    def __contains__(self, item):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __unicode__(self):
        raise NotImplementedError
