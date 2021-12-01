import collections


class Model(collections.UserDict):
    def __setitem__(self, key, value):
        # TODO: Add packagecheck
        super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v


class MetaSwapModel(Model):
    """
    Contains data and writes consistent model input files
    """

    _pkg_id = "model"

    def __init__(self):
        super().__init__()

    def write(self, directory):
        """
        Write packages
        """

        # write package contents
        for pkgname in self:
            self[pkgname].write(directory)
