"""
Contains an imod model object
"""

# This class allows only imod packages as values
class Model(dict):
    def __init__(self, ibound):
        dict.__init__(self)
        self["ibound"] = ibound

    def __setitem__(self, key, value):
        if not hasattr(value, "_pkgcheck"):
            raise ValueError("not a package")
        dict.__setitem__(self, key, value)
    
    def update(self, *arg, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v
    
    def render(self):
        # Create groups for chd, drn, ghb, riv, wel
