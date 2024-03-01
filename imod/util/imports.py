class MissingOptionalModule:
    """
    Presents a clear error for optional modules.
    """

    def __init__(self, name):
        self.name = name

    def __getattr__(self, name):
        raise ImportError(f"{self.name} is required for this functionality")
