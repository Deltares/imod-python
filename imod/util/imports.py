from typing import NoReturn


class MissingOptionalModule:
    """
    Presents a clear error for optional modules.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __getattr__(self, name: str) -> NoReturn:
        raise ImportError(f"{self.name} is required for this functionality")
