from abc import ABC, abstractmethod
from typing import List


class StatusInfoBase(ABC):
    def __init__(self, title: str = ""):
        self.__title: str = title

    @property
    def title(self) -> str:
        return self.__title

    @title.setter
    def title(self, title: str) -> None:
        self.__title = title

    @property
    @abstractmethod
    def errors(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def has_errors(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def to_string(self) -> str:
        raise NotImplementedError


class StatusInfo(StatusInfoBase):
    """
    This class can be used to collect any status messages.
    In its current state, the object is limited to error messages.
    But this can be extended later with e.g. warnings.
    """

    def __init__(self, title: str = ""):
        super().__init__(title)
        self.__errors: List[str] = []

    def add_error(self, message: str) -> None:
        self.__errors.append(message)

    @property
    def errors(self) -> List[str]:
        return self.__errors

    def has_errors(self) -> bool:
        return any(self.__errors)

    def to_string(self) -> str:
        header = self.title + ":" + "\n"
        indented_errors = "{1}{0}".format("\n".join(self.errors), "\t* ")
        return header + indented_errors


class NestedStatusInfo(StatusInfoBase):
    """
    This class can be used to collect any nested status messages.
    """

    def __init__(self, title: str = ""):
        super().__init__(title)
        self.__children: List[StatusInfoBase] = []

    def add(self, status_info: StatusInfoBase):
        self.__children.append(status_info)

    @property
    def errors(self) -> List[str]:
        errors: List[str] = []
        for child in self.__children:
            errors += child.errors
        return errors

    def has_errors(self) -> bool:
        for child in self.__children:
            if child.has_errors():
                return True
        return False

    def to_string(self) -> str:
        string = ""
        for child in self.__children:
            string += "\n* " + child.to_string()

        string = string.replace("\n", "\n\t")
        return self.title + ":" + string
