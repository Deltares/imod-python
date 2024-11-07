from abc import ABC, abstractmethod


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
    def errors(self) -> list[str]:
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
        self.__errors: list[str] = []

    def add_error(self, message: str) -> None:
        self.__errors.append(message)

    @property
    def errors(self) -> list[str]:
        return self.__errors

    def has_errors(self) -> bool:
        return any(self.__errors)

    def to_string(self) -> str:
        header = self.title + ":" + "\n"
        bullet = "\t* "
        indented_errors = f"{bullet}"+f"\n{bullet}".join(self.errors)
        return header + indented_errors


class NestedStatusInfo(StatusInfoBase):
    """
    This class can be used to collect any nested status messages.
    """

    def __init__(self, title: str = ""):
        super().__init__(title)
        self.__children: list[StatusInfoBase] = []
        self.__footer_text: str = ""

    def add(self, status_info: StatusInfoBase):
        self.__children.append(status_info)

    @property
    def errors(self) -> list[str]:
        errors: list[str] = []
        for child in self.__children:
            errors += child.errors
        return errors

    def has_errors(self) -> bool:
        for child in self.__children:
            if child.has_errors():
                return True
        return False

    def set_footer_text(self, text: str) -> None:
        self.__footer_text = text

    def to_string(self) -> str:
        string = ""
        for child in self.__children:
            string += "\n* " + child.to_string()

        string = string.replace("\n", "\n\t")
        footer = "\n" + self.__footer_text + "\n"
        return self.title + ":" + string + footer
