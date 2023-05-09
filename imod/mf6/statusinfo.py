class StatusInfo(object):
    """
    This class can be used to collect any status messages.
    """

    def __init__(self):
        self.__errors = []

    def add_error(self, message):
        self.__errors.append(message)

    @property
    def errors(self):
        return self.__errors

    def has_errors(self):
        return any(self.__errors)

    def __iadd__(self, other):
        self.__errors = self.__errors + other.__errors
        return self
