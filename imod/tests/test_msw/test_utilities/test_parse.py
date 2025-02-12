from pytest_cases import parametrize_with_cases

from imod.msw.utilities.parse import (
    _try_parsing_string_to_number,
    correct_unsa_svat_path,
)


class ParseCases:
    def case_int(self):
        return "1", int

    def case_float(self):
        return "1.0", float

    def case_string(self):
        return "a", str

    def case_aterisk(self):
        return "*", str

    def case_exclamation(self):
        return "!", str


class UnsaSvatPathCases:
    def case_relpath(self):
        path = '"$unsat\\"'
        path_expected = "./unsat"
        return path, path_expected

    def case_abspath(self):
        path = '"C:\\Program Files\\MetaSWAP\\unsat\\"'
        path_expected = "C:\\Program Files\\MetaSWAP\\unsat"
        return path, path_expected

    def case_quoted(self):
        path = '"a/b/c"'
        path_expected = "a/b/c"
        return path, path_expected


@parametrize_with_cases(["s", "expected_type"], cases=ParseCases)
def test_try_parsing_string_to_value(s, expected_type):
    # Act
    parsed = _try_parsing_string_to_number(s)
    # Assert
    assert type(parsed) is expected_type


@parametrize_with_cases(["path", "path_expected"], cases=UnsaSvatPathCases)
def test_correct_unsa_svat_path(path, path_expected):
    # Act
    path_corrected = correct_unsa_svat_path(path)
    # Assert
    assert path_corrected == path_expected
