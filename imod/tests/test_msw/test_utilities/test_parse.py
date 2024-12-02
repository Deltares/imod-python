from pytest_cases import parametrize_with_cases

from imod.msw.utilities.parse import _try_parsing_string_to_value


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


@parametrize_with_cases(["s", "expected_type"], cases=ParseCases)
def test_try_parsing_string_to_value(s, expected_type):
    # Act
    parsed = _try_parsing_string_to_value(s)
    # Assert
    assert type(parsed) is expected_type
