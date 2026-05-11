from pytest_cases import parametrize_with_cases

from imod.formats.common import LineDelimiterInfo, infer_line_delimiter_info


class InferLineDelimiterInfoCases:
    def case_space_delimited(self):
        return (
            "1 2 3",
            3,
            True,
            True,
        )

    def case_comma_delimited(self):
        return (
            "1,2,3",
            3,
            False,
            True,
        )

    def case_comma_delimited_wrong_ncol(self):
        return (
            "1,2,3",
            4,
            False,
            False,
        )

    def case_comma_delimited_with_spaces(self):
        return (
            "1, 2, 3",
            3,
            False,
            True,
        )

    def case_tab_delimited(self):
        return (
            "1\t2\t3",
            3,
            True,
            True,
        )

    def case_mixed_delimiters(self):
        return (
            "1 2,3",
            3,
            False,
            False,
        )


@parametrize_with_cases(
    "line, ncol, has_whitespace, has_expected_ncols", cases=InferLineDelimiterInfoCases
)
def test_infer_delimwhitespace(line, ncol, has_whitespace, has_expected_ncols):
    assert infer_line_delimiter_info(line, ncol) == LineDelimiterInfo(
        has_whitespace, has_expected_ncols
    )
