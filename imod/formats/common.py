import csv
from typing import NamedTuple


class LineDelimiterInfo(NamedTuple):
    has_whitespace: bool
    has_expected_cols: bool


def infer_line_delimiter_info(line: str, ncol: int) -> LineDelimiterInfo:
    """
    Infer whether the line is delimited by whitespace or commas, based on the
    number of columns. Also returns whether the line has the amount of expected
    columns if delimited by commas.

    Parameters
    ----------
    line : str
        The line to analyze.
    ncol : int
        The expected number of columns if line delimited by commas.

    Returns
    -------
    LineDelimiterInfo
        has_whitespace : bool
            Whether the line is delimited by whitespace.
        has_expected_cols : bool
            Whether the line has the expected number of columns if delimited by commas.
    """
    n_elem = len(next(csv.reader([line])))
    if n_elem == 1:
        return LineDelimiterInfo(has_whitespace=True, has_expected_cols=True)
    elif n_elem == ncol:
        return LineDelimiterInfo(has_whitespace=False, has_expected_cols=True)
    else:
        return LineDelimiterInfo(has_whitespace=False, has_expected_cols=False)
