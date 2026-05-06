import csv


def infer_delimwhitespace(line: str, ncol: int):
    """
    Infer whether the line is delimited by whitespace or commas, based on the number of columns.
    Also returns whether the line has the amount of expected columns.

    Parameters
    ----------
    line : str
        The line to analyze.
    ncol : int
        The expected number of columns.

    Returns
    -------
    has_whitespace : bool
        Whether the line is delimited by whitespace.
    has_expected_cols : bool
        Whether the line has the expected number of columns.
    """
    n_elem = len(next(csv.reader([line])))
    if n_elem == 1:
        return True, True
    elif n_elem == ncol:
        return False, True
    else:
        return False, False
