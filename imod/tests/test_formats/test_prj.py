import pytest

from imod.formats import prj


def test_tokenize():
    assert prj.tokenize("a b c") == ["a", "b", "c"]
    assert prj.tokenize("a,b,c") == ["a", "b", "c"]
    assert prj.tokenize("a, b, c") == ["a", "b", "c"]
    assert prj.tokenize("a, 'b', c") == ["a", "b", "c"]
    assert prj.tokenize("a, 'b d', c") == ["a", "b d", "c"]

    # We don't expect commas in our quoted strings since they're paths:
    with pytest.raises(ValueError, match="No closing quotation"):
        prj.tokenize("a, 'b,d', c")

    # From the examples:
    with pytest.raises(ValueError, match="No closing quotation"):
        prj.tokenize("That's life")
    assert prj.tokenize("That 's life'") == ["That", "s life"]
    assert prj.tokenize("That,'s life'") == ["That", "s life"]
    assert prj.tokenize("Thats life") == ["Thats", "life"]


class TestLineIterator:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.lines = prj.LineIterator(
            [
                ["This", "is", "the", "first", "line"],
                ["this", "is", "the", "second"],
            ]
        )

    def test_init(self):
        assert self.lines.count == -1

    def test_traversal(self):
        line = next(self.lines)
        assert line == ["This", "is", "the", "first", "line"]
        assert self.lines.count == 0
        line = next(self.lines)
        assert line == ["this", "is", "the", "second"]
        assert self.lines.count == 1
        assert self.lines.finished

        with pytest.raises(StopIteration):
            next(self.lines)

        assert self.lines.count == 1
        self.lines.back()
        assert self.lines.count == 0
        self.lines.back()
        assert self.lines.count == -1
        # Shouldn't go further back than -1
        self.lines.back()
        assert self.lines.count == -1

    def test_iter(self):
        lines = [line for line in self.lines]
        assert lines == [
            ["This", "is", "the", "first", "line"],
            ["this", "is", "the", "second"],
        ]


def test_parseblockheader():
    lines = prj.LineIterator(
        [
            ["abc", "def"],
            [],
        ]
    )
    assert prj.parse_blockheader(lines) == (None, None, None)
    assert prj.parse_blockheader(lines) == (None, None, None)

    lines = prj.LineIterator(
        [
            ["periods"],
            ["species"],
        ]
    )
    assert prj.parse_blockheader(lines) == (1, "periods", True)
    assert prj.parse_blockheader(lines) == (1, "species", True)

    lines = prj.LineIterator(
        [
            ["001", "(RIV)", "1"],
            ["002", "(GHB)", "0"],
            ["003", "(DRN)", "1", "extra", "content"],
        ]
    )
    assert prj.parse_blockheader(lines) == (1, "(riv)", True)
    assert prj.parse_blockheader(lines) == (2, "(ghb)", False)
    assert prj.parse_blockheader(lines) == (3, "(drn)", True)


def test_parse_time():
    lines = prj.LineIterator(
        [
            ["steady-state"],
            ["2000-01-01"],
            ["2000-01-01", "12:01:02"],
        ]
    )
    assert prj.parse_time(lines) == "steady-state"
    assert prj.parse_time(lines) == "2000-01-01 00:00:00"
    assert prj.parse_time(lines) == "2000-01-01 12:01:02"


def test_parse_blockline():
    lines = prj.LineIterator(
        [
            ["1", "2", "001", "1.0", "0.0", "-999.99", "ibound.idf"],
            ["2", "2", "012", "2.0", "3.0", "-999.99", "ibound.idf"],
            ["1", "1", "012", "2.0", "3.0", "-999.99", "ibound.idf"],
        ]
    )
