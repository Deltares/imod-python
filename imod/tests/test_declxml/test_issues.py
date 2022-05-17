"""
Contains unit tests that reproduce previous issues. Whenever a new issue is
discovered, a failing test should be added to this file first before the issue
is fixed.
"""
from collections import namedtuple

from imod import declxml as xml

from .helpers import strip_xml


def test_parse_missing_non_required_namedtuple_issue_24():
    """Parse XML with a non-required namedtuple value."""
    Author = namedtuple(
        "Author",
        [
            "name",
            "genre",
        ],
    )
    Genre = namedtuple(
        "Genre",
        [
            "name",
        ],
    )

    processor = xml.named_tuple(
        "author",
        Author,
        [
            xml.string("name"),
            xml.named_tuple("genre", Genre, [xml.string("name")], required=False),
        ],
    )

    author_xml = """
    <author>
        <name>Robert A. Heinlein</name>
    </author>
    """

    expected_value = Author(name="Robert A. Heinlein", genre=None)

    actual_value = xml.parse_from_string(processor, author_xml)

    assert expected_value == actual_value


def test_serialize_none_namedtuple_issue_7():
    """Tests attempting to serialize a named tuple value that is None"""
    Athlete = namedtuple(
        "Athlete",
        [
            "name",
            "age",
        ],
    )

    processor = xml.dictionary(
        "race-result",
        [
            xml.floating_point("time"),
            xml.named_tuple(
                "athlete",
                Athlete,
                [
                    xml.string("name"),
                    xml.integer("age"),
                ],
                required=False,
            ),
        ],
    )

    value = {
        "time": 87.5,
        "athlete": None,
    }

    expected_xml = strip_xml(
        """
    <race-result>
        <time>87.5</time>
    </race-result>
    """
    )

    actual_xml = xml.serialize_to_string(processor, value)

    assert expected_xml == actual_xml


def test_serialize_none_object_issue_7():
    """Attempts to serialize an object value that is None"""

    class Athlete(object):
        def __init__(self):
            self.name = ""
            self.age = 0

    processor = xml.dictionary(
        "race-result",
        [
            xml.floating_point("time"),
            xml.user_object(
                "athlete",
                Athlete,
                [
                    xml.string("name"),
                    xml.integer("age"),
                ],
                required=False,
            ),
        ],
    )

    value = {
        "time": 87.5,
        "athlete": None,
    }

    expected_xml = strip_xml(
        """
    <race-result>
        <time>87.5</time>
    </race-result>
    """
    )

    actual_xml = xml.serialize_to_string(processor, value)

    assert expected_xml == actual_xml
