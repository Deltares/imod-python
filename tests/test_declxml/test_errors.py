"""Tests for error messages"""
import pytest

from imod import declxml as xml

_PROCESSOR = xml.dictionary(
    "genre-authors",
    [
        xml.string("genre"),
        xml.array(
            xml.dictionary(
                "author",
                [
                    xml.string("name"),
                    xml.integer("birth-year"),
                    xml.array(
                        xml.dictionary(
                            "book", [xml.string("title"), xml.integer("year-published")]
                        ),
                        alias="books",
                    ),
                ],
            ),
            alias="authors",
        ),
    ],
)


class _ErrorMessageTestCase(object):
    """Base class for error message test cases"""

    processor = None
    value = None
    xml_string = None
    expected_exception_type = None
    expected_message = None

    def test_parse(self):
        """Tests parsing with error message"""
        with pytest.raises(self.expected_exception_type) as exception_info:
            xml.parse_from_string(self.processor, self.xml_string)

        actual_message = str(exception_info.value)
        assert actual_message.endswith(self.expected_message)

    def test_serialize(self):
        """Tests serializing with error message"""
        with pytest.raises(self.expected_exception_type) as exception_info:
            xml.serialize_to_string(self.processor, self.value)

        actual_message = str(exception_info.value)
        assert actual_message.endswith(self.expected_message)


class TestMissingWithDot(_ErrorMessageTestCase):
    """Missing value with dot specifier"""

    xml_string = """
    <files>
        <file name="a.txt" size="123" />
        <file name="b.txt" />
        <file name="c.txt" size="546" />
    </files>
    """

    value = [
        {
            "name": "a.txt",
            "size": 123,
        },
        {
            "name": "b.txt",
        },
        {
            "name": "c.txt",
            "size": 546,
        },
    ]

    processor = xml.array(
        xml.dictionary(
            "file",
            [xml.string(".", attribute="name"), xml.string(".", attribute="size")],
        ),
        nested="files",
    )

    expected_exception_type = xml.MissingValue

    expected_message = "files/file[1]"


class TestMissingWithSlash(_ErrorMessageTestCase):
    """Missing with slash specifier"""

    xml_string = """
    <root>
        <places>
            <city>
                <name>Kansas City</name>
                <location>
                    <lat>39.0997</lat>
                </location>
            </city>
            <city>
                <name>Lincoln</name>
                <location>
                    <lat>40.8258</lat>
                    <lon>96.6852</lon>
                </location>
            </city>
        </places>
    </root>
    """

    value = {
        "cities": [
            {
                "name": "Kansas City",
                "lat": 39.0997,
            },
            {"name": "Lincoln", "lat": 40.8258, "lon": 96.6852},
        ]
    }

    processor = xml.dictionary(
        "root/places",
        [
            xml.array(
                xml.dictionary(
                    "city",
                    [
                        xml.string("name"),
                        xml.floating_point("location/lat", alias="lat"),
                        xml.floating_point("location/lon", alias="lon"),
                    ],
                ),
                alias="cities",
            )
        ],
    )

    expected_exception_type = xml.MissingValue

    expected_message = "root/places/city[0]/location/lon"


class TestMissingArrayItem(_ErrorMessageTestCase):
    """Process with a missing array item"""

    xml_string = """
    <genre-authors>
        <genre>Science Fiction</genre>
        <authors>
            <author>
                <name>Robert A. Heinlein</name>
                <birth-year>1907</birth-year>
                <book>
                    <title>Starship Troopers</title>
                    <year-published>1959</year-published>
                </book>
                <book>
                    <title>Stranger in a Strange Land</title>
                    <year-published>1961</year-published>
                </book>
            </author>
            <author>
                <name>Isaac Asimov</name>
                <birth-year>1920</birth-year>
                <book>
                    <title>I, Robot</title>
                </book>
                <book>
                    <title>Foundation</title>
                    <year-published>1951</year-published>
                </book>
            </author>
        </authors>
    </genre-authors>
    """

    processor = xml.dictionary(
        "genre-authors",
        [
            xml.string("genre"),
            xml.array(
                xml.dictionary(
                    "author",
                    [
                        xml.string("name"),
                        xml.integer("birth-year"),
                        xml.array(
                            xml.dictionary(
                                "book",
                                [xml.string("title"), xml.integer("year-published")],
                            ),
                            alias="books",
                        ),
                    ],
                ),
                nested="authors",
            ),
        ],
    )

    value = {
        "genre": "Science Fiction",
        "authors": [
            {
                "name": "Robert A. Heinlein",
                "birth-year": 1907,
                "books": [
                    {"title": "Starship Troopers", "year-published": 1959},
                    {"title": "Stranger in a Strange Land", "year-published": 1961},
                ],
            },
            {
                "name": "Isaac Asimov",
                "birth-year": 1920,
                "books": [
                    {
                        "title": "I, Robot",
                    },
                    {"title": "Foundation", "year-published": 1951},
                ],
            },
        ],
    }

    expected_exception_type = xml.MissingValue

    expected_message = "genre-authors/authors/author[1]/book[0]/year-published"
