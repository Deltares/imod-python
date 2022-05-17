"""Contains unit tests for user object processors"""
from collections import namedtuple

import pytest

from imod import declxml as xml

from .helpers import strip_xml

Author = namedtuple("Author", ["name", "books"])
Book = namedtuple("Book", ["title", "year_published"])


class Person(object):
    """Class to represent data for a person"""

    def __init__(self):
        self.name = None
        self.age = None
        self.gender = None
        self.pets = None

    def set_values(self, name=None, age=None, gender=None, pets=None):
        """Sets the values for a person object"""
        self.name = name
        self.age = age
        self.gender = gender
        self.pets = pets
        return self

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.age == other.age
            and self.gender == other.gender
        )

    def __repr__(self):
        return "Person(name={}, age={}, gender={}, pets={})".format(
            self.name, self.age, self.gender, self.pets
        )


def test_named_tuple_parse():
    """Parse a namedtuple value"""
    xml_string = """
    <author>
        <name>Robert A. Heinlein</name>
        <book>
            <title>Starship Troopers</title>
            <year-published>1959</year-published>
        </book>
        <book>
            <title>Stranger in a Strange Land</title>
            <year-published>1961</year-published>
        </book>
    </author>
    """

    processor = xml.named_tuple(
        "author",
        Author,
        [
            xml.string("name"),
            xml.array(
                xml.named_tuple(
                    "book",
                    Book,
                    [
                        xml.string("title"),
                        xml.integer("year-published", alias="year_published"),
                    ],
                ),
                alias="books",
            ),
        ],
    )

    expected = Author(
        name="Robert A. Heinlein",
        books=[
            Book(title="Starship Troopers", year_published=1959),
            Book(title="Stranger in a Strange Land", year_published=1961),
        ],
    )

    actual = xml.parse_from_string(processor, xml_string)

    assert expected == actual


def test_named_tuple_parse_missing_field():
    """Tests attempting to a parse a named tuple with fields missing from the processor"""
    xml_string = """
    <author>
        <name>Robert A. Heinlein</name>
        <book>
            <title>Starship Troopers</title>
            <year-published>1959</year-published>
        </book>
        <book>
            <title>Stranger in a Strange Land</title>
            <year-published>1961</year-published>
        </book>
    </author>
    """

    # User forgot to specify processor for namedtuple's books field which
    # should lead to an error when attempting to create the named tuple value.
    processor = xml.named_tuple("author", Author, [xml.string("name")])

    with pytest.raises(TypeError):
        xml.parse_from_string(processor, xml_string)


def test_named_tuple_serialize():
    """Serialize a namedtuple value"""
    value = Author(
        name="Robert A. Heinlein",
        books=[
            Book(title="Starship Troopers", year_published=1959),
            Book(title="Stranger in a Strange Land", year_published=1961),
        ],
    )

    processor = xml.named_tuple(
        "author",
        Author,
        [
            xml.string("name"),
            xml.array(
                xml.named_tuple(
                    "book",
                    Book,
                    [
                        xml.string("title"),
                        xml.integer("year-published", alias="year_published"),
                    ],
                ),
                alias="books",
            ),
        ],
    )

    expected = strip_xml(
        """
    <author>
        <name>Robert A. Heinlein</name>
        <book>
            <title>Starship Troopers</title>
            <year-published>1959</year-published>
        </book>
        <book>
            <title>Stranger in a Strange Land</title>
            <year-published>1961</year-published>
        </book>
    </author>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_user_object_parse_aliased():
    """Parse an aliased user object"""
    xml_string = """
    <root>
        <book>Harry Potter</book>
        <person>
            <name>Ron</name>
            <age>18</age>
        </person>
    </root>
    """

    processor = xml.dictionary(
        "root",
        [
            xml.string("book"),
            xml.user_object(
                "person",
                Person,
                [xml.string("name"), xml.integer("age")],
                alias="character",
            ),
        ],
    )

    expected = {
        "book": "Harry Potter",
        "character": Person().set_values(name="Ron", age=18),
    }

    actual = xml.parse_from_string(processor, xml_string)

    assert expected == actual


def test_user_object_parse_array():
    """Parse a user object in an array"""
    xml_string = """
    <root>
        <team>A Team</team>
        <person>
            <name>Bob</name>
            <age>23</age>
        </person>
        <person>
            <name>Jane</name>
            <age>26</age>
        </person>
    </root>
    """

    processor = xml.dictionary(
        "root",
        [
            xml.string("team"),
            xml.array(
                xml.user_object(
                    "person", Person, [xml.string("name"), xml.integer("age")]
                ),
                alias="members",
            ),
        ],
    )

    expected = {
        "team": "A Team",
        "members": [
            Person().set_values(name="Bob", age=23),
            Person().set_values(name="Jane", age=26),
        ],
    }

    actual = xml.parse_from_string(processor, xml_string)

    assert expected == actual


def test_user_object_parse_nested():
    """Parse a user object as a nested element in the document"""
    xml_string = """
    <root>
        <position>quarterback</position>
        <person>
            <name>Bob</name>
            <pet>Fluffy</pet>
            <pet>Spots</pet>
        </person>
    </root>
    """

    processor = xml.dictionary(
        "root",
        [
            xml.string("position"),
            xml.user_object(
                "person",
                Person,
                [
                    xml.string("name"),
                    xml.array(xml.string("pet"), alias="pets"),
                ],
            ),
        ],
    )

    expected = {
        "position": "quarterback",
        "person": Person().set_values(name="Bob", pets=["Fluffy", "Spots"]),
    }

    actual = xml.parse_from_string(processor, xml_string)

    assert expected == actual


def test_user_object_parse_root():
    """Parse a user object as the root of the document"""
    xml_string = """
    <person>
        <name>Bob</name>
        <age>23</age>
        <gender>male</gender>
    </person>
    """

    processor = xml.user_object(
        "person", Person, [xml.string("name"), xml.integer("age"), xml.string("gender")]
    )

    expected = Person().set_values(name="Bob", age=23, gender="male")

    actual = xml.parse_from_string(processor, xml_string)

    assert expected == actual


def test_user_object_serialize_aliased():
    """Serializes an aliased user object"""
    value = {
        "book": "Harry Potter",
        "character": Person().set_values(name="Malfoy", age=17),
    }

    processor = xml.dictionary(
        "root",
        [
            xml.string("book"),
            xml.user_object(
                "person",
                Person,
                [xml.string("name"), xml.integer("age")],
                alias="character",
            ),
        ],
    )

    expected = strip_xml(
        """
    <root>
        <book>Harry Potter</book>
        <person>
            <name>Malfoy</name>
            <age>17</age>
        </person>
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_user_object_serialize_array():
    """Serializes an array of user objects"""
    value = [
        Person().set_values(name="Bob", age=33),
        Person().set_values(name="Jan", age=24),
        Person().set_values(name="Tom", age=29),
    ]

    processor = xml.array(
        xml.user_object("person", Person, [xml.string("name"), xml.integer("age")]),
        nested="people",
    )

    expected = strip_xml(
        """
    <people>
        <person>
            <name>Bob</name>
            <age>33</age>
        </person>
        <person>
            <name>Jan</name>
            <age>24</age>
        </person>
        <person>
            <name>Tom</name>
            <age>29</age>
        </person>
    </people>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_user_object_serialize_nested():
    """Serializes a nested user object"""
    value = {
        "position": "quarterback",
        "person": Person().set_values(name="Bob", pets=["Fluffy", "Spots"]),
    }

    processor = xml.dictionary(
        "root",
        [
            xml.string("position"),
            xml.user_object(
                "person",
                Person,
                [
                    xml.string("name"),
                    xml.array(xml.string("pet"), alias="pets"),
                ],
            ),
        ],
    )

    expected = strip_xml(
        """
    <root>
        <position>quarterback</position>
        <person>
            <name>Bob</name>
            <pet>Fluffy</pet>
            <pet>Spots</pet>
        </person>
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_user_object_serialize_root():
    """Serializes a user object as the root of the document"""
    value = Person().set_values(name="Bob", age=27, gender="male")

    processor = xml.user_object(
        "person", Person, [xml.string("name"), xml.integer("age"), xml.string("gender")]
    )

    expected = strip_xml(
        """
    <person>
        <name>Bob</name>
        <age>27</age>
        <gender>male</gender>
    </person>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual
