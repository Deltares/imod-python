# -*- coding: utf-8 -*-
"""Contains unit tests for parsing logic"""
import pytest

from imod import declxml as xml


def test_parse_array_embedded():
    """Parse array embedded within its parent element"""
    xml_string = """
    <root>
        <message>Hello, World!</message>
        <value>21</value>
        <value>17</value>
        <value>90</value>
        <value>6</value>
    </root>
    """

    values_array = xml.array(xml.integer("value"))

    processor = xml.dictionary(
        "root",
        [
            xml.string("message"),
            values_array,
        ],
    )

    expected = {
        "message": "Hello, World!",
        "value": [21, 17, 90, 6],
    }

    actual = xml.parse_from_string(processor, xml_string)

    assert expected == actual


def test_parse_array_embedded_aliased():
    """Parse array embedded within its parent element"""
    xml_string = """
    <root>
        <message>Goodbye, World!</message>
        <value>765</value>
        <value>3456</value>
    </root>
    """

    values_array = xml.array(xml.integer("value"), alias="numbers")

    processor = xml.dictionary(
        "root",
        [
            xml.string("message"),
            values_array,
        ],
    )

    expected = {
        "message": "Goodbye, World!",
        "numbers": [765, 3456],
    }

    actual = xml.parse_from_string(processor, xml_string)

    assert expected == actual


def test_parse_array_missing():
    """Parse missing array"""
    xml_string = """
    <root>
        <message>Hello, World!</message>
    </root>
    """

    values_array = xml.array(xml.integer("value"))

    processor = xml.dictionary(
        "root",
        [
            xml.string("message"),
            values_array,
        ],
    )

    with pytest.raises(xml.MissingValue):
        xml.parse_from_string(processor, xml_string)


def test_parse_array_missing_optional():
    """Parse missing optional array"""
    xml_string = """
    <root>
        <message>Hello, World!</message>
    </root>
    """

    values_array = xml.array(xml.integer("value", required=False), alias="numbers")

    processor = xml.dictionary(
        "root",
        [
            xml.string("message"),
            values_array,
        ],
    )

    expected = {
        "message": "Hello, World!",
        "numbers": [],
    }

    actual = xml.parse_from_string(processor, xml_string)

    assert expected == actual


def test_parse_array_nested():
    """Parse nested array"""
    xml_string = """
    <root>
        <message>Hello, World!</message>
        <numbers>
            <number>1</number>
            <number>2</number>
        </numbers>
    </root>
    """

    numbers_array = xml.array(xml.integer("number"), nested="numbers")

    processor = xml.dictionary(
        "root",
        [
            xml.string("message"),
            numbers_array,
        ],
    )

    expected = {
        "message": "Hello, World!",
        "numbers": [1, 2],
    }

    actual = xml.parse_from_string(processor, xml_string)

    assert expected == actual


def test_parse_array_nested_empty_optional():
    """Parse nested empty array"""
    xml_string = """
    <root>
        <message>Hello, World!</message>
        <numbers />
    </root>
    """

    numbers_array = xml.array(xml.integer("number", required=False), nested="numbers")

    processor = xml.dictionary(
        "root",
        [
            xml.string("message"),
            numbers_array,
        ],
    )

    expected = {
        "message": "Hello, World!",
        "numbers": [],
    }

    actual = xml.parse_from_string(processor, xml_string)

    assert expected == actual


def test_parse_array_nested_missing_optional():
    """Parse a nested missing array"""
    xml_string = """
    <root>
        <message>Hello</message>
    </root>
    """

    processor = xml.dictionary(
        "root",
        [
            xml.string("message"),
            xml.array(xml.integer("number", required=False), nested="numbers"),
        ],
    )

    expected = {
        "message": "Hello",
        "numbers": [],
    }

    actual = xml.parse_from_string(processor, xml_string)

    assert expected == actual


def test_parse_array_of_arrays():
    """Parse array of arrays"""
    xml_string = """
    <root-array>
        <values>
            <value>1</value>
            <value>17</value>
            <value>33</value>
        </values>
        <values>
            <value>99</value>
        </values>
    </root-array>
    """

    values_array = xml.array(xml.integer("value"), nested="values")

    root_processor = xml.array(values_array, nested="root-array")

    expected = [
        [1, 17, 33],
        [99],
    ]

    actual = xml.parse_from_string(root_processor, xml_string)

    assert expected == actual


def test_parse_array_of_dicts():
    """Parse array of dictionaries"""
    xml_string = """
    <array>
        <person>
            <name>John</name>
            <age>27</age>
        </person>
        <person>
            <name>Jane</name>
            <age>30</age>
        </person>
    </array>
    """

    person = xml.dictionary(
        "person",
        [
            xml.string("name"),
            xml.integer("age"),
        ],
    )

    processor = xml.array(person, nested="array")

    expected = [
        {
            "name": "John",
            "age": 27,
        },
        {
            "name": "Jane",
            "age": 30,
        },
    ]

    actual = xml.parse_from_string(processor, xml_string)

    assert expected == actual


def test_parse_array_of_primitives():
    """Parse array of primitive values"""
    xml_string = """
    <array>
        <value>1</value>
        <value>2</value>
        <value>3</value>
    </array>
    """

    processor = xml.array(xml.integer("value"), nested="array")

    expected = [1, 2, 3]

    actual = xml.parse_from_string(processor, xml_string)

    assert expected == actual


def test_parse_array_optional_present():
    """Parse optional array that is present"""
    xml_string = """
    <root>
        <message>Hello, World!</message>
        <value>45</value>
        <value>908</value>
    </root>
    """

    values_array = xml.array(xml.integer("value", required=False), alias="numbers")

    processor = xml.dictionary(
        "root",
        [
            xml.string("message"),
            values_array,
        ],
    )

    expected = {
        "message": "Hello, World!",
        "numbers": [45, 908],
    }

    actual = xml.parse_from_string(processor, xml_string)

    assert expected == actual


def test_parse_array_root_missing():
    """Parse an array as the root processor"""
    xml_string = """
    <wrong-array>
        <value>1</value>
        <value>2</value>
    </wrong-array>
    """

    processor = xml.array(xml.integer("value"), nested="array")

    with pytest.raises(xml.MissingValue):
        xml.parse_from_string(processor, xml_string)


def test_parse_array_root_non_nested():
    """Parse a non-nested array as the root processor"""
    xml_string = """
    <root>
        <value>1</value>
        <value>2</value>
    </root>
    """

    processor = xml.array(xml.integer("value"))

    with pytest.raises(xml.InvalidRootProcessor):
        xml.parse_from_string(processor, xml_string)


def test_parse_array_root_optional():
    """Parse an array as the root processor"""
    xml_string = """
    <wrong-array>
        <value>1</value>
        <value>2</value>
    </wrong-array>
    """

    processor = xml.array(xml.integer("value", required=False), nested="array")

    expected = []

    actual = xml.parse_from_string(processor, xml_string)

    assert expected == actual


def test_parse_array_root_optional_present():
    """Parse an array as the root processor"""
    xml_string = """
    <array>
        <value>1</value>
        <value>2</value>
    </array>
    """

    processor = xml.array(xml.integer("value", required=False), nested="array")

    expected = [1, 2]

    actual = xml.parse_from_string(processor, xml_string)

    assert expected == actual


def test_parse_attribute():
    """Parses an attribute value"""
    xml_string = """
        <root>
            <element value="hello">27</element>
        </root>
    """

    processor = xml.dictionary(
        "root",
        [
            xml.string("element", attribute="value"),
        ],
    )

    expected = {
        "value": "hello",
    }

    actual = xml.parse_from_string(processor, xml_string)

    assert expected == actual


def test_parse_attribute_aliased():
    """Parses an attribute value with an alias"""
    xml_string = """
        <root>
            <element value="hello">27</element>
        </root>
    """

    processor = xml.dictionary(
        "root",
        [
            xml.string("element", attribute="value", alias="value_alias"),
        ],
    )

    expected = {
        "value_alias": "hello",
    }

    actual = xml.parse_from_string(processor, xml_string)

    assert expected == actual


def test_parse_attribute_default():
    """Parses an attribute with a default value specified"""
    xml_string = """
    <root>
        <element>27</element>
    </root>
    """

    processor = xml.dictionary(
        "root",
        [
            xml.string("element", attribute="value", required=False, default="Goodbye"),
        ],
    )

    expected = {
        "value": "Goodbye",
    }

    actual = xml.parse_from_string(processor, xml_string)

    assert expected == actual


def test_parse_attribute_default_present():
    """Parses an attribute with a default value specified"""
    xml_string = """
    <root>
        <element value="Hello">27</element>
    </root>
    """

    processor = xml.dictionary(
        "root",
        [
            xml.string("element", attribute="value", required=False, default="Goodbye"),
        ],
    )

    expected = {
        "value": "Hello",
    }

    actual = xml.parse_from_string(processor, xml_string)

    assert expected == actual


def test_parse_attribute_missing():
    """Parses a missing attribute value"""
    xml_string = """
    <root>
        <element wrong-value="27">Hello</element>
    </root>
    """

    processor = xml.dictionary("root", [xml.integer("element", attribute="value")])

    with pytest.raises(xml.MissingValue):
        xml.parse_from_string(processor, xml_string)


def test_parse_attribute_missing_element():
    """Parses an attribute value with a missing element"""
    xml_string = """
    <root>
        <wrong-element value="27">Hello</wrong-element>
    </root>
    """

    processor = xml.dictionary("root", [xml.integer("element", attribute="value")])

    with pytest.raises(xml.MissingValue):
        xml.parse_from_string(processor, xml_string)


def test_parse_attribute_only():
    """Parses an attribute value that is the only value in an element"""
    xml_string = """
        <root>
            <element value="hello" />
        </root>
    """

    processor = xml.dictionary(
        "root",
        [
            xml.string("element", attribute="value"),
        ],
    )

    expected = {
        "value": "hello",
    }

    actual = xml.parse_from_string(processor, xml_string)

    assert expected == actual


def test_parse_boolean_invalid():
    """Parse an invalid boolean value"""
    xml_string = """
    <root>
        <value>hello</value>
    </root>
    """

    processor = xml.dictionary(
        "root",
        [
            xml.boolean("value"),
        ],
    )

    with pytest.raises(xml.InvalidPrimitiveValue):
        xml.parse_from_string(processor, xml_string)


def test_parse_dictionary_aliased():
    """Parses a dictionary value that is aliased"""
    xml_string = """
    <person>
        <name>John Doe</name>
        <demographics>
            <age>25</age>
            <gender>male</gender>
        </demographics>
    </person>
    """

    stats = xml.dictionary(
        "demographics",
        [
            xml.integer("age"),
            xml.string("gender"),
        ],
        alias="stats",
    )

    person = xml.dictionary(
        "person",
        [
            xml.string("name"),
            stats,
        ],
    )

    expected = {
        "name": "John Doe",
        "stats": {
            "age": 25,
            "gender": "male",
        },
    }

    actual = xml.parse_from_string(person, xml_string)

    assert expected == actual


def test_parse_dictionary_empty_optional():
    """Parse an empty, optional dictionary"""
    xml_string = """
    <person>
        <name>John Doe</name>
        <demographics>
            <age>25</age>
            <gender>male</gender>
        </demographics>
        <address />
    </person>
    """

    demographics = xml.dictionary(
        "demographics",
        [
            xml.integer("age"),
            xml.string("gender"),
        ],
    )

    address = xml.dictionary(
        "address",
        [
            xml.string("street", required=False),
            xml.integer("zip", required=False),
            xml.string("state", required=False),
        ],
    )

    person = xml.dictionary(
        "person",
        [
            xml.string("name"),
            demographics,
            address,
        ],
    )

    expected = {
        "name": "John Doe",
        "demographics": {
            "age": 25,
            "gender": "male",
        },
        "address": {"street": "", "zip": 0, "state": ""},
    }

    actual = xml.parse_from_string(person, xml_string)

    assert expected == actual


def test_parse_dictionary_missing():
    """Parse a missing dictionary"""
    xml_string = """
    <person>
        <name>John Doe</name>
        <demographics>
            <age>25</age>
            <gender>male</gender>
        </demographics>
    </person>
    """

    demographics = xml.dictionary(
        "demographics",
        [
            xml.integer("age"),
            xml.string("gender"),
        ],
    )

    address = xml.dictionary(
        "address",
        [
            xml.string("street"),
            xml.integer("zip"),
            xml.string("state"),
        ],
    )

    person = xml.dictionary(
        "person",
        [
            xml.string("name"),
            demographics,
            address,
        ],
    )

    with pytest.raises(xml.MissingValue):
        xml.parse_from_string(person, xml_string)


def test_parse_dictionary_missing_optional():
    """Parse missing optional dictionaries"""
    xml_string = """
    <person>
        <name>John Doe</name>
        <demographics>
            <age>25</age>
            <gender>male</gender>
        </demographics>
    </person>
    """

    demographics = xml.dictionary(
        "demographics",
        [
            xml.integer("age"),
            xml.string("gender"),
        ],
    )

    address = xml.dictionary(
        "address",
        [
            xml.string("street"),
            xml.integer("zip"),
            xml.string("state"),
        ],
        required=False,
    )

    person = xml.dictionary(
        "person",
        [
            xml.string("name"),
            demographics,
            address,
        ],
    )

    expected = {
        "name": "John Doe",
        "demographics": {
            "age": 25,
            "gender": "male",
        },
        "address": {},
    }

    actual = xml.parse_from_string(person, xml_string)

    assert expected == actual


def test_parse_dictionary_nested():
    """Parse nested dictionaries"""
    xml_string = """
    <person>
        <name>John Doe</name>
        <demographics>
            <age>25</age>
            <gender>male</gender>
        </demographics>
        <address>
            <street>123 ABC Street</street>
            <zip>123456</zip>
            <state>NY</state>
        </address>
    </person>
    """

    demographics = xml.dictionary(
        "demographics",
        [
            xml.integer("age"),
            xml.string("gender"),
        ],
    )

    address = xml.dictionary(
        "address",
        [
            xml.string("street"),
            xml.integer("zip"),
            xml.string("state"),
        ],
    )

    person = xml.dictionary(
        "person",
        [
            xml.string("name"),
            demographics,
            address,
        ],
    )

    expected = {
        "name": "John Doe",
        "demographics": {
            "age": 25,
            "gender": "male",
        },
        "address": {
            "street": "123 ABC Street",
            "zip": 123456,
            "state": "NY",
        },
    }

    actual = xml.parse_from_string(person, xml_string)

    assert expected == actual


def test_parse_dictionary_root_missing():
    """Parse a dictionary as root"""
    xml_string = """
    <wrong-root>
        <value>hello</value>
    </wrong-root>
    """

    processor = xml.dictionary(
        "root",
        [
            xml.string("value"),
        ],
    )

    with pytest.raises(xml.MissingValue):
        xml.parse_from_string(processor, xml_string)


def test_parse_dictionary_root_optional():
    """Parse a dictionary as root"""
    xml_string = """
    <wrong-root>
        <value>hello</value>
    </wrong-root>
    """

    processor = xml.dictionary(
        "root",
        [
            xml.string("value"),
        ],
        required=False,
    )

    expected = {}

    actual = xml.parse_from_string(processor, xml_string)

    assert expected == actual


def test_parse_float_invalid():
    """Parse an invalid float value"""
    xml_string = """
    <root>
        <value>hello</value>
    </root>
    """

    processor = xml.dictionary(
        "root",
        [
            xml.floating_point("value"),
        ],
    )

    with pytest.raises(xml.InvalidPrimitiveValue):
        xml.parse_from_string(processor, xml_string)


def test_parse_from_file(tmpdir):
    """Tests parsing an XML file"""
    xml_contents = """
    <root>
        <value>27</value>
    </root>
    """

    xml_file = tmpdir.join("data.xml")
    xml_file.write(xml_contents)

    processor = xml.dictionary(
        "root",
        [
            xml.integer("value"),
        ],
    )

    expected = {
        "value": 27,
    }

    actual = xml.parse_from_file(processor, xml_file.strpath)

    assert expected == actual


def test_parse_int_invalid():
    """Parse an invalid int value"""
    xml_string = """
    <root>
        <value>hello</value>
    </root>
    """

    processor = xml.dictionary(
        "root",
        [
            xml.integer("value"),
        ],
    )

    with pytest.raises(xml.InvalidPrimitiveValue):
        xml.parse_from_string(processor, xml_string)


def test_parse_namespace():
    """Parses an xml document with a namespace"""
    xml_string = """
    <root xmlns="http://www.w3.org/1999/xhtml">
        <message>Hello</message>
        <values>
            <int>321</int>
            <float>3.14</float>
        </values>
    </root>
    """

    processor = xml.dictionary(
        "root",
        [
            xml.string("message"),
            xml.dictionary("values", [xml.integer("int"), xml.floating_point("float")]),
        ],
    )

    expected = {"message": "Hello", "values": {"int": 321, "float": 3.14}}

    actual = xml.parse_from_string(processor, xml_string)

    assert expected == actual


def test_parse_primitive():
    """Parse primitve XML values"""
    xml_string = """
    <root>
        <boolean>true</boolean>
        <float>3.14</float>
        <int>1</int>
        <string>Hello, World</string>
    </root>
    """

    processor = xml.dictionary(
        "root",
        [
            xml.boolean("boolean"),
            xml.floating_point("float"),
            xml.integer("int"),
            xml.string("string"),
        ],
    )

    expected = {
        "boolean": True,
        "float": 3.14,
        "int": 1,
        "string": "Hello, World",
    }

    actual = xml.parse_from_string(processor, xml_string)

    assert expected == actual


def test_parse_primitive_aliased():
    """Parses primitive values with aliases"""
    xml_string = """
    <root>
        <boolean>true</boolean>
        <float>3.14</float>
        <int>1</int>
        <string>Hello, World</string>
    </root>
    """

    processor = xml.dictionary(
        "root",
        [
            xml.boolean("boolean", alias="b"),
            xml.floating_point("float", alias="f"),
            xml.integer("int", alias="i"),
            xml.string("string", alias="s"),
        ],
    )

    expected = {
        "b": True,
        "f": 3.14,
        "i": 1,
        "s": "Hello, World",
    }

    actual = xml.parse_from_string(processor, xml_string)

    assert expected == actual


def test_primitive_default():
    """Parses primitive values with defaults specified"""
    xml_string = """
    <root>
    </root>
    """

    processor = xml.dictionary(
        "root",
        [
            xml.boolean("boolean", required=False, default=False),
            xml.floating_point("float", required=False, default=0.0),
            xml.integer("int", required=False, default=0),
            xml.string("string", required=False, default=""),
        ],
    )

    expected = {
        "boolean": False,
        "float": 0.0,
        "int": 0,
        "string": "",
    }

    actual = xml.parse_from_string(processor, xml_string)

    assert expected == actual


def test_primitive_default_present():
    """Parses primitive values with defaults specified"""
    xml_string = """
    <root>
        <boolean>false</boolean>
        <float>3.14</float>
        <int>1</int>
        <string>Hello, World</string>
    </root>
    """

    processor = xml.dictionary(
        "root",
        [
            xml.boolean("boolean", required=False, default=True),
            xml.floating_point("float", required=False, default=0.0),
            xml.integer("int", required=False, default=0),
            xml.string("string", required=False, default=""),
        ],
    )

    expected = {
        "boolean": False,
        "float": 3.14,
        "int": 1,
        "string": "Hello, World",
    }

    actual = xml.parse_from_string(processor, xml_string)

    assert expected == actual


def test_parse_primitive_missing():
    """Parses a missing primitive value"""
    xml_string = """
    <root>
        <wrong-value>15</wrong-value>
    </root>
    """

    processor = xml.dictionary("root", [xml.integer("value")])

    with pytest.raises(xml.MissingValue):
        xml.parse_from_string(processor, xml_string)


def test_parse_primitive_root_parser():
    """Parse with a primitive-valued root element"""
    xml_string = """
    <root>15</root>
    """

    processor = xml.integer("root")

    with pytest.raises(xml.InvalidRootProcessor):
        xml.parse_from_string(processor, xml_string)


def test_parse_string_empty_optional():
    """Parse an empty, optional string"""
    xml_string = """
    <root>
        <message />
    </root>
    """

    processor = xml.dictionary("root", [xml.string("message", required=False)])

    expected = {
        "message": "",
    }

    actual = xml.parse_from_string(processor, xml_string)

    assert expected == actual


def test_parse_string_leave_whitespace():
    """Parses a string value without stripping whitespace"""
    xml_string = """
    <root>
        <value>    Hello, World! </value>
    </root>
    """

    processor = xml.dictionary("root", [xml.string("value", strip_whitespace=False)])

    expected = {
        "value": "    Hello, World! ",
    }

    actual = xml.parse_from_string(processor, xml_string)

    assert expected == actual


def test_parse_string_strip_whitespace():
    """Parses a string value stripping whitespace"""
    xml_string = """
    <root>
        <value>    Hello, World! </value>
    </root>
    """

    processor = xml.dictionary("root", [xml.string("value")])

    expected = {
        "value": "Hello, World!",
    }

    actual = xml.parse_from_string(processor, xml_string)

    assert expected == actual
