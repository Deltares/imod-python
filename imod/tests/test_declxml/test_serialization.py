# -*- coding: utf-8 -*-
"""Contains unit tests for serialization logic"""
import os

import pytest

from imod import declxml as xml

from .helpers import strip_xml


def test_array_serialize_aggregate():
    """Serialize an array of aggregate values"""
    value = {"people": [{"name": "Bob", "age": 27}, {"name": "Jane", "age": 25}]}

    processor = xml.dictionary(
        "root",
        [
            xml.array(
                xml.dictionary("person", [xml.string("name"), xml.integer("age")]),
                alias="people",
            )
        ],
    )

    expected = strip_xml(
        """
    <root>
        <person>
            <name>Bob</name>
            <age>27</age>
        </person>
        <person>
            <name>Jane</name>
            <age>25</age>
        </person>
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_array_serialize_array_of_arrays():
    """Tests serializing arrays of arrays"""
    value = {
        "results": [
            [3, 2, 4],
            [4, 3],
            [12, 32, 87, 9],
        ]
    }

    processor = xml.dictionary(
        "root",
        [
            xml.array(
                xml.array(xml.integer("value"), nested="test-run"), alias="results"
            )
        ],
    )

    expected = strip_xml(
        """
    <root>
        <test-run>
            <value>3</value>
            <value>2</value>
            <value>4</value>
        </test-run>
        <test-run>
            <value>4</value>
            <value>3</value>
        </test-run>
        <test-run>
            <value>12</value>
            <value>32</value>
            <value>87</value>
            <value>9</value>
        </test-run>
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_array_serialize_array_of_arrays_omit_empty():
    """Tests serializing arrays of arrays with omit empty option specified"""
    value = {
        "results": [
            [3, 2, 4],
            [],
            [12, 32, 87, 9],
        ]
    }

    processor = xml.dictionary(
        "root",
        [
            xml.array(
                xml.array(
                    xml.integer("value", required=False),
                    nested="test-run",
                    omit_empty=True,
                ),
                alias="results",
            )
        ],
    )

    # Empty arrays contained within arrays should never be omitted because we would
    # lose information when serialzing to XML
    expected = strip_xml(
        """
    <root>
        <test-run>
            <value>3</value>
            <value>2</value>
            <value>4</value>
        </test-run>
        <test-run />
        <test-run>
            <value>12</value>
            <value>32</value>
            <value>87</value>
            <value>9</value>
        </test-run>
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_array_serialize_missing():
    """Serialize a missing array"""
    value = {"message": "Hello", "data": []}

    processor = xml.dictionary(
        "root", [xml.string("message"), xml.array(xml.integer("value"), alias="data")]
    )

    with pytest.raises(xml.MissingValue):
        xml.serialize_to_string(processor, value)


def test_array_serialize_missing_optional():
    """Serialize a missing array"""
    value = {
        "message": "Hello",
    }

    processor = xml.dictionary(
        "root",
        [
            xml.string("message"),
            xml.array(xml.integer("value", required=False), alias="data"),
        ],
    )

    expected = strip_xml(
        """
    <root>
        <message>Hello</message>
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_array_serialize_missing_optional_nested():
    """Seralize a missing optional nested array"""
    value = {
        "message": "Hello",
        "data": [],
    }

    processor = xml.dictionary(
        "root",
        [
            xml.string("message"),
            xml.array(xml.integer("value", required=False), nested="data"),
        ],
    )

    expected = strip_xml(
        """
    <root>
        <message>Hello</message>
        <data />
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_array_serialize_omit_empty():
    """Seralizes an empty array with the omit_empty option"""
    value = {
        "message": "Hello",
        "data": [],
    }

    processor = xml.dictionary(
        "root",
        [
            xml.string("message"),
            xml.array(
                xml.integer("value", required=False), nested="data", omit_empty=True
            ),
        ],
    )

    expected = strip_xml(
        """
    <root>
        <message>Hello</message>
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_array_serialize_omit_empty_non_nested():
    """Seralizes an array with the omit_empty option"""
    with pytest.warns(UserWarning):
        # Should get a warning when specifying omit_empty for non-nested arrays
        xml.dictionary(
            "root",
            [
                xml.string("message"),
                xml.array(xml.integer("value", required=False), omit_empty=True),
            ],
        )


def test_array_serialize_omit_empty_required():
    """Seralizes an array with the omit_empty option"""
    with pytest.warns(UserWarning):
        # Should get a warning when specifying omit_empty for required arrays
        xml.dictionary(
            "root",
            [
                xml.string("message"),
                xml.array(xml.integer("value"), nested="data", omit_empty=True),
            ],
        )


def test_array_serialize_omit_empty_present():
    """Seralizes a non-empty array with the omit_empty option"""
    value = {
        "message": "Hello",
        "data": [3, 17],
    }

    processor = xml.dictionary(
        "root",
        [
            xml.string("message"),
            xml.array(
                xml.integer("value", required=False), nested="data", omit_empty=True
            ),
        ],
    )

    expected = strip_xml(
        """
    <root>
        <message>Hello</message>
        <data>
            <value>3</value>
            <value>17</value>
        </data>
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_array_serialize_missing_root():
    """Serialize a missing array"""
    value = []

    processor = xml.array(xml.integer("value"), nested="data")

    with pytest.raises(xml.MissingValue):
        xml.serialize_to_string(processor, value)


def test_array_serialize_nested():
    """Tests serializing nested arrays"""
    value = {
        "date": "3-21",
        "data-points": [
            21.1,
            1897.17,
            13.1,
        ],
    }

    processor = xml.dictionary(
        "root",
        [
            xml.string("date"),
            xml.array(xml.floating_point("value"), nested="data-points"),
        ],
    )

    expected = strip_xml(
        """
    <root>
        <date>3-21</date>
        <data-points>
            <value>21.1</value>
            <value>1897.17</value>
            <value>13.1</value>
        </data-points>
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_array_serialize_optinal_present():
    """Serializes an optional array that is present"""
    value = {"message": "Hello", "value": [1, 14]}

    processor = xml.dictionary(
        "root", [xml.string("message"), xml.array(xml.integer("value", required=False))]
    )

    expected = strip_xml(
        """
    <root>
        <message>Hello</message>
        <value>1</value>
        <value>14</value>
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_array_serialize_primitive():
    """Serialize an array of primitive values"""
    value = {"values": [14, 3, 17, 5]}

    processor = xml.dictionary(
        "root", [xml.array(xml.integer("value"), alias="values")]
    )

    expected = strip_xml(
        """
    <root>
        <value>14</value>
        <value>3</value>
        <value>17</value>
        <value>5</value>
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_array_serialize_primitive_omit_empty():
    """Serialize an array of primitive values with omit empty option specified"""
    value = {"values": ["Hello", "", "Hola", "Bonjour"]}

    processor = xml.dictionary(
        "root",
        [
            xml.array(
                xml.string("value", required=False, omit_empty=True), alias="values"
            )
        ],
    )

    # Even with the omit empty option specified, we should never omit
    # Falsey values contained within an array because that would cause
    # us to lose information from the original array.
    expected = strip_xml(
        """
    <root>
        <value>Hello</value>
        <value />
        <value>Hola</value>
        <value>Bonjour</value>
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_array_serialize_root():
    """Serialize an array that is the root"""
    value = [3.14, 13.7, 6.22]

    processor = xml.array(xml.floating_point("constant"), nested="constants")

    expected = strip_xml(
        """
    <constants>
        <constant>3.14</constant>
        <constant>13.7</constant>
        <constant>6.22</constant>
    </constants>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_array_serialize_root_not_nested():
    """Serialize an array that is the root"""
    value = [3.14, 13.7, 6.22]

    processor = xml.array(xml.floating_point("constant"))

    with pytest.raises(xml.InvalidRootProcessor):
        xml.serialize_to_string(processor, value)


def test_array_serialize_shared_element():
    """Serialize an array on an element shared with an attribute"""
    value = {"units": "grams", "results": [32.4, 3.11]}

    processor = xml.dictionary(
        "root",
        [
            xml.string("results", attribute="units"),
            xml.array(xml.floating_point("value"), nested="results"),
        ],
    )

    expected = strip_xml(
        """
    <root>
        <results units="grams">
            <value>32.4</value>
            <value>3.11</value>
        </results>
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_attribute_serialize():
    """Serialize an attribute value"""
    value = {"value": "Hello, World"}

    processor = xml.dictionary("root", [xml.string("element", attribute="value")])

    expected = strip_xml(
        """
    <root>
        <element value="Hello, World" />
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_attribute_serialize_aliased():
    """Serialize an aliased attribute"""
    value = {
        "pi": 3.14,
    }

    processor = xml.dictionary(
        "root",
        [
            xml.floating_point("constant", attribute="value", alias="pi"),
        ],
    )

    expected = strip_xml(
        """
    <root>
        <constant value="3.14" />
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_attribute_serialize_default_missing():
    """Serializes a missing attribute value with a default specified"""
    value = {
        "data": 123,
    }

    processor = xml.dictionary(
        "root",
        [
            xml.integer("data"),
            xml.string("data", attribute="units", required=False, default="feet"),
        ],
    )

    expected = strip_xml(
        """
    <root>
        <data units="feet">123</data>
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_attribute_serialize_default_present():
    """Serializes an attribute value with a default specified"""
    value = {"data": 123, "units": "miles"}

    processor = xml.dictionary(
        "root",
        [
            xml.integer("data"),
            xml.string("data", attribute="units", required=False, default="feet"),
        ],
    )

    expected = strip_xml(
        """
    <root>
        <data units="miles">123</data>
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_attribute_serialize_falsey():
    """Tests serializing a Falsey attribute value"""
    value = {
        "message": "",
        "data": 123,
    }

    processor = xml.dictionary(
        "root",
        [xml.integer("data"), xml.string("data", attribute="message", required=False)],
    )

    expected = strip_xml(
        """
    <root>
        <data message="">123</data>
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_attribute_serialize_falsey_omitted():
    """Tests serializing a Falsey attribute value"""
    value = {
        "message": "",
        "data": 123,
    }

    processor = xml.dictionary(
        "root",
        [
            xml.integer("data"),
            xml.string("data", attribute="message", required=False, omit_empty=True),
        ],
    )

    expected = strip_xml(
        """
    <root>
        <data>123</data>
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_attribute_serialize_missing():
    """Serialize a missing attribute value"""
    value = {
        "data": 123,
    }

    processor = xml.dictionary(
        "root", [xml.integer("data"), xml.string("data", attribute="units")]
    )

    with pytest.raises(xml.MissingValue):
        xml.serialize_to_string(processor, value)


def test_attribute_serialize_missing_empty():
    """Tests serializing a Falsey attribute value"""
    value = {
        "data": 123,
    }

    processor = xml.dictionary(
        "root",
        [
            xml.integer("data"),
            xml.string("data", attribute="message", required=False, omit_empty=True),
        ],
    )

    expected = strip_xml(
        """
    <root>
        <data>123</data>
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_attribute_serialize_multiple():
    """Serializing multiple attributes to the same element"""
    value = {
        "attribute_a": "Hello, World",
        "attribute_b": True,
        "data": 1,
        "message": "Hello, World",
    }

    processor = xml.dictionary(
        "root",
        [
            xml.integer("data"),
            xml.string("data", attribute="attribute_a"),
            xml.boolean("data", attribute="attribute_b"),
            xml.string("message"),
        ],
    )

    expected = strip_xml(
        """
    <root>
        <data attribute_a="Hello, World" attribute_b="True">1</data>
        <message>Hello, World</message>
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_dictionary_serialize_root_empty():
    """Serliazes an empty root dictionary"""
    value = {}

    processor = xml.dictionary("root", [xml.string("message")])

    with pytest.raises(xml.MissingValue):
        xml.serialize_to_string(processor, value)


def test_dictionary_serialize_nested():
    """Serializes nested dictionaries"""
    value = {
        "name": "John Doe",
        "demographics": {
            "age": 27,
            "gender": "male",
        },
        "favorites": {"food": "pizza", "color": "blue"},
    }

    processor = xml.dictionary(
        "root",
        [
            xml.string("name"),
            xml.dictionary("demographics", [xml.integer("age"), xml.string("gender")]),
            xml.dictionary("favorites", [xml.string("food"), xml.string("color")]),
        ],
    )

    expected = strip_xml(
        """
    <root>
        <name>John Doe</name>
        <demographics>
            <age>27</age>
            <gender>male</gender>
        </demographics>
        <favorites>
            <food>pizza</food>
            <color>blue</color>
        </favorites>
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_dictionary_serialize_nested_aliased():
    """Serializes nested aliased dictionaries"""
    value = {
        "name": "John Doe",
        "stats": {
            "age": 27,
            "gender": "male",
        },
    }

    processor = xml.dictionary(
        "root",
        [
            xml.string("name"),
            xml.dictionary(
                "demographics",
                [xml.integer("age"), xml.string("gender")],
                alias="stats",
            ),
        ],
    )

    expected = strip_xml(
        """
    <root>
        <name>John Doe</name>
        <demographics>
            <age>27</age>
            <gender>male</gender>
        </demographics>
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_dictionary_serialize_nested_missing():
    """Serializes nested dictionaries"""
    value = {
        "name": "John Doe",
        "demographics": {
            "age": 27,
            "gender": "male",
        },
    }

    processor = xml.dictionary(
        "root",
        [
            xml.string("name"),
            xml.dictionary("demographics", [xml.integer("age"), xml.string("gender")]),
            xml.dictionary("favorites", [xml.string("food"), xml.string("color")]),
        ],
    )

    with pytest.raises(xml.MissingValue):
        xml.serialize_to_string(processor, value)


def test_dictionary_serialize_nested_missing_optional():
    """Serializes nested dictionaries"""
    value = {
        "name": "John Doe",
        "demographics": {
            "age": 27,
            "gender": "male",
        },
    }

    processor = xml.dictionary(
        "root",
        [
            xml.string("name"),
            xml.dictionary("demographics", [xml.integer("age"), xml.string("gender")]),
            xml.dictionary(
                "favorites", [xml.string("food"), xml.string("color")], required=False
            ),
        ],
    )

    expected = strip_xml(
        """
    <root>
        <name>John Doe</name>
        <demographics>
            <age>27</age>
            <gender>male</gender>
        </demographics>
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_dictionary_serialize_nested_optional_present():
    """Serializes nested dictionaries"""
    value = {
        "name": "John Doe",
        "demographics": {
            "age": 27,
            "gender": "male",
        },
        "favorites": {"movie": "Monty Python"},
    }

    processor = xml.dictionary(
        "root",
        [
            xml.string("name"),
            xml.dictionary("demographics", [xml.integer("age"), xml.string("gender")]),
            xml.dictionary(
                "favorites",
                [
                    xml.string("movie"),
                ],
                required=False,
            ),
        ],
    )

    expected = strip_xml(
        """
    <root>
        <name>John Doe</name>
        <demographics>
            <age>27</age>
            <gender>male</gender>
        </demographics>
        <favorites>
            <movie>Monty Python</movie>
        </favorites>
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_dictionary_serialize_shared_element():
    """Serialize a dictionary to an element shared with an attribute value"""
    value = {"gender": "male", "person": {"name": "Bob", "age": 27}}

    processor = xml.dictionary(
        "root",
        [
            xml.string("person", attribute="gender"),
            xml.dictionary("person", [xml.string("name"), xml.integer("age")]),
        ],
    )

    expected = strip_xml(
        """
    <root>
        <person gender="male">
            <name>Bob</name>
            <age>27</age>
        </person>
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_primitive_omit_empty_and_required():
    """Create a processor with both required and omit empty specified"""
    with pytest.warns(UserWarning):
        xml.integer("value", required=True, omit_empty=True)


def test_primitive_serialize_aliased():
    """Serializes an aliased primitive value"""
    value = {
        "data": 456,
    }

    processor = xml.dictionary(
        "root",
        [
            xml.integer("element", alias="data"),
        ],
    )

    expected = strip_xml(
        """
    <root>
        <element>456</element>
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_primitive_serialize_default_missing():
    """Serializes a missing primitive value with a default specified"""
    value = {
        "number": 898,
    }

    processor = xml.dictionary(
        "root",
        [
            xml.string("message", required=False, default="Hello, World"),
        ],
    )

    expected = strip_xml(
        """
    <root>
        <message>Hello, World</message>
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_primitive_serialize_default_present():
    """Serializes a missing primitive value with a defualt specified"""
    value = {"message": "Hola, Mars"}

    processor = xml.dictionary(
        "root",
        [
            xml.string("message", required=False, default="Hello, World"),
        ],
    )

    expected = strip_xml(
        """
    <root>
        <message>Hola, Mars</message>
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_primitive_serialize_missing():
    """Serializes a missing primitive value"""
    value = {"data": 1}

    processor = xml.dictionary("root", [xml.string("message"), xml.integer("data")])

    with pytest.raises(xml.MissingValue):
        xml.serialize_to_string(processor, value)


def test_primitive_serialize_missing_omitted():
    """Serializes a missing primitive value"""
    value = {"data": 1}

    processor = xml.dictionary(
        "root",
        [xml.string("message", required=False, omit_empty=True), xml.integer("data")],
    )

    expected = strip_xml(
        """
    <root>
        <data>1</data>
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_primitive_serialize_root():
    """Serialize a primitive value as the root of the document"""
    value = "Hello"

    processor = xml.string("message")

    with pytest.raises(xml.InvalidRootProcessor):
        xml.serialize_to_string(processor, value)


def test_primitive_values_serialize():
    """Serializes primitive values"""
    value = {"boolean": True, "float": 3.14, "int": 1, "string": "Hello, World"}

    processor = xml.dictionary(
        "root",
        [
            xml.boolean("boolean"),
            xml.floating_point("float"),
            xml.integer("int"),
            xml.string("string"),
        ],
    )

    expected = strip_xml(
        """
    <root>
        <boolean>True</boolean>
        <float>3.14</float>
        <int>1</int>
        <string>Hello, World</string>
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_primitive_values_serialize_falsey():
    """Serialize false primitive values"""
    value = {"boolean": False, "float": 0.0, "int": 0, "string": ""}

    processor = xml.dictionary(
        "root",
        [
            xml.boolean("boolean", required=False),
            xml.floating_point("float", required=False),
            xml.integer("int", required=False),
            xml.string("string", required=False),
        ],
    )

    expected = strip_xml(
        """
    <root>
        <boolean>False</boolean>
        <float>0.0</float>
        <int>0</int>
        <string />
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_primitive_values_serialize_falsey_omitted():
    """Serialize false primitive values"""
    value = {"boolean": False, "float": 0.0, "int": 0, "string": ""}

    processor = xml.dictionary(
        "root",
        [
            xml.boolean("boolean", required=False, omit_empty=True),
            xml.floating_point("float", required=False, omit_empty=True),
            xml.integer("int", required=False, omit_empty=True),
            xml.string("string", required=False, omit_empty=True),
        ],
    )

    expected = strip_xml(
        """
    <root />
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_primitive_values_serialize_none_default():
    """Serialize primitive values where the default for the value is None"""
    value = {}

    processor = xml.dictionary(
        "root",
        [
            xml.boolean("boolean", required=False, default=None),
            xml.floating_point("float", required=False, default=None),
            xml.integer("int", required=False, default=None),
            xml.string("string", required=False, default=None),
        ],
        required=False,
    )

    expected = strip_xml(
        """
    <root>
        <boolean />
        <float />
        <int />
        <string />
    </root>
    """
    )

    actual = xml.serialize_to_string(processor, value)

    assert expected == actual


def test_serialize_pretty():
    """Serialize a pretty-formatted XML string"""
    value = {"name": "Bob", "age": 27}

    processor = xml.dictionary("root", [xml.string("name"), xml.integer("age")])

    expected = """<?xml version="1.0" encoding="utf-8"?>
<root>
    <name>Bob</name>
    <age>27</age>
</root>
"""

    actual = xml.serialize_to_string(processor, value, indent="    ")

    assert expected == actual


def test_serialize_to_file(tmpdir):
    """Serialize XML data to a file"""
    value = {"boolean": True, "float": 3.14, "int": 1, "string": "Hello, World"}

    processor = xml.dictionary(
        "root",
        [
            xml.boolean("boolean"),
            xml.floating_point("float"),
            xml.integer("int"),
            xml.string("string"),
        ],
    )

    expected = strip_xml(
        """
    <root>
        <boolean>True</boolean>
        <float>3.14</float>
        <int>1</int>
        <string>Hello, World</string>
    </root>
    """
    )

    xml_file_name = "data.xml"
    xml_file_path = os.path.join(tmpdir.strpath, xml_file_name)

    xml.serialize_to_file(processor, value, xml_file_path)

    # Ensure the file contents match what is expected.
    xml_file = tmpdir.join(xml_file_name)
    actual = xml_file.read()

    assert expected == actual
