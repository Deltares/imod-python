"""Tests using hooks for validation"""
from collections import namedtuple

import pytest

from imod import declxml as xml

from .helpers import strip_xml

_UserTuple = namedtuple(
    "_UserTuple",
    [
        "name",
        "age",
    ],
)


class _UserClass(object):
    def __init__(self, name=None, age=None):
        self.name = name
        self.age = age

    def __eq__(self, other):
        return (
            isinstance(other, _UserClass)
            and other.name == self.name
            and other.age == self.age
        )

    def __repr__(self):
        return "_UserClass(name='{}', age={})".format(self.name, self.age)


class _ValidationError(Exception):
    """Custom validation error class"""


class TestCustomErrorMessage(object):
    """Provide custom validation error messages."""

    def test_array_non_root(self):
        """Custom error message for array values."""
        processor = xml.dictionary(
            "data",
            [xml.array(xml.integer("value"), nested="values", hooks=self._hooks)],
        )

        xml_string = strip_xml(
            """
        <data>
            <values>
                <value>1</value>
            </values>
        </data>
        """
        )

        value = {
            "values": [1],
        }

        location = "data/values"

        self._assert_error_message(processor, value, xml_string, location)

    def test_array_root(self):
        """Custom error message for array values."""
        processor = xml.array(xml.integer("value"), nested="data", hooks=self._hooks)

        xml_string = strip_xml(
            """
        <data>
            <value>1</value>
        </data>
        """
        )

        value = [1]

        location = "data"

        self._assert_error_message(processor, value, xml_string, location)

    def test_dictionary_non_root(self):
        """Custom error message for dictionary values."""
        processor = xml.dictionary(
            "data",
            [
                xml.dictionary(
                    "user",
                    [
                        xml.string("name"),
                        xml.integer("age"),
                    ],
                    hooks=self._hooks,
                )
            ],
        )

        xml_string = strip_xml(
            """
        <data>
            <user>
                <name>Bob</name>
                <age>24</age>
            </user>
        </data>
        """
        )

        value = {
            "user": {
                "name": "Bob",
                "age": 24,
            }
        }

        location = "data/user"

        self._assert_error_message(processor, value, xml_string, location)

    def test_dictionary_root(self):
        """Custom error message for dictionary values."""
        processor = xml.dictionary(
            "data",
            [
                xml.string("name"),
                xml.integer("age"),
            ],
            hooks=self._hooks,
        )

        xml_string = strip_xml(
            """
        <data>
            <name>Bob</name>
            <age>24</age>
        </data>
        """
        )

        value = {
            "name": "Bob",
            "age": 24,
        }

        location = "data"

        self._assert_error_message(processor, value, xml_string, location)

    def test_named_tuple_non_root(self):
        """Custom error message for namedtuple values."""
        processor = xml.dictionary(
            "data",
            [
                xml.named_tuple(
                    "user",
                    _UserTuple,
                    [
                        xml.string("name"),
                        xml.integer("age"),
                    ],
                    hooks=self._hooks,
                )
            ],
        )

        xml_string = strip_xml(
            """
        <data>
            <user>
                <name>Bob</name>
                <age>24</age>
            </user>
        </data>
        """
        )

        value = {"user": _UserTuple(name="Bob", age=24)}

        location = "data/user"

        self._assert_error_message(processor, value, xml_string, location)

    def test_named_tuple_root(self):
        """Custom error message for namedtuple values."""
        processor = xml.named_tuple(
            "data",
            _UserTuple,
            [
                xml.string("name"),
                xml.integer("age"),
            ],
            hooks=self._hooks,
        )

        xml_string = strip_xml(
            """
        <data>
            <name>Bob</name>
            <age>24</age>
        </data>
        """
        )

        value = _UserTuple(name="Bob", age=24)

        location = "data"

        self._assert_error_message(processor, value, xml_string, location)

    def test_primitive(self):
        """Custom error message for primitive values."""
        processor = xml.dictionary("data", [xml.integer("value", hooks=self._hooks)])

        xml_string = strip_xml(
            """
        <data>
            <value>1</value>
        </data>
        """
        )

        value = {"value": 1}

        location = "data/value"

        self._assert_error_message(processor, value, xml_string, location)

    def test_user_object_non_root(self):
        """Custom error message for user object values."""
        processor = xml.dictionary(
            "data",
            [
                xml.user_object(
                    "user",
                    _UserClass,
                    [
                        xml.string("name"),
                        xml.integer("age"),
                    ],
                    hooks=self._hooks,
                )
            ],
        )

        xml_string = strip_xml(
            """
        <data>
            <user>
                <name>Bob</name>
                <age>24</age>
            </user>
        </data>
        """
        )

        value = {"user": _UserClass(name="Bob", age=24)}

        location = "data/user"

        self._assert_error_message(processor, value, xml_string, location)

    def test_user_object_root(self):
        """Custom error message for user object values."""
        processor = xml.user_object(
            "data",
            _UserClass,
            [
                xml.string("name"),
                xml.integer("age"),
            ],
            hooks=self._hooks,
        )

        xml_string = strip_xml(
            """
        <data>
            <name>Bob</name>
            <age>24</age>
        </data>
        """
        )

        value = _UserClass(name="Bob", age=24)

        location = "data"

        self._assert_error_message(processor, value, xml_string, location)

    @staticmethod
    def _assert_error_message(processor, value, xml_string, expected_location):
        with pytest.raises(_ValidationError) as parse_exception:
            xml.parse_from_string(processor, xml_string)

        actual_parse_message = str(parse_exception.value)
        print(actual_parse_message)
        assert actual_parse_message.endswith(expected_location)

        with pytest.raises(_ValidationError) as serialize_exception:
            xml.serialize_to_string(processor, value)

        actual_serialize_message = str(serialize_exception.value)
        assert actual_serialize_message.endswith(expected_location)

    @property
    def _hooks(self):
        def validate(state, _):
            state.raise_error(_ValidationError, "Invalid value")

        return xml.Hooks(
            after_parse=validate,
            before_serialize=validate,
        )


class TestValidateArray(object):
    """Use hooks to validate array values."""

    def test_invalid(self):
        """Invalid array value."""
        xml_string = strip_xml(
            """
        <data>
            <value>1</value>
            <value>3</value>
            <value>3</value>
        </data>
        """
        )

        value = [1, 3, 3]

        _assert_invalid(self._processor, value, xml_string)

    def test_valid(self):
        """Valid array value."""
        xml_string = strip_xml(
            """
        <data>
            <value>1</value>
            <value>2</value>
            <value>3</value>
        </data>
        """
        )

        value = [1, 2, 3]

        _assert_valid(self._processor, value, xml_string)

    @property
    def _processor(self):
        def validate(state, value):
            if len(value) != len(set(value)):
                state.raise_error(_ValidationError)
            return value

        hooks = xml.Hooks(
            after_parse=validate,
            before_serialize=validate,
        )

        processor = xml.array(xml.integer("value"), hooks=hooks, nested="data")

        return processor


class TestValidateDictionary(object):
    """Use hooks to validate dictionary values."""

    def test_invalid(self):
        """Invalid dictionary value."""
        xml_string = strip_xml(
            """
        <data>
            <a>5</a>
            <b>6</b>
        </data>
        """
        )

        value = {
            "a": 5,
            "b": 6,
        }

        _assert_invalid(self._processor, value, xml_string)

    def test_valid(self):
        """Valid dictionary value."""
        xml_string = strip_xml(
            """
        <data>
            <a>32</a>
            <b>67</b>
        </data>
        """
        )

        value = {
            "a": 32,
            "b": 67,
        }

        _assert_valid(self._processor, value, xml_string)

    @property
    def _processor(self):
        def validate(state, value):
            if value["a"] == 5 and value["b"] == 6:
                state.raise_error(_ValidationError)
            return value

        hooks = xml.Hooks(
            after_parse=validate,
            before_serialize=validate,
        )

        processor = xml.dictionary(
            "data",
            [
                xml.integer("a"),
                xml.integer("b"),
            ],
            hooks=hooks,
        )

        return processor


class TestValidateNamedTuple(object):
    """Use hooks for validating namedtuple values."""

    def test_invalid(self):
        """Invalid namedtuple value"""
        xml_string = strip_xml(
            """
        <user>
            <name>Bob</name>
            <age>24</age>
        </user>
        """
        )

        value = _UserTuple(name="Bob", age=24)

        _assert_invalid(self._processor, value, xml_string)

    def test_valid(self):
        """Valid namedtuple value"""
        xml_string = strip_xml(
            """
        <user>
            <name>Jill</name>
            <age>28</age>
        </user>
        """
        )

        value = _UserTuple(name="Jill", age=28)

        _assert_valid(self._processor, value, xml_string)

    @property
    def _processor(self):
        def validate(state, value):
            if value.name == "Bob" and value.age == 24:
                state.raise_error(_ValidationError)
            return value

        hooks = xml.Hooks(
            after_parse=validate,
            before_serialize=validate,
        )

        processor = xml.named_tuple(
            "user", _UserTuple, [xml.string("name"), xml.integer("age")], hooks=hooks
        )

        return processor


class TestValidatePrimitive(object):
    """Use hooks for validating primitive values."""

    def test_invalid(self):
        """Invalid primitive value"""
        xml_string = strip_xml(
            """
        <data>
            <value>-91</value>
        </data>
        """
        )

        value = {"value": -91}

        _assert_invalid(self._processor, value, xml_string)

    def test_valid(self):
        """Valid primitive value"""
        xml_string = strip_xml(
            """
        <data>
            <value>32</value>
        </data>
        """
        )

        value = {"value": 32}

        _assert_valid(self._processor, value, xml_string)

    @property
    def _processor(self):
        def validate(state, value):
            if value < 0:
                state.raise_error(_ValidationError)

            return value

        hooks = xml.Hooks(after_parse=validate, before_serialize=validate)

        processor = xml.dictionary("data", [xml.integer("value", hooks=hooks)])

        return processor


class TestValidateUserObject(object):
    """Use hooks for validating user object values."""

    def test_invalid(self):
        """Invalid namedtuple value"""
        xml_string = strip_xml(
            """
        <user>
            <name>Bob</name>
            <age>24</age>
        </user>
        """
        )

        value = _UserClass(name="Bob", age=24)

        _assert_invalid(self._processor, value, xml_string)

    def test_valid(self):
        """Valid namedtuple value"""
        xml_string = strip_xml(
            """
        <user>
            <name>Jill</name>
            <age>28</age>
        </user>
        """
        )

        value = _UserClass(name="Jill", age=28)

        _assert_valid(self._processor, value, xml_string)

    @property
    def _processor(self):
        def validate(state, value):
            if value.name == "Bob" and value.age == 24:
                state.raise_error(_ValidationError)
            return value

        hooks = xml.Hooks(
            after_parse=validate,
            before_serialize=validate,
        )

        processor = xml.user_object(
            "user", _UserClass, [xml.string("name"), xml.integer("age")], hooks=hooks
        )

        return processor


def test_aggregate_missing_hooks():
    """Process with missing aggregate hooks."""
    hooks = xml.Hooks(after_parse=None, before_serialize=None)

    processor = xml.dictionary(
        "data", [xml.integer("a"), xml.integer("b")], hooks=hooks
    )

    xml_string = strip_xml(
        """
    <data>
        <a>1</a>
        <b>2</b>
    </data>
    """
    )

    value = {
        "a": 1,
        "b": 2,
    }

    _assert_valid(processor, value, xml_string)


def test_primitive_missing_hooks():
    """Process primitive value with missing hooks."""
    hooks = xml.Hooks(after_parse=None, before_serialize=None)

    processor = xml.dictionary("data", [xml.integer("value", hooks=hooks)])

    xml_string = strip_xml(
        """
    <data>
        <value>1</value>
    </data>
    """
    )

    value = {"value": 1}

    _assert_valid(processor, value, xml_string)


def test_processor_locations_parsing():
    """Get processor location in hooks callback."""
    expected_locations = [
        xml.ProcessorLocation(element_path="data", array_index=None),
        xml.ProcessorLocation(element_path="value", array_index=None),
    ]

    def trace(state, _):
        assert isinstance(state, xml.ProcessorStateView)
        assert expected_locations == list(state.locations)

    hooks = xml.Hooks(
        after_parse=trace,
        before_serialize=trace,
    )

    processor = xml.dictionary(
        "data",
        [
            xml.integer("value", hooks=hooks),
        ],
    )

    xml_string = strip_xml(
        """
    <data>
        <value>1</value>
    </data>
    """
    )

    value = {"value": 1}

    xml.parse_from_string(processor, xml_string)
    xml.serialize_to_string(processor, value)


def _assert_invalid(processor, value, xml_string):
    """Assert the processor rejects the XML and value as invalid."""
    with pytest.raises(_ValidationError):
        xml.parse_from_string(processor, xml_string)

    with pytest.raises(_ValidationError):
        xml.serialize_to_string(processor, value)


def _assert_valid(processor, value, xml_string):
    """Assert the processor accepts the XML and value as valid."""
    actual_value = xml.parse_from_string(processor, xml_string)
    assert value == actual_value

    actual_xml_string = xml.serialize_to_string(processor, value)
    assert xml_string == actual_xml_string
