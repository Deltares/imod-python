"""Tests parsing and serializing classes created with attrs."""
import attr

from imod import declxml as xml

from .helpers import assert_can_roundtrip_xml_value


@attr.s
class _BookWithDefaults(object):
    title = attr.ib(default=None)
    author = attr.ib(default=None)


@attr.s
class _BookWithNoDefaults(object):
    title = attr.ib()
    author = attr.ib()


@attr.s(frozen=True)
class _FrozenBook(object):
    title = attr.ib(default=None)
    author = attr.ib(default=None)


def test_can_parse_and_serialize_attrs_class_with_defaults():
    """Parse and serialize an attrs class."""
    processor = xml.user_object(
        "book",
        _BookWithDefaults,
        [
            xml.string("title"),
            xml.string("author"),
        ],
    )

    value = _BookWithDefaults(title="The Three Body Problem", author="Liu Cixin")

    assert_can_roundtrip_xml_value(processor, value)


def test_can_parse_and_serialize_attrs_class_with_no_defaults():
    """Parse and serialize an attrs class."""
    processor = xml.user_object(
        "book",
        _BookWithNoDefaults,
        [
            xml.string("title"),
            xml.string("author"),
        ],
    )

    value = _BookWithNoDefaults(title="The Three Body Problem", author="Liu Cixin")

    assert_can_roundtrip_xml_value(processor, value)


def test_can_parse_and_serialize_frozen_attrs_class():
    """Parse and serialize a frozen attrs class."""
    processor = xml.user_object(
        "book",
        _FrozenBook,
        [
            xml.string("title"),
            xml.string("author"),
        ],
    )

    value = _FrozenBook(title="The Three Body Problem", author="Liu Cixin")

    assert_can_roundtrip_xml_value(processor, value)
