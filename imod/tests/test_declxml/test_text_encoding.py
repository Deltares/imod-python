# -*- coding: utf-8 -*-
"""Tests handling of text encoding"""
import os

import pytest

from imod import declxml as xml

_PROCESSOR = xml.dictionary("root", [xml.string("value")])


_VALUE = {
    "value": "Hello, 世界!",
}

_XML_STRING = """<root><value>Hello, 世界!</value></root>"""

_XML_STRING_INDENTED = """<?xml version="1.0" encoding="utf-8"?>
<root>
    <value>Hello, 世界!</value>
</root>
"""

_ENCODINGS = [
    "utf-8",
    "utf-16",
    "utf-32",
    "cp950",
    "gb18030",
]


def test_parse_from_string():
    """Parse a unicode string"""
    actual = xml.parse_from_string(_PROCESSOR, _XML_STRING)

    assert _VALUE == actual


@pytest.mark.parametrize("encoding", _ENCODINGS)
def test_parse_from_file(tmpdir, encoding):
    """Tests parsing an XML file"""
    xml_file = tmpdir.join("data.xml")
    xml_file.write_text(_XML_STRING, encoding)

    actual = xml.parse_from_file(_PROCESSOR, xml_file.strpath, encoding=encoding)

    assert _VALUE == actual


@pytest.mark.parametrize("encoding", _ENCODINGS)
def test_serialize_to_file(tmpdir, encoding):
    xml_file_name = "data.xml"
    xml_file_path = os.path.join(tmpdir.strpath, xml_file_name)

    xml.serialize_to_file(_PROCESSOR, _VALUE, xml_file_path, encoding=encoding)

    xml_file = tmpdir.join(xml_file_name)
    actual = xml_file.read_text(encoding)

    assert _XML_STRING == actual


@pytest.mark.parametrize("encoding", _ENCODINGS)
def test_serialize_to_file_indented(tmpdir, encoding):
    xml_file_name = "data.xml"
    xml_file_path = os.path.join(tmpdir.strpath, xml_file_name)

    xml.serialize_to_file(
        _PROCESSOR, _VALUE, xml_file_path, indent="    ", encoding=encoding
    )

    xml_file = tmpdir.join(xml_file_name)
    actual = xml_file.read_text(encoding)

    assert _XML_STRING_INDENTED == actual


def test_serialize_to_string():
    """Serialize a value to a unicode string"""
    actual = xml.serialize_to_string(_PROCESSOR, _VALUE)

    assert _XML_STRING == actual


def test_serialize_to_string_indent():
    """Serialize a value to a unicode string"""
    actual = xml.serialize_to_string(_PROCESSOR, _VALUE, indent="    ")

    assert _XML_STRING_INDENTED == actual
