"""Contains helper functions forunit tests"""
import re

from imod import declxml as xml


def assert_can_roundtrip_xml_value(processor, value):
    """Assert that a value can be converted to and from XML."""
    xml_string = xml.serialize_to_string(processor, value)
    actual_value = xml.parse_from_string(processor, xml_string)

    assert value == actual_value


def strip_xml(xml_string):
    """Prepares the XML string so it can be compared to the actual serialized output"""
    # Strip internal whitespace between tags
    stripped = re.sub(r">\s+<", "><", xml_string)

    # Strip external whitespace
    return stripped.strip()
