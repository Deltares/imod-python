# -*- coding: utf-8 -*-
"""
declxml is a library for declaratively processing XML documents.

Processors
---------------
The declxml library uses processors to define the structure of XML documents. Processors can be
divided into two categories: *primitive processors* and *aggregate processors*.

Primitive Processors
----------------------
Primitive processors are used to parse and serialize simple, primitive values. The following module
functions are used to construct primitive processors:

.. autofunction:: declxml.boolean
.. autofunction:: declxml.floating_point
.. autofunction:: declxml.integer
.. autofunction:: declxml.string

Aggregate Processors
---------------------
Aggregate processors are composed of other processors. These processors are used for values such as
dictionaries, arrays, and user objects.

.. autofunction:: declxml.array
.. autofunction:: declxml.dictionary
.. autofunction:: declxml.named_tuple
.. autofunction:: declxml.user_object

Hooks
------------------
.. autoclass:: declxml.Hooks
.. autoclass:: declxml.ProcessorStateView
    :members:

Parsing
----------
.. autofunction:: declxml.parse_from_file
.. autofunction:: declxml.parse_from_string

Serialization
---------------
.. autofunction:: declxml.serialize_to_file
.. autofunction:: declxml.serialize_to_string

Exceptions
------------
.. autoexception:: declxml.XmlError
.. autoexception:: declxml.InvalidPrimitiveValue
.. autoexception:: declxml.InvalidRootProcessor
.. autoexception:: declxml.MissingValue
"""
import sys
import warnings
import xml.etree.ElementTree as ET
from io import open
from typing import (  # noqa pylint: disable=unused-import
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    NoReturn,
    Optional,
    Text,
    Tuple,
    Type,
    Union,
)
from xml.dom import minidom  # type: ignore

_PY2 = sys.version_info[0] == 2


class XmlError(Exception):
    """Base error class representing errors processing XML data."""


class InvalidPrimitiveValue(XmlError):
    """Represents errors due to invalid primitive values."""


class InvalidRootProcessor(XmlError):
    """Represents errors due to invalid root processors."""


class MissingValue(XmlError):
    """Represents errors due to missing required values."""


ProcessorLocation = NamedTuple(
    "ProcessorLocation",
    [
        (
            "element_path",
            Text,
        ),  # Path to the element relative to the previous location.
        (
            "array_index",
            Optional[int],
        ),  # Index of the element if in an array, None if not in an array.
    ],
)


class ProcessorStateView(object):
    """Provides an immutable view of the processor state."""

    def __init__(
        self, processor_state  # type: _ProcessorState
    ):
        # type: (...) -> None
        """
        Create a new processor state view.

        :param processor_state: Underlying processor state object.
        """
        self._processor_state = processor_state

    @property
    def locations(self):
        # type: () -> Iterable[ProcessorLocation]
        """
        Get iterator of the ProcessorLocations visited by the processor.

        Represents the current location of the processor in the XML document.

        A ProcessorLocation represents a single location of a processor in an
        XML document. It is a namedtuple with two fields: element_path and
        array_index. The element_path field contains the path to the element
        in the XML document relative to the previous location in the list of
        locations. The array_index field contains the index of the element if
        it is in an array, otherwise it is None if the element is not in an
        array.

        :return: Iterator of ProcessorLocation objects.
        """
        return self._processor_state.locations

    def raise_error(
        self,
        exception_type,  # type: Type[Exception]
        message="",  # type: Text
    ):
        # type: (...) -> None
        """
        Raise an error with the processor state included in the error message.

        :param exception_type: Type of exception to raise
        :param message: Error message
        """
        self._processor_state.raise_error(exception_type, message)

    def __repr__(self):
        return repr(self._processor_state)


HookFunction = Callable[[ProcessorStateView, Any], Any]


class Hooks(object):
    """
    Contains functions to be invoked during parsing and serialization.

    A Hooks object contains two functions: after_parse and before_serialize. Both of these
    functions receive two arguments and should return one value.

    As their first argument, both functions will receive a ProcessorStateView object representing
    the current state of the processor when the function is invoked.

    The after_parse function will receive as its second argument the value parsed by the processor
    from the XML data during parsing. This function must return a value that will be used by the
    processor as its parse result.

    Similarly, the before_serialize function will receive as its second argument the value to be
    serialized to XML by the processor during serialization. This function must return a value
    that the processor will serialize to XML.

    Both the after_parse and before_serialize functions are optional.
    """

    def __init__(
        self,
        after_parse=None,  # type: Optional[HookFunction]
        before_serialize=None,  # type: Optional[HookFunction]
    ):
        # type: (...) -> None
        """
        Create a new Hooks object.

        :param after_parse: Function to be invoked after a value has been parsed.
        :param before_serialize: Function to be invoked before a value is serialized.
        """
        self.after_parse = after_parse
        self.before_serialize = before_serialize


class Processor(object):  # pragma: no cover
    """Abstract protocol for processors."""

    @property
    def alias(self):
        # type: (...) -> Text
        """Get processor's alias."""
        pass

    @property
    def element_path(self):
        # type: (...) -> Text
        """Get path to processor's element."""
        pass

    @property
    def required(self):
        # type: (...) -> bool
        """Get whether the processor's value is required."""
        pass

    def parse_at_element(
        self,
        element,  # type: ET.Element
        state,  # type: _ProcessorState
    ):
        # type: (...) -> Any
        """Parse value at element."""
        pass

    def parse_from_parent(
        self,
        parent,  # type: ET.Element
        state,  # type: _ProcessorState
    ):
        # type: (...) -> Any
        """Parse value from parent element."""
        pass

    def serialize(
        self,
        value,  # type: Any
        state,  # type: _ProcessorState
    ):
        # type: (...) -> ET.Element
        """Serialize a value to an XML element."""
        pass

    def serialize_on_parent(
        self,
        parent,  # type: ET.Element
        value,  # type: Any
        state,  # type: _ProcessorState
    ):
        # type: (...) -> None
        """Serialize value on parent element."""
        pass


class RootProcessor(Processor):  # pragma: no cover
    """Abstract protocol for root processors."""

    def parse_at_root(
        self,
        root,  # type: ET.Element
        state,  # type: _ProcessorState
    ):
        # type: (...) -> Any
        """Parse value from root element."""
        pass


def parse_from_file(
    root_processor,  # type: RootProcessor
    xml_file_path,  # type: Text
    encoding="utf-8",  # type: Text
):
    # type: (...) -> Any
    """
    Parse the XML file using the processor starting from the root of the document.

    :param root_processor: Root processor of the XML document.
    :param xml_file_path: Path to XML file to parse.
    :param encoding: Encoding of the file.

    :return: Parsed value.
    """
    with open(xml_file_path, "r", encoding=encoding) as xml_file:
        xml_string = xml_file.read()

    parsed_value = parse_from_string(root_processor, xml_string)

    return parsed_value


def parse_from_string(
    root_processor,  # type: RootProcessor
    xml_string,  # type: Text
):
    # type: (...) -> Any
    """
    Parse the XML string using the processor starting from the root of the document.

    :param xml_string: XML string to parse.

    See also :func:`declxml.parse_from_file`
    """
    if not _is_valid_root_processor(root_processor):
        raise InvalidRootProcessor("Invalid root processor")

    parseable_xml_string = xml_string  # type: Union[Text, bytes]
    if _PY2 and isinstance(xml_string, Text):
        parseable_xml_string = xml_string.encode("utf-8")

    root = ET.fromstring(parseable_xml_string)
    _xml_namespace_strip(root)

    state = _ProcessorState()
    state.push_location(root_processor.element_path)
    return root_processor.parse_at_root(root, state)


def serialize_to_file(
    root_processor,  # type: RootProcessor
    value,  # type: Any
    xml_file_path,  # type: Text
    encoding="utf-8",  # type: Text
    indent=None,  # type: Optional[Text]
):
    # type: (...) -> None
    """
    Serialize the value to an XML file using the root processor.

    :param root_processor: Root processor of the XML document.
    :param value: Value to serialize.
    :param xml_file_path: Path to the XML file to which the serialized value will be written.
    :param encoding: Encoding of the file.
    :param indent: If specified, then the XML will be formatted with the specified indentation.
    """
    serialized_value = serialize_to_string(root_processor, value, indent)

    with open(xml_file_path, "w", encoding=encoding) as xml_file:
        xml_file.write(serialized_value)


def serialize_to_string(
    root_processor,  # type: RootProcessor
    value,  # type: Any
    indent=None,  # type: Optional[Text]
):
    # type: (...) -> Text
    """
    Serialize the value to an XML string using the root processor.

    :return: The serialized XML string.

    See also :func:`declxml.serialize_to_file`
    """
    if not _is_valid_root_processor(root_processor):
        raise InvalidRootProcessor("Invalid root processor")

    state = _ProcessorState()
    state.push_location(root_processor.element_path)

    root = root_processor.serialize(value, state)

    state.pop_location()

    # Always encode to UTF-8 because element tree does not support other
    # encodings in earlier Python versions. See: https://bugs.python.org/issue1767933
    serialized_value = ET.tostring(root, encoding="utf-8")

    # Since element tree does not support pretty printing XML, we use minidom to do the pretty
    # printing
    if indent:
        serialized_value = minidom.parseString(serialized_value).toprettyxml(
            indent=indent, encoding="utf-8"
        )

    return serialized_value.decode("utf-8")


def array(
    item_processor,  # type: Processor
    alias=None,  # type: Optional[Text]
    nested=None,  # type: Optional[Text]
    omit_empty=False,  # type: bool
    hooks=None,  # type: Optional[Hooks]
):
    # type: (...) -> RootProcessor
    """
    Create an array processor that can be used to parse and serialize array data.

    XML arrays may be nested within an array element, or they may be embedded
    within their parent. A nested array would look like:

    .. sourcecode:: xml

        <root-element>
            <some-element>ABC</some-element>
            <nested-array>
                <array-item>0</array-item>
                <array-item>1</array-item>
            </nested-array>
        </root-element>

    The corresponding embedded array would look like:

    .. sourcecode:: xml

        <root-element>
            <some-element>ABC</some-element>
            <array-item>0</array-item>
            <array-item>1</array-item>
        </root-element>


    An array is considered required when its item processor is configured as being
    required.

    :param item_processor: A declxml processor object for the items of the array.
    :param alias: If specified, the name given to the array when read from XML.
        If not specified, then the name of the item processor is used instead.
    :param nested: If the array is a nested array, then this should be the name of
        the element under which all array items are located. If not specified, then
        the array is treated as an embedded array. Can also be specified using supported
        XPath syntax.
    :param omit_empty: If True, then nested arrays will be omitted when serializing if
        they are empty. Only valid when nested is specified. Note that an empty array
        may only be omitted if it is not itself contained within an array. That is,
        for an array of arrays, any empty arrays in the outer array will always be
        serialized to prevent information about the original array from being lost
        when serializing.
    :param hooks: A Hooks object.

    :return: A declxml processor object.
    """
    processor = _Array(item_processor, alias, nested, omit_empty)
    return _processor_wrap_if_hooks(processor, hooks)


def boolean(
    element_name,  # type: Text
    attribute=None,  # type: Optional[Text]
    required=True,  # type: bool
    alias=None,  # type: Optional[Text]
    default=False,  # type: Optional[bool]
    omit_empty=False,  # type: bool
    hooks=None,  # type: Optional[Hooks]
):
    # type: (...) -> Processor
    """
    Create a processor for boolean values.

    :param element_name: Name of the XML element containing the value. Can also be specified
        using supported XPath syntax.
    :param attribute: If specified, then the value is searched for under the
        attribute within the element specified by element_name. If not specified,
        then the value is searched for as the contents of the element specified by
        element_name.
    :param required: Indicates whether the value is required when parsing and serializing.
    :param alias: If specified, then this is used as the name of the value when read from
        XML. If not specified, then the element_name is used as the name of the value.
    :param default: Default value to use if the element is not present. This option is only
        valid if required is specified as False.
    :param omit_empty: If True, then Falsey values will be omitted when serializing to XML. Note
        that Falsey values are never omitted when they are elements of an array. Falsey values can
        be omitted only when they are standalone elements.
    :param hooks: A Hooks object.

    :return: A declxml processor object.
    """
    return _PrimitiveValue(
        element_name,
        _parse_boolean,
        attribute,
        required,
        alias,
        default,
        omit_empty,
        hooks,
    )


def dictionary(
    element_name,  # type: Text
    children,  # type: List[Processor]
    required=True,  # type: bool
    alias=None,  # type: Optional[Text]
    hooks=None,  # type: Optional[Hooks]
):
    # type: (...) -> RootProcessor
    """
    Create a processor for dictionary values.

    :param element_name: Name of the XML element containing the dictionary value. Can also be
        specified using supported XPath syntax.
    :param children: List of declxml processor objects for processing the children
        contained within the dictionary.
    :param required: Indicates whether the value is required when parsing and serializing.
    :param alias: If specified, then this is used as the name of the value when read from
        XML. If not specified, then the element_name is used as the name of the value.
    :param hooks: A Hooks object.

    :return: A declxml processor object.
    """
    processor = _Dictionary(element_name, children, required, alias)
    return _processor_wrap_if_hooks(processor, hooks)


def floating_point(
    element_name,  # type: Text
    attribute=None,  # type: Optional[Text]
    required=True,  # type: bool
    alias=None,  # type: Optional[Text]
    default=0.0,  # type: Optional[float]
    omit_empty=False,  # type: bool
    hooks=None,  # type: Optional[Hooks]
):
    # type: (...) -> Processor
    """
    Create a processor for floating point values.

    See also :func:`declxml.boolean`
    """
    value_parser = _number_parser(float)
    return _PrimitiveValue(
        element_name,
        value_parser,
        attribute,
        required,
        alias,
        default,
        omit_empty,
        hooks,
    )


def integer(
    element_name,  # type: Text
    attribute=None,  # type: Optional[Text]
    required=True,  # type: bool
    alias=None,  # type: Optional[Text]
    default=0,  # type: Optional[int]
    omit_empty=False,  # type: bool
    hooks=None,  # type: Optional[Hooks]
):
    # type: (...) -> Processor
    """
    Create a processor for integer values.

    See also :func:`declxml.boolean`
    """
    value_parser = _number_parser(int)
    return _PrimitiveValue(
        element_name,
        value_parser,
        attribute,
        required,
        alias,
        default,
        omit_empty,
        hooks,
    )


def named_tuple(
    element_name,  # type: Text
    tuple_type,  # type: Type[Tuple]
    child_processors,  # type: List[Processor]
    required=True,  # type: bool
    alias=None,  # type: Optional[Text]
    hooks=None,  # type: Optional[Hooks]
):
    # type: (...) -> RootProcessor
    """
    Create a processor for namedtuple values.

    :param tuple_type: The namedtuple type.

    See also :func:`declxml.dictionary`
    """
    converter = _named_tuple_converter(tuple_type)
    processor = _Aggregate(element_name, converter, child_processors, required, alias)
    return _processor_wrap_if_hooks(processor, hooks)


def string(
    element_name,  # type: Text
    attribute=None,  # type: Optional[Text]
    required=True,  # type: bool
    alias=None,  # type: Optional[Text]
    default="",  # type: Optional[Text]
    omit_empty=False,  # type: bool
    strip_whitespace=True,  # type: bool
    hooks=None,  # type: Optional[Hooks]
):
    # type: (...) -> Processor
    """
    Create a processor for string values.

    :param strip_whitespace: Indicates whether leading and trailing whitespace should be stripped
        from parsed string values.

    See also :func:`declxml.boolean`
    """
    value_parser = _string_parser(strip_whitespace)
    return _PrimitiveValue(
        element_name,
        value_parser,
        attribute,
        required,
        alias,
        default,
        omit_empty,
        hooks,
    )


def user_object(
    element_name,  # type: Text
    cls,  # type: Type[Any]
    child_processors,  # type: List[Processor]
    required=True,  # type: bool
    alias=None,  # type: Optional[Text]
    hooks=None,  # type: Optional[Hooks]
):
    # type: (...) -> RootProcessor
    """
    Create a processor for user objects.

    :param cls: Class object with a no-argument constructor or other callable no-argument object.

    See also :func:`declxml.dictionary`
    """
    converter = _user_object_converter(cls)
    processor = _Aggregate(element_name, converter, child_processors, required, alias)
    return _processor_wrap_if_hooks(processor, hooks)


# Defines pair of functions to convert between aggregates and dictionaries
_AggregateConverter = NamedTuple(
    "_AggregateConverter",
    [
        ("from_dict", Callable[[Dict], Any]),
        ("to_dict", Callable[[Any], Dict]),
    ],
)


class _Aggregate(RootProcessor):
    """An XML processor for processing aggregates."""

    def __init__(
        self,
        element_path,  # type: Text
        converter,  # type: _AggregateConverter
        child_processors,  # type: List[Processor]
        required=True,  # type: bool
        alias=None,  # type: Optional[Text]
    ):
        # type: (...) -> None
        self._element_path = element_path
        self._converter = converter
        self._required = required
        self._dictionary = _Dictionary(element_path, child_processors, required, alias)
        if alias:
            self._alias = alias
        else:
            self._alias = element_path

    @property
    def alias(self):
        # type: (...) -> Text
        """Get processor's alias."""
        return self._alias

    @property
    def element_path(self):
        # type: (...) -> Text
        """Get path to processor's element."""
        return self._element_path

    @property
    def required(self):
        # type: (...) -> bool
        """Get whether the processor's value is required."""
        return self._required

    def parse_at_element(
        self,
        element,  # type: ET.Element
        state,  # type: _ProcessorState
    ):
        # type: (...) -> Any
        """Parse the provided element as an aggregate."""
        parsed_dict = self._dictionary.parse_at_element(element, state)
        return self._converter.from_dict(parsed_dict)

    def parse_at_root(
        self,
        root,  # type: ET.Element
        state,  # type: _ProcessorState
    ):
        # type: (...) -> Any
        """Parse the root XML element as an aggregate."""
        parsed_dict = self._dictionary.parse_at_root(root, state)
        return self._converter.from_dict(parsed_dict)

    def parse_from_parent(
        self,
        parent,  # type: ET.Element
        state,  # type: _ProcessorState
    ):
        # type: (...) -> Any
        """Parse the aggregate from the provided parent XML element."""
        parsed_dict = self._dictionary.parse_from_parent(parent, state)
        return self._converter.from_dict(parsed_dict)

    def serialize(
        self,
        value,  # type: Any
        state,  # type: _ProcessorState
    ):
        # type: (...) -> ET.Element
        """Serialize the value to a new element and returns the element."""
        dict_value = self._converter.to_dict(value)
        return self._dictionary.serialize(dict_value, state)

    def serialize_on_parent(
        self,
        parent,  # type: ET.Element
        value,  # type: Any
        state,  # type: _ProcessorState
    ):
        # type: (...) -> None
        """Serialize the value and adds it to the parent."""
        dict_value = self._converter.to_dict(value)
        self._dictionary.serialize_on_parent(parent, dict_value, state)


class _Array(RootProcessor):
    """An XML processor for Array values."""

    def __init__(
        self,
        item_processor,  # type: Processor
        alias=None,  # type: Optional[Text]
        nested=None,  # type: Optional[Text]
        omit_empty=False,  # type: bool
    ):
        # type: (...) -> None
        self._item_processor = item_processor
        self._nested = nested
        self._required = item_processor.required

        if alias:
            self._alias = alias
        elif nested:
            self._alias = nested
        else:
            self._alias = item_processor.alias

        if self._nested:
            self._element_path = self._nested
        else:
            self._element_path = "."  # Array is embedded directly on parent

        self._item_path = self.element_path + "/" + self._item_processor.element_path

        if not nested or self.required:
            self.omit_empty = False
            if omit_empty:
                warnings.warn(
                    "omit_empty ignored for non-nested and/or required arrays"
                )
        else:
            self.omit_empty = omit_empty

    @property
    def alias(self):
        # type: (...) -> Text
        """Get processor's alias."""
        return self._alias

    @property
    def element_path(self):
        # type: (...) -> Text
        """Get path to processor's element."""
        return self._element_path

    @property
    def required(self):
        # type: (...) -> bool
        """Get whether the processor's value is required."""
        return self._required

    def parse_at_element(
        self,
        element,  # type: ET.Element
        state,  # type: _ProcessorState
    ):
        # type: (...) -> Any
        """Parse the provided element as an array."""
        item_iter = element.findall(self._item_processor.element_path)
        return self._parse(item_iter, state)

    def parse_at_root(
        self,
        root,  # type: ET.Element
        state,  # type: _ProcessorState
    ):
        # type: (...) -> Any
        """Parse the root XML element as an array."""
        if not self._nested:
            raise InvalidRootProcessor(
                'Non-nested array "{}" cannot be root element'.format(self.alias)
            )

        parsed_array = []  # type: List

        array_element = _element_find_from_root(root, self._nested)
        if array_element is not None:
            parsed_array = self.parse_at_element(array_element, state)
        elif self.required:
            raise MissingValue(
                'Missing required array at root: "{}"'.format(self._nested)
            )

        return parsed_array

    def parse_from_parent(
        self,
        parent,  # type: ET.Element
        state,  # type: _ProcessorState
    ):
        # type: (...) -> Any
        """Parse the array data from the provided parent XML element."""
        item_iter = parent.findall(self._item_path)
        return self._parse(item_iter, state)

    def serialize(
        self,
        value,  # type: Any
        state,  # type: _ProcessorState
    ):
        # type: (...) -> ET.Element
        """Serialize the value into a new Element object and return it."""
        if self._nested is None:
            state.raise_error(
                InvalidRootProcessor,
                'Cannot directly serialize a non-nested array "{}"'.format(self.alias),
            )

        if not value and self.required:
            state.raise_error(
                MissingValue, 'Missing required array: "{}"'.format(self.alias)
            )

        start_element, end_element = _element_path_create_new(self._nested)
        self._serialize(end_element, value, state)

        return start_element

    def serialize_on_parent(
        self,
        parent,  # type: ET.Element
        value,  # type: Any
        state,  # type: _ProcessorState
    ):
        # type: (...) -> None
        """Serialize the value and append it to the parent element."""
        if not value and self.required:
            state.raise_error(
                MissingValue, 'Missing required array: "{}"'.format(self.alias)
            )

        if not value and self.omit_empty:
            return  # Do nothing

        if self._nested is not None:
            array_parent = _element_get_or_add_from_parent(parent, self._nested)
        else:
            # Embedded array has all items serialized directly on the parent.
            array_parent = parent

        self._serialize(array_parent, value, state)

    def _parse(
        self,
        item_iter,  # type: Iterable[ET.Element]
        state,  # type: _ProcessorState
    ):
        # type: (...) -> List
        """Parse the array data using the provided iterator of XML elements."""
        parsed_array = []

        for i, item in enumerate(item_iter):
            state.push_location(self._item_processor.element_path, i)
            parsed_array.append(self._item_processor.parse_at_element(item, state))
            state.pop_location()

        if not parsed_array and self.required:
            state.raise_error(
                MissingValue, 'Missing required array "{}"'.format(self.alias)
            )

        return parsed_array

    def _serialize(
        self,
        array_parent,  # type: ET.Element
        value,  # type: List
        state,  # type: _ProcessorState
    ):
        # type: (...) -> None
        """Serialize the array value and add it to the array parent element."""
        if not value:
            # Nothing to do. Avoid attempting to iterate over a possibly
            # None value.
            return

        for i, item_value in enumerate(value):
            state.push_location(self._item_processor.element_path, i)
            item_element = self._item_processor.serialize(item_value, state)
            array_parent.append(item_element)
            state.pop_location()


class _Dictionary(RootProcessor):
    """An XML processor for dictionary values."""

    def __init__(
        self,
        element_path,  # type: Text
        child_processors,  # type: List[Processor]
        required=True,  # type: bool
        alias=None,  # type: Optional[Text]
    ):
        # type: (...) -> None
        self._element_path = element_path
        self._child_processors = child_processors
        self._required = required
        if alias:
            self._alias = alias
        else:
            self._alias = element_path

    @property
    def alias(self):
        # type: (...) -> Text
        """Get processor's alias."""
        return self._alias

    @property
    def element_path(self):
        # type: (...) -> Text
        """Get path to processor's element."""
        return self._element_path

    @property
    def required(self):
        # type: (...) -> bool
        """Get whether the processor's value is required."""
        return self._required

    def parse_at_element(
        self,
        element,  # type: ET.Element
        state,  # type: _ProcessorState
    ):
        # type: (...) -> Any
        """Parse the provided element as a dictionary."""
        parsed_dict = {}

        for child in self._child_processors:
            state.push_location(child.element_path)
            parsed_dict[child.alias] = child.parse_from_parent(element, state)
            state.pop_location()

        return parsed_dict

    def parse_at_root(
        self,
        root,  # type: ET.Element
        state,  # type: _ProcessorState
    ):
        # type: (...) -> Any
        """Parse the root XML element as a dictionary."""
        parsed_dict = {}  # type: Dict

        dict_element = _element_find_from_root(root, self.element_path)
        if dict_element is not None:
            parsed_dict = self.parse_at_element(dict_element, state)
        elif self.required:
            raise MissingValue(
                'Missing required root aggregate "{}"'.format(self.element_path)
            )

        return parsed_dict

    def parse_from_parent(
        self,
        parent,  # type: ET.Element
        state,  # type: _ProcessorState
    ):
        # type: (...) -> Any
        """Parse the dictionary data from the provided parent XML element."""
        element = parent.find(self.element_path)

        if element is None and self.required:
            state.raise_error(
                MissingValue,
                'Missing required aggregate "{}"'.format(self.element_path),
            )
        elif element is not None:
            return self.parse_at_element(element, state)

        return {}

    def serialize(
        self,
        value,  # type: Any
        state,  # type: _ProcessorState
    ):
        # type: (...) -> ET.Element
        """Serialize the value to a new element and return the element."""
        if not value and self.required:
            state.raise_error(
                MissingValue,
                'Missing required aggregate "{}"'.format(self.element_path),
            )

        start_element, end_element = _element_path_create_new(self.element_path)
        self._serialize(end_element, value, state)
        return start_element

    def serialize_on_parent(
        self,
        parent,  # type: ET.Element
        value,  # type: Any
        state,  # type: _ProcessorState
    ):
        # type: (...) -> None
        """Serialize the value and add it to the parent."""
        if not value and self.required:
            state.raise_error(
                MissingValue,
                'Missing required aggregate "{}"'.format(self.element_path),
            )

        if not value:
            return  # Do Nothing

        element = _element_get_or_add_from_parent(parent, self.element_path)
        self._serialize(element, value, state)

    def _serialize(
        self,
        element,  # type: ET.Element
        value,  # type: Dict
        state,  # type: _ProcessorState
    ):
        # type: (...) -> None
        """Serialize the dictionary and append all serialized children to the element."""
        for child in self._child_processors:
            state.push_location(child.element_path)
            child_value = value.get(child.alias)
            child.serialize_on_parent(element, child_value, state)
            state.pop_location()


class _HookedAggregate(RootProcessor):
    """A processor which decorates a processor and applies hooks to all values processed."""

    def __init__(
        self,
        processor,  # type: RootProcessor
        hooks,  # type: Hooks
    ):
        # type: (...) -> None
        self._element_path = processor.element_path
        self._required = processor.required
        self._alias = processor.alias
        self._processor = processor
        self._hooks = hooks

    @property
    def alias(self):
        # type: (...) -> Text
        """Get processor's alias."""
        return self._alias

    @property
    def element_path(self):
        # type: (...) -> Text
        """Get path to processor's element."""
        return self._element_path

    @property
    def required(self):
        # type: (...) -> bool
        """Get whether the processor's value is required."""
        return self._required

    def parse_at_element(
        self,
        element,  # type: ET.Element
        state,  # type: _ProcessorState
    ):
        # type: (...) -> Any
        """Parse the given element."""
        xml_value = self._processor.parse_at_element(element, state)
        return _hooks_apply_after_parse(self._hooks, state, xml_value)

    def parse_at_root(
        self,
        root,  # type: ET.Element
        state,  # type: _ProcessorState
    ):
        # type: (...) -> Any
        """Parse the given element as the root of the document."""
        xml_value = self._processor.parse_at_root(root, state)
        return _hooks_apply_after_parse(self._hooks, state, xml_value)

    def parse_from_parent(
        self,
        parent,  # type: ET.Element
        state,  # type: _ProcessorState
    ):
        # type: (...) -> Any
        """Parse the element from the given parent element."""
        xml_value = self._processor.parse_from_parent(parent, state)
        return _hooks_apply_after_parse(self._hooks, state, xml_value)

    def serialize(
        self,
        value,  # type: Any
        state,  # type: _ProcessorState
    ):
        # type: (...) -> ET.Element
        """Serialize the value and returns it."""
        xml_value = _hooks_apply_before_serialize(self._hooks, state, value)
        return self._processor.serialize(xml_value, state)

    def serialize_on_parent(
        self,
        parent,  # type: ET.Element
        value,  # type: Any
        state,  # type: _ProcessorState
    ):
        # type: (...) -> None
        """Serialize the value directory on the parent."""
        xml_value = _hooks_apply_before_serialize(self._hooks, state, value)
        self._processor.serialize_on_parent(parent, xml_value, state)


class _PrimitiveValue(Processor):
    """An XML processor for processing primitive values."""

    def __init__(
        self,
        element_path,  # type: Text
        parser_func,  # type: Callable[[Optional[Text], _ProcessorState], Any]
        attribute=None,  # type: Optional[Text]
        required=True,  # type: bool
        alias=None,  # type: Optional[Text]
        default=None,  # type: Optional[Any]
        omit_empty=False,  # type: bool
        hooks=None,  # type: Optional[Hooks]
    ):
        # type: (...) -> None
        """
        Create a new processor for primitive values.

        :param element_path: Path to XML element containing the value.
        :param parser_func: Function to parse the raw XML value. Should take a string and return
            the value parsed from the raw string.
        :param required: Indicates whether the value is required.
        :param alias: Alternative name to give to the value. If not specified, element_path is used.
        :param default: Default value. Only valid if required is False.
        :param omit_empty: Omit the value when serializing if it is a falsey value. Only valid if
            required is False.
        :param hooks: A Hooks object.
        """
        self._element_path = element_path
        self._parser_func = parser_func
        self._attribute = attribute
        self._required = required
        self._default = default
        self._hooks = hooks

        if alias:
            self._alias = alias
        elif attribute:
            self._alias = attribute
        else:
            self._alias = element_path

        # If a value is required, then it will never be omitted when serialized. This
        # is to ensure that data that is serialized by a processor can also be parsed
        # by a processor. Omitting required values would lead to an error when attempting
        # to parse the XML data. For primitives, a value must be None to be considered
        # missing, but is considered empty if it is falsey.
        if required:
            self.omit_empty = False
            if omit_empty:
                warnings.warn(
                    "omit_empty ignored on primitive values when required is specified"
                )
        else:
            self.omit_empty = omit_empty

    @property
    def alias(self):
        # type: (...) -> Text
        """Get processor's alias."""
        return self._alias

    @property
    def element_path(self):
        # type: (...) -> Text
        """Get path to processor's element."""
        return self._element_path

    @property
    def required(self):
        # type: (...) -> bool
        """Get whether the processor's value is required."""
        return self._required

    def parse_at_element(
        self,
        element,  # type: ET.Element
        state,  # type: _ProcessorState
    ):
        # type: (...) -> Any
        """Parse the primitive value at the XML element."""
        if self._attribute:
            parsed_value = self._parse_attribute(element, self._attribute, state)
        else:
            parsed_value = self._parser_func(element.text, state)

        return _hooks_apply_after_parse(self._hooks, state, parsed_value)

    def parse_from_parent(
        self,
        parent,  # type: ET.Element
        state,  # type: _ProcessorState
    ):
        # type: (...) -> Any
        """Parse the primitive value under the parent XML element."""
        element = parent.find(self.element_path)

        if element is None and self.required:
            state.raise_error(
                MissingValue, 'Missing required element "{}"'.format(self.element_path)
            )
        elif element is not None:
            return self.parse_at_element(element, state)

        return _hooks_apply_after_parse(self._hooks, state, self._default)

    def serialize(
        self,
        value,  # type: Any
        state,  # type: _ProcessorState
    ):
        # type: (...) -> ET.Element
        """
        Serialize the value into a new element object and return the element.

        If the omit_empty option was specified and the value is falsey, then this will return None.
        """
        # For primitive values, this is only called when the value is part of an array,
        # in which case we do not need to check for missing or omitted values.
        start_element, end_element = _element_path_create_new(self.element_path)
        self._serialize(end_element, value, state)
        return start_element

    def serialize_on_parent(
        self,
        parent,  # type: ET.Element
        value,  # type: Any
        state,  # type: _ProcessorState
    ):
        # type: (...) -> None
        """Serialize the value and add it to the parent element."""
        # Note that falsey values are not treated as missing, but they may be omitted.
        if value is None and self.required:
            state.raise_error(MissingValue, self._missing_value_message(parent))

        if not value and self.omit_empty:
            return  # Do Nothing

        element = _element_get_or_add_from_parent(parent, self.element_path)
        self._serialize(element, value, state)

    def _missing_value_message(self, parent):
        # type: (ET.Element) -> Text
        """Return the message to report that the value needed for serialization is missing."""
        if self._attribute is None:
            message = 'Missing required value for element "{}"'.format(
                self.element_path
            )
        else:
            if self.element_path == ".":
                parent_name = parent.tag
            else:
                parent_name = self.element_path

            message = (
                'Missing required value for attribute "{}" on element "{}"'.format(
                    self._attribute, parent_name
                )
            )

        return message

    def _parse_attribute(
        self,
        element,  # type: ET.Element
        attribute,  # type: Text
        state,  # type: _ProcessorState
    ):
        # type: (...) -> Any
        """Parse the primitive value within the XML element's attribute."""
        parsed_value = self._default
        attribute_value = element.get(attribute, None)

        if attribute_value is not None:
            parsed_value = self._parser_func(attribute_value, state)
        elif self.required:
            state.raise_error(
                MissingValue,
                'Missing required attribute "{}" on element "{}"'.format(
                    self._attribute, element.tag
                ),
            )

        return parsed_value

    def _serialize(
        self,
        element,  # type: ET.Element
        value,  # type: Any
        state,  # type: _ProcessorState
    ):
        # type: (...) -> None
        """Serialize the value to the element."""
        xml_value = _hooks_apply_before_serialize(self._hooks, state, value)

        # A value is only considered missing, and hence eligible to be replaced by its
        # default only if it is None. Falsey values are not considered missing and are
        # not replaced by the default.
        if xml_value is None:
            if self._default is None:
                serialized_value = Text("")
            else:
                serialized_value = Text(self._default)
        else:
            serialized_value = Text(xml_value)

        if self._attribute:
            element.set(self._attribute, serialized_value)
        else:
            element.text = serialized_value


class _ProcessorState(object):
    """Keeps track of the state of the processor in order to provide useful error messages."""

    def __init__(self):
        self._locations = []  # type: List[ProcessorLocation]

    @property
    def locations(self):
        # type: () -> Iterable[ProcessorLocation]
        """Get iterator of locations representing current location of the processor."""
        return iter(self._locations)

    def pop_location(self):
        # type: () -> ProcessorLocation
        """Pop the most recently pushed location from the state's stack of locations."""
        return self._locations.pop()

    def push_location(
        self,
        element_path,  # type: Text
        array_index=None,  # type: Optional[int]
    ):
        # type: (...) -> None
        """Push an item onto the state's stack of locations."""
        location = ProcessorLocation(element_path=element_path, array_index=array_index)
        self._locations.append(location)

    def raise_error(
        self,
        exception_type,  # type: Type[Exception]
        message,  # type: Text
    ):
        # type: (...) -> NoReturn
        """Raise an exception with the current parser state information and error message."""
        error_message = "{} at {}".format(message, repr(self))
        raise exception_type(error_message)

    def __repr__(self):
        # Exclude the any locations specified with a dot which just means the "current location"
        # from the path string.
        location_strings = (
            self._location_to_string(location)
            for location in self._locations
            if location.element_path != "."
        )
        return "/".join(location_strings)

    @staticmethod
    def _location_to_string(location):
        # type: (ProcessorLocation) -> Text
        if location.array_index is not None:
            location_str = u"{}[{}]".format(location.element_path, location.array_index)
        else:
            location_str = location.element_path

        return location_str


def _element_append_path(
    start_element,  # type: ET.Element
    element_names,  # type: Iterable[Text]
):
    # type: (...) -> ET.Element
    """
    Append the list of element names as a path to the provided start element.

    :return: The final element along the path.
    """
    end_element = start_element
    for element_name in element_names:
        new_element = ET.Element(element_name)
        end_element.append(new_element)
        end_element = new_element

    return end_element


def _element_find_from_root(
    root,  # type: ET.Element
    element_path,  # type: Text
):
    # type: (...) -> Optional[ET.Element]
    """
    Find the element specified by the given path starting from the root element of the document.

    The first component of the element path is expected to be the name of the root element. Return
    None if the element is not found.
    """
    element = None

    element_names = element_path.split("/")
    if element_names[0] == root.tag:
        if len(element_names) > 1:
            element = root.find("/".join(element_names[1:]))
        else:
            element = root

    return element


def _element_get_or_add_from_parent(
    parent,  # type: ET.Element
    element_path,  # type: Text
):
    # type: (...) -> ET.Element
    """
    Ensure all elements specified in the given path relative to the provided parent element exist.

    Create new elements along the path only when needed, and return the final element specified
    by the path.
    """
    element_names = element_path.split("/")

    # Starting from the parent, walk the element path until we find the first element in the path
    # that does not exist. Create that element and all the elements following it in the path. If
    # all elements along the path exist, then we will simply walk the full path to the final
    # element we want to return.
    existing_element = None
    previous_element = parent
    for i, element_name in enumerate(element_names):
        existing_element = previous_element.find(element_name)
        if existing_element is None:
            existing_element = _element_append_path(previous_element, element_names[i:])
            break

        previous_element = existing_element

    assert existing_element is not None
    return existing_element


def _element_path_create_new(element_path):
    # type: (Text) -> Tuple[ET.Element, ET.Element]
    """
    Create an entirely new element path.

    Return a tuple where the first item is the first element in the path, and the second item is
    the final element in the path.
    """
    element_names = element_path.split("/")

    start_element = ET.Element(element_names[0])
    end_element = _element_append_path(start_element, element_names[1:])

    return start_element, end_element


def _hooks_apply_after_parse(
    hooks,  # type: Optional[Hooks]
    state,  # type: _ProcessorState
    value,  # type: Any
):
    # type: (...) -> Any
    """Apply the after parse hook."""
    if hooks and hooks.after_parse:
        return hooks.after_parse(ProcessorStateView(state), value)

    return value


def _hooks_apply_before_serialize(
    hooks,  # type: Optional[Hooks]
    state,  # type: _ProcessorState
    value,  # type: Any
):
    # type: (...) -> Any
    """Apply the before serialize hook."""
    if hooks and hooks.before_serialize:
        return hooks.before_serialize(ProcessorStateView(state), value)

    return value


def _is_valid_root_processor(processor):
    # type: (Processor) -> bool
    """Return True if the given XML processor can be used as a root processor."""
    return hasattr(processor, "parse_at_root")


def _named_tuple_converter(tuple_type):
    # type: (Type[Tuple]) -> _AggregateConverter
    """Return an _AggregateConverter for named tuples of the given type."""

    def _from_dict(dict_value):
        if dict_value:
            return tuple_type(**dict_value)

        # Cannot construct a namedtuple value from an empty dictionary
        return None

    def _to_dict(value):
        if value:
            return value._asdict()

        return {}

    converter = _AggregateConverter(from_dict=_from_dict, to_dict=_to_dict)
    return converter


def _number_parser(str_to_number_func):
    """Return a function to parse numbers."""

    def _parse_number_value(element_text, state):
        value = None

        try:
            value = str_to_number_func(element_text)
        except (ValueError, TypeError):
            state.raise_error(
                InvalidPrimitiveValue, 'Invalid numeric value "{}"'.format(element_text)
            )

        return value

    return _parse_number_value


def _parse_boolean(element_text, state):
    """Parse the raw XML string as a boolean value."""
    value = None

    lowered_text = element_text.lower()
    if lowered_text == "true":
        value = True
    elif lowered_text == "false":
        value = False
    else:
        state.raise_error(
            InvalidPrimitiveValue, 'Invalid boolean value "{}"'.format(element_text)
        )

    return value


def _processor_wrap_if_hooks(
    processor,  # type: RootProcessor
    hooks,  # type: Optional[Hooks]
):
    # type: (...) -> RootProcessor
    """Create a hooked processor if a valid hooks object is provided."""
    if hooks:
        return _HookedAggregate(processor, hooks)

    return processor


def _string_parser(strip_whitespace):
    """Return a parser function for parsing string values."""

    def _parse_string_value(element_text, _state):
        if element_text is None:
            value = ""
        elif strip_whitespace:
            value = element_text.strip()
        else:
            value = element_text

        return value

    return _parse_string_value


def _user_object_converter(cls):
    # type: (Type[Any]) -> _AggregateConverter
    """Return an _AggregateConverter for a user object of the given class."""

    def _from_dict(dict_value):
        try:
            object_value = cls(**dict_value)
        except TypeError:
            # Constructor does not support keyword arguments, try setting each
            # field individually.
            object_value = cls()
            for field_name, field_value in dict_value.items():
                setattr(object_value, field_name, field_value)

        return object_value

    def _to_dict(value):
        if value:
            return value.__dict__

        return {}

    return _AggregateConverter(from_dict=_from_dict, to_dict=_to_dict)


def _xml_namespace_strip(root):
    # type: (ET.Element) -> None
    """Strip the XML namespace prefix from all element tags under the given root Element."""
    if "}" not in root.tag:
        return  # Nothing to do, no namespace present

    for element in root.iter():
        if "}" in element.tag:
            element.tag = element.tag.split("}")[1]
        else:  # pragma: no cover
            # We should never get here. If there is a namespace, then the namespace should be
            # included in all elements.
            pass
