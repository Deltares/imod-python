"""Contains unit tests for handling XPath syntax in processors"""
from imod import declxml as xml

from .helpers import strip_xml


class _ProcessorTestCase(object):
    """Base class for processor test cases"""

    xml_string = None
    value = None
    processor = None

    def test_parse(self):
        """Tests parsing"""
        actual_value = xml.parse_from_string(
            self.__class__.processor, self.__class__.xml_string
        )

        assert self.__class__.value == actual_value

    def test_serialize(self):
        """Tests serializing"""
        actual_xml_string = xml.serialize_to_string(
            self.__class__.processor, self.__class__.value
        )

        assert self.__class__.xml_string == actual_xml_string


class TestDot(_ProcessorTestCase):
    """Parse/serialize with dot specifier"""

    xml_string = strip_xml(
        """
    <files>
        <file name="a.txt" size="236" />
        <file name="b.txt" size="7654" />
    </files>
    """
    )

    value = [
        {
            "name": "a.txt",
            "size": 236,
        },
        {
            "name": "b.txt",
            "size": 7654,
        },
    ]

    processor = xml.array(
        xml.dictionary(
            "file",
            [xml.string(".", attribute="name"), xml.integer(".", attribute="size")],
        ),
        nested="files",
    )


class TestMixed(_ProcessorTestCase):
    """Process with a mix of specifiers"""

    xml_string = strip_xml(
        """
    <location name="Kansas City">
        <coordinates>
            <lat>39.0997</lat>
            <lon>94.5786</lon>
        </coordinates>
    </location>
    """
    )

    value = {
        "name": "Kansas City",
        "lat": 39.0997,
        "lon": 94.5786,
    }

    processor = xml.dictionary(
        "location",
        [
            xml.string(".", attribute="name"),
            xml.floating_point("coordinates/lat", alias="lat"),
            xml.floating_point("coordinates/lon", alias="lon"),
        ],
    )


class TestNestedSlash(_ProcessorTestCase):
    """Process with nested slash specifiers"""

    xml_string = strip_xml(
        """
    <root>
        <place>
            <city>
                <name>Kansas City</name>
                <location>
                    <coordinates>
                        <lat>39.0997</lat>
                        <lon>94.5786</lon>
                    </coordinates>
                </location>
            </city>
        </place>
    </root>
    """
    )

    value = {
        "name": "Kansas City",
        "lat": 39.0997,
        "lon": 94.5786,
    }

    processor = xml.dictionary(
        "root",
        [
            xml.string("place/city/name", alias="name"),
            xml.floating_point("place/city/location/coordinates/lat", alias="lat"),
            xml.floating_point("place/city/location/coordinates/lon", alias="lon"),
        ],
    )


class TestSlash(_ProcessorTestCase):
    """Process with slash specifier"""

    xml_string = strip_xml(
        """
    <city>
        <name>Kansas City</name>
        <coordinates>
            <lat>39.0997</lat>
            <lon>94.5786</lon>
        </coordinates>
    </city>
    """
    )

    value = {
        "name": "Kansas City",
        "lat": 39.0997,
        "lon": 94.5786,
    }

    processor = xml.dictionary(
        "city",
        [
            xml.string("name"),
            xml.floating_point("coordinates/lat", alias="lat"),
            xml.floating_point("coordinates/lon", alias="lon"),
        ],
    )


class TestSlashArrayOfArrays(_ProcessorTestCase):
    """Process with slashes to arrays of arrays"""

    xml_string = strip_xml(
        """
    <root>
        <data>
            <experiments>
                <experiment>
                    <results>
                        <datapoints>
                            <datapoint>
                                <value>3</value>
                            </datapoint>
                            <datapoint>
                                <value>1</value>
                            </datapoint>
                            <datapoint>
                                <value>5</value>
                            </datapoint>
                        </datapoints>
                    </results>
                </experiment>
                <experiment>
                    <results>
                        <datapoints>
                            <datapoint>
                                <value>49</value>
                            </datapoint>
                            <datapoint>
                                <value>42</value>
                            </datapoint>
                        </datapoints>
                    </results>
                </experiment>
            </experiments>
        </data>
    </root>
    """
    )

    value = [[3, 1, 5], [49, 42]]

    processor = xml.array(
        xml.array(
            xml.integer("datapoint/value"), nested="experiment/results/datapoints"
        ),
        nested="root/data/experiments",
    )


class TestSlashDifferentLevels(_ProcessorTestCase):
    """Process a slash to different levels of the same path"""

    xml_string = strip_xml(
        """
    <root>
        <level1>
            <level2>
                <valueB>86</valueB>
                <level3>
                    <valueC>13</valueC>
                </level3>
            </level2>
            <valueA>27</valueA>
        </level1>
    </root>
    """
    )

    value = {
        "valueB": 86,
        "valueA": 27,
        "valueC": 13,
    }

    processor = xml.dictionary(
        "root",
        [
            xml.integer("level1/level2/valueB", alias="valueB"),
            xml.integer("level1/valueA", alias="valueA"),
            xml.integer("level1/level2/level3/valueC", alias="valueC"),
        ],
    )


class TestSlashFromRootArray(_ProcessorTestCase):
    """Process with a slash starting from the root element"""

    xml_string = strip_xml(
        """
    <locations>
        <cities>
            <city name="Kansas City" state="MO" />
            <city name="Lincoln" state="NE" />
            <city name="Salt Lake City" state="UT" />
        </cities>
    </locations>
    """
    )

    value = [
        {"name": "Kansas City", "state": "MO"},
        {"name": "Lincoln", "state": "NE"},
        {"name": "Salt Lake City", "state": "UT"},
    ]

    processor = xml.array(
        xml.dictionary(
            "city",
            [
                xml.string(".", attribute="name"),
                xml.string(".", attribute="state"),
            ],
        ),
        nested="locations/cities",
        alias="cities",
    )


class TestSlashFromRootDict(_ProcessorTestCase):
    """Process with a slash starting from the root element"""

    xml_string = strip_xml(
        """
    <location>
        <city>
            <name>Kansas City</name>
            <coordinates>
                <lat>39.0997</lat>
                <lon>94.5786</lon>
            </coordinates>
        </city>
    </location>
    """
    )

    value = {
        "name": "Kansas City",
        "lat": 39.0997,
        "lon": 94.5786,
    }

    processor = xml.dictionary(
        "location/city",
        [
            xml.string("name"),
            xml.floating_point("coordinates/lat", alias="lat"),
            xml.floating_point("coordinates/lon", alias="lon"),
        ],
    )


class TestSlashPrimitiveInArray(_ProcessorTestCase):
    """Process with a slash to a primitive value in an array"""

    xml_string = strip_xml(
        """
    <dataset>
        <source>Census</source>
        <person>
            <demographics>
                <age>23</age>
            </demographics>
        </person>
        <person>
            <demographics>
                <age>27</age>
            </demographics>
        </person>
        <person>
            <demographics>
                <age>31</age>
            </demographics>
        </person>
    </dataset>
    """
    )

    value = {"source": "Census", "ages": [23, 27, 31]}

    processor = xml.dictionary(
        "dataset",
        [
            xml.string("source"),
            xml.array(
                xml.integer("person/demographics/age", alias="age"), alias="ages"
            ),
        ],
    )


class TestSlashPrimitiveInNestedArray(_ProcessorTestCase):
    """Process with a slash to a primitive value in an array"""

    xml_string = strip_xml(
        """
    <dataset>
        <source>Census</source>
        <people>
            <person>
                <demographics>
                    <age>23</age>
                </demographics>
            </person>
            <person>
                <demographics>
                    <age>27</age>
                </demographics>
            </person>
            <person>
                <demographics>
                    <age>31</age>
                </demographics>
            </person>
        </people>
    </dataset>
    """
    )

    value = {"source": "Census", "ages": [23, 27, 31]}

    processor = xml.dictionary(
        "dataset",
        [
            xml.string("source"),
            xml.array(
                xml.integer("person/demographics/age", alias="age"),
                nested="people",
                alias="ages",
            ),
        ],
    )


class TestSlashPrimitiveInNestedRootArray(_ProcessorTestCase):
    """Process with a slash to a primitive value in an array"""

    xml_string = strip_xml(
        """
    <people>
        <person>
            <demographics>
                <age>23</age>
            </demographics>
        </person>
        <person>
            <demographics>
                <age>27</age>
            </demographics>
        </person>
        <person>
            <demographics>
                <age>31</age>
            </demographics>
        </person>
    </people>
    """
    )

    value = [23, 27, 31]

    processor = xml.array(
        xml.integer("person/demographics/age", alias="age"), nested="people"
    )
