from imod.schemata import BaseSchema


def filter_schemata_dict(
    schemata_dict: dict[str, list[BaseSchema]], schema_types: tuple[type[BaseSchema]]
) -> dict[str, list[BaseSchema]]:
    """
    Filter schemata dict with a tuple of schema types. Keys which do not have
    provided types in their corresponding schema list are dropped. The schema
    list in the values is reduced to contain the schema_types only.

    Example
    -------
    >>> _write_schemata = {
        "stage": [
            AllValueSchema(">=", "bottom_elevation"),
            OtherCoordsSchema("idomain"),
            AllNoDataSchema(),  # Check for all nan, can occur while clipping
            AllInsideNoDataSchema(other="idomain", is_other_notnull=(">", 0)),
        ],
        "conductance": [IdentityNoDataSchema("stage"), AllValueSchema(">", 0.0)],
    }

    >>> print(filter_schemata_dict(write_schemata, (AllNoDataSchema())))

    Prints ``{'stage': [<imod.schemata.AllNoDataSchema at 0x1b152b12aa0>]}``
    """

    dict = {}
    for key, schema_ls in schemata_dict.items():
        schema_match = [
            schema for schema in schema_ls if isinstance(schema, schema_types)
        ]
        if schema_match:
            dict[key] = schema_match
    return dict
