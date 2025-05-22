from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from typing import Optional

from imod.common.statusinfo import NestedStatusInfo, StatusInfo, StatusInfoBase
from imod.schemata import BaseSchema, SchemataDict, ValidationError


def filter_schemata_dict(
    schemata_dict: SchemataDict,
    schema_types: tuple[type[BaseSchema], ...],
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

    d = {}
    for key, schema_ls in schemata_dict.items():
        schema_match = [
            schema for schema in schema_ls if isinstance(schema, schema_types)
        ]
        if schema_match:
            d[key] = schema_match
    return d


def concatenate_schemata_dicts(
    schemata1: SchemataDict, schemata2: SchemataDict
) -> SchemataDict:
    """
    Concatenate two schemata dictionaries. If a key is present in both
    dictionaries, the values are concatenated into a list. If a key is only
    present in one dictionary, it is added to the new dictionary as is.
    """
    schemata = deepcopy(schemata1)
    for key, value in schemata2.items():
        if key not in schemata.keys():
            schemata[key] = value
        else:
            # Force to list to be able to concatenate
            schemata[key] = list(schemata[key]) + list(value)
    return schemata


def validate_schemata_dict(
    schemata: SchemataDict, data: Mapping, **kwargs
) -> dict[str, list[ValidationError]]:
    """
    Validate a data mapping against a schemata dictionary. Returns a dictionary
    of errors for each variable in the schemata dictionary. The errors are
    stored in a list for each variable.
    """
    errors = defaultdict(list)
    for variable, var_schemata in schemata.items():
        for schema in var_schemata:
            if variable in data.keys():
                try:
                    schema.validate(data[variable], **kwargs)
                except ValidationError as e:
                    errors[variable].append(e)
    return errors


def validation_pkg_error_message(pkg_errors):
    messages = []
    for var, var_errors in pkg_errors.items():
        messages.append(f"- {var}")
        messages.extend(f"    - {error}" for error in var_errors)
    return "\n" + "\n".join(messages)


def pkg_errors_to_status_info(
    pkg_name: str,
    pkg_errors: dict[str, list[ValidationError]],
    footer_text: Optional[str],
) -> StatusInfoBase:
    pkg_status_info = NestedStatusInfo(f"{pkg_name} package")
    for var_name, var_errors in pkg_errors.items():
        var_status_info = StatusInfo(var_name)
        for var_error in var_errors:
            var_status_info.add_error(str(var_error))
        pkg_status_info.add(var_status_info)
    pkg_status_info.set_footer_text(footer_text)
    return pkg_status_info
