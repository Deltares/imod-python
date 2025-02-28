from pytest_cases import parametrize_with_cases

from imod.common.utilities.schemata import filter_schemata_dict
from imod.mf6.riv import River
from imod.schemata import AllNoDataSchema, IdentityNoDataSchema, IndexesSchema


class CasesFilteredSchemata:
    def case_empty(self):
        schemata = {}
        arg = (AllNoDataSchema,)
        expected = {}
        return schemata, arg, expected

    def case_river_allnodata(self):
        schemata = River._write_schemata
        arg = (AllNoDataSchema,)
        expected = {"stage": [AllNoDataSchema()]}
        return schemata, arg, expected

    def case_river_allnodata_identitynodataschema(self):
        schemata = River._write_schemata
        arg = (AllNoDataSchema, IdentityNoDataSchema)
        expected = {
            "stage": [AllNoDataSchema()],
            "conductance": [IdentityNoDataSchema("stage")],
            "bottom_elevation": [IdentityNoDataSchema("stage")],
            "concentration": [IdentityNoDataSchema("stage")],
        }
        return schemata, arg, expected

    def case_river_not_found(self):
        # IndexesSchema part of _init_schemata, so should not be in
        # _write_schemata.
        schemata = River._write_schemata
        arg = (IndexesSchema,)
        expected = {}
        return schemata, arg, expected


@parametrize_with_cases(("schemata", "arg", "expected"), cases=CasesFilteredSchemata)
def test_filter_schemata_dict(schemata, arg, expected):
    # Act
    filtered_dict = filter_schemata_dict(schemata, arg)

    # Assert
    # Test if same variable names present in dicts
    assert filtered_dict.keys() == expected.keys()

    # Test if scheme types in list equal.
    for key in filtered_dict.keys():
        schema_types = [type(s) for s in filtered_dict[key]]
        expected_types = [type(s) for s in expected[key]]
        assert schema_types == expected_types
