import pytest


@pytest.fixture(scope="session")
def fixed_format_parser():
    def function(file, metadata_dict):
        results = {}
        with open(file) as f:
            for varname, metadata in metadata_dict.items():
                results[varname] = f.read(metadata.column_width)
        return results

    return function
