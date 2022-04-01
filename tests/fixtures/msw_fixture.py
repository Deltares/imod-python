import pytest


# Seperate function from the one in fixed_format module. This one fails if not
# able to read value for tests.
@pytest.fixture(scope="session")
def fixed_format_parser():
    def function(file, metadata_dict):
        results = {}
        for key in metadata_dict:
            results[key] = []

        with open(file) as f:
            lines = f.readlines()
            for line in lines:
                if line == "\n":
                    continue
                for varname, metadata in metadata_dict.items():
                    # Take first part of line
                    value = line[: metadata.column_width]
                    # Convert to correct type
                    converted_value = metadata.dtype(value)
                    # Add to results
                    results[varname].append(converted_value)
                    # Truncate line
                    line = line[metadata.column_width :]
        return results

    return function
