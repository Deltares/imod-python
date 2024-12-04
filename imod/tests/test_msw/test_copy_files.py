from pytest_cases import parametrize_with_cases

from imod.msw.copy_files import CopyFiles


def write_test_files(directory, filenames):
    paths = [directory / filename for filename in filenames]
    for p in paths:
        with open(p, mode="w") as f:
            f.write("test")
    return paths


def case_simple_files(tmp_path_factory):
    directory = tmp_path_factory.mktemp("simple_files")
    filenames = [
        "a.inp",
        "b.inp",
        "c.inp",
    ]
    return write_test_files(directory, filenames)


def case_imod5_extra_files(tmp_path_factory):
    directory = tmp_path_factory.mktemp("imod5_extra_files")
    filenames = [
        "a.inp",
        "b.inp",
        "c.inp",
        "mete_grid.inp",
        "para_sim.inp",
        "svat2precgrid.inp",
        "svat2etrefgrid.inp",
    ]
    return write_test_files(directory, filenames)


@parametrize_with_cases("src_files", cases=".")
def test_copyfile_init(src_files):
    # Act
    copyfiles = CopyFiles(src_files)
    # Arrange
    assert "paths" in copyfiles.dataset.keys()
    assert len(copyfiles.dataset["paths"]) == len(src_files)


@parametrize_with_cases("src_files", cases=".")
def test_copyfile_write(src_files, tmp_path):
    # Arrange
    expected_filenames = {f.name for f in src_files}
    # Act
    copyfiles = CopyFiles(src_files)
    copyfiles.write(tmp_path)
    # Assert
    actual_filepaths = tmp_path.glob("*.inp")
    actual_filenames = {f.name for f in actual_filepaths}
    diff = expected_filenames ^ actual_filenames
    assert len(diff) == 0


@parametrize_with_cases("src_files", cases=".")
def test_from_imod5_data(src_files):
    # Arrange
    imod5_ls = [[p] for p in src_files]
    imod5_data = {"extra": {"paths": imod5_ls}}
    # Act
    copyfiles = CopyFiles.from_imod5_data(imod5_data)
    # Assert
    len(copyfiles.dataset["paths"]) == 3
