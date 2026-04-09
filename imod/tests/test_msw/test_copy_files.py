import pytest
from pytest_cases import parametrize_with_cases

from imod.msw.copy_files import FileCopier


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
    copyfiles = FileCopier(src_files)
    # Arrange
    assert "paths" in copyfiles.dataset.keys()
    assert len(copyfiles.dataset["paths"]) == len(src_files)


@parametrize_with_cases("src_files", cases=".")
def test_clip_box(src_files):
    # Act
    copyfiles = FileCopier(src_files)
    clipped = copyfiles.clip_box(x_min=1.0, x_max=2.5, y_min=1.0, y_max=2.5)
    # Arrange
    assert copyfiles.dataset.identical(clipped.dataset)


@parametrize_with_cases("src_files", cases=".")
def test_copyfile_write(src_files, tmp_path):
    # Arrange
    expected_filenames = {f.name for f in src_files}
    # Act
    copyfiles = FileCopier(src_files)
    copyfiles.write(tmp_path)
    # Assert
    actual_filepaths = tmp_path.glob("*.inp")
    actual_filenames = {f.name for f in actual_filepaths}
    diff = expected_filenames ^ actual_filenames
    assert len(diff) == 0


@pytest.mark.unittest_jit
@parametrize_with_cases("src_files", cases=".")
def test_from_imod5_data(src_files):
    # Arrange
    imod5_ls = [[p] for p in src_files]
    imod5_data = {"extra": {"paths": imod5_ls}}
    # Act
    copyfiles = FileCopier.from_imod5_data(imod5_data)
    # Assert
    assert len(copyfiles.dataset["paths"]) == 3
