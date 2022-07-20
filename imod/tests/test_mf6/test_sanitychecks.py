import filecmp
import os.path
import shutil

import instance_creation as ic
import numpy as np

import imod
from imod.mf6.pkgbase import Package


def are_dir_trees_equal(dir1, dir2):
    """
    Compare two directories recursively. Files in each directory are
    assumed to be equal if their names and contents are equal.

    @param dir1: First directory path
    @param dir2: Second directory path

    @return: True if the directory trees are the same and
        there were no errors while accessing the directories or files,
        False otherwise.
    """

    dirs_cmp = filecmp.dircmp(dir1, dir2)
    if (
        len(dirs_cmp.left_only) > 0
        or len(dirs_cmp.right_only) > 0
        or len(dirs_cmp.funny_files) > 0
    ):
        return False
    (_, mismatch, errors) = filecmp.cmpfiles(
        dir1, dir2, dirs_cmp.common_files, shallow=False
    )
    if len(mismatch) > 0 or len(errors) > 0:
        return False
    for common_dir in dirs_cmp.common_dirs:
        new_dir1 = os.path.join(dir1, common_dir)
        new_dir2 = os.path.join(dir2, common_dir)
        if not are_dir_trees_equal(new_dir1, new_dir2):
            return False
    return True


def check_package_does_not_have_nonstandard_attributes(instance):
    props_of_baseclass = [a for a in dir(Package) if not a.startswith("__")]
    allowed_props = [
        "_pkg_id",
        "_template",
        "_metadata_dict",
        "_grid_data",
        "_keyword_map",
        "dataset",
    ]
    props = [a for a in dir(instance) if not a.startswith("__")]
    for p in props:
        if p not in props_of_baseclass and p not in allowed_props:
            if not callable(getattr(instance, p)):
                print(
                    f"class {instance.__name__} has a nonstandard non-callable attribute called {p}"
                )
                assert False


def test_packages_have_expected_attributes():
    for cls in Package.__subclasses__():
        # skip other base classes such as BoundaryCondition
        if cls.__subclasses__() == []:
            # checking for _pkg_id
            props = [a for a in dir(cls) if not a.startswith("__")]
            assert "_pkg_id" in props
            assert cls._pkg_id != ""


def test_packages_do_not_have_nonstandard_attributes():
    for cls in Package.__subclasses__():
        # skip other base classes such as BoundaryCondition
        if cls.__subclasses__() == []:
            instances_of_class = ic.get_instances(cls)
            for instance in instances_of_class:
                check_package_does_not_have_nonstandard_attributes(instance)


def test_packages_render_same_text_twice():
    globaltimes = [np.datetime64("2000-01-01")]
    with imod.util.temporary_directory() as modeldir:
        modeldir.mkdir()
        for cls in Package.__subclasses__():
            # skip other base classes such as BoundaryCondition
            if cls.__subclasses__() == []:
                instances_of_class = ic.get_instances(cls)
                for instance in instances_of_class:
                    text1 = instance.render(modeldir, "test", globaltimes, False)
                    text2 = instance.render(modeldir, "test", globaltimes, False)
                    assert text1 == text2


def test_package_load_save():
    globaltimes = [np.datetime64("2000-01-01")]
    for cls in Package.__subclasses__():
        # skip other base classes such as BoundaryCondition
        if cls.__subclasses__() == []:
            instances_of_class = ic.get_instances(cls)
            for instance in instances_of_class:
                with imod.util.temporary_directory() as modeldir:
                    modeldir.mkdir()

                    # save the original object as a netcdf file
                    netcdf_path = modeldir / f"{cls.__name__}.nc"
                    instance.dataset.to_netcdf(netcdf_path)

                    # also write the original object including period data
                    subdir_1 = modeldir / "before_loading"
                    subdir_1.mkdir()
                    subdir1_1 = subdir_1 / "write"
                    subdir1_1.mkdir()
                    if cls.__name__ != imod.mf6.TimeDiscretization.__name__:
                        instance.write(subdir1_1, "test", globaltimes, False)
                    else:
                        # time discretization.write has a non-conforming interface
                        instance.write(subdir1_1, "test")
                    # load new instance from the netcdf file we just saved
                    newinstance = getattr(cls, "from_file")(netcdf_path)

                    # write the new object including period data
                    subdir_2 = modeldir / "after_loading"
                    subdir_2.mkdir()
                    subdir2_1 = subdir_2 / "write"
                    subdir2_1.mkdir()
                    if cls.__name__ != imod.mf6.TimeDiscretization.__name__:
                        newinstance.write(subdir2_1, "test", globaltimes, False)
                    else:
                        # time discretization.write has a non-conforming interface
                        newinstance.write(subdir2_1, "test")

                    # compare the 2 directory trees
                    equal = are_dir_trees_equal(subdir1_1, subdir2_1)

                    # cleanup
                    shutil.rmtree(subdir_1)
                    shutil.rmtree(subdir_2)
                    newinstance.dataset.close()
                    instance.dataset.close()
                    shutil.rmtree(modeldir)

                    assert equal
