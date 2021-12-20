import os
import shutil


def remove_dir_content(path: str) -> None:
    for root, dirs, files in os.walk(path):
        for f in files:
            if f != ".gitkeep":
                os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))


remove_dir_content("examples/imod-wq")
remove_dir_content("examples/imodflow")
remove_dir_content("examples/mf6")
remove_dir_content("examples/prepare")
remove_dir_content("examples/visualize")

remove_dir_content("user-guide")

remove_dir_content("sample_data")

remove_dir_content("api/generated")

remove_dir_content("_build")
