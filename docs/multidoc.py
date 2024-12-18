import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import git
import jinja2
import packaging.version


class MultiDoc:
    def __init__(self):
        # Define useful paths
        self.current_dir = Path(os.path.dirname(os.path.abspath(__file__)))

        self.root_dir = self.current_dir.parent
        self.patch_file = self.current_dir / "version-switcher-patch.diff"

        # Attach to existing repo
        self.repo = git.Repo.init(self.root_dir)

        # Parse arguments
        root_parser = self.setup_arguments()
        args = root_parser.parse_args()
        self.config = vars(args)

        # Set additional paths
        self.work_dir = (
            Path(self.config["build_folder"])
            if os.path.isabs(self.config["build_folder"])
            else self.current_dir / self.config["build_folder"]
        )
        self.doc_dir = (
            Path(self.config["doc_folder"])
            if os.path.isabs(self.config["doc_folder"])
            else self.current_dir / self.config["doc_folder"]
        )

        # Setup url
        self.baseurl = (
            "https://deltares.github.io/imod-python"
            if not self.config["local_build"]
            else self.doc_dir.as_uri()
        )
        self.json_location = "_static/switcher.json"

        # Execute command
        if self.config["command"] is None:
            root_parser.print_help()
            sys.exit(0)

        getattr(self, self.config["command"].replace("-", "_"))()

    def setup_arguments(self):
        root_parser = argparse.ArgumentParser(
            description="A simple multi version sphinx doc builder.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            exit_on_error=True,
        )

        # Common arguments shared by all commands
        root_parser.add_argument(
            "--doc-folder",
            action="store",
            default=self.current_dir / "_build" / "html",
            help="Folder that contains the existing documentation and where new documentation will be added",
        )
        root_parser.add_argument(
            "--build-folder",
            action="store",
            default=self.current_dir / "workdir",
            help="Folder in which the version is checked out and build.",
        )
        root_parser.add_argument(
            "--local-build",
            action="store_true",
            help="Changes the hardcoded url of the switcher.json to a local file url. This makes it possible to use the version switcher locally.",
        )
        subparsers = root_parser.add_subparsers(dest="command")

        # Parser for "add-version"
        add_version_parser = subparsers.add_parser(
            "add-version",
            help="Build and add a version to the documentation.",
            exit_on_error=True,
        )
        add_version_parser.add_argument("version", help="Version to add.")

        # Parser for "update-version"
        update_version_parser = subparsers.add_parser(
            "update-version",
            help="Build and override a version of the documentation.",
            exit_on_error=True,
        )
        update_version_parser.add_argument("version", help="Version to update.")

        # Parser for "remove-version"
        remove_version_parser = subparsers.add_parser(
            "remove-version",
            help="Remove a version from the documentation.",
            exit_on_error=True,
        )
        remove_version_parser.add_argument("version", help="Version to remove.")

        # Parser for "list-versions"
        _ = subparsers.add_parser(
            "list-versions", help="List present versions in the documentation"
        )
        _ = subparsers.add_parser(
            "create-switcher", help="List present versions in the documentation"
        )

        return root_parser

    def add_version(self):
        version = self.config["version"]
        print(f"add-version: {version}")

        self._build_version(version)
        self._build_switcher()

    def update_version(self):
        version = self.config["version"]
        print(f"update-version: {version}")

        self._build_version(version)

    def remove_version(self):
        version = self.config["version"]
        print(f"remove-version: {version}")

        shutil.rmtree(self.doc_dir / version)
        self._build_switcher()

    def _build_version(self, version):
        with GitWorktreeManager(self.repo, self.work_dir, version):
            # Define the branch documentation source folder and build folder
            local_source_dir = self.work_dir / "docs"
            local_build_dir = self.work_dir / "builddir"

            # Apply patch to older version. Once it is known in which version(branch/tag) this file will be added
            # we can add a check to apply this patch only to older versions
            print("Applying patch")
            _ = subprocess.run(
                ["git", "apply", self.patch_file],
                cwd=self.work_dir,
                check=True,
            )

            # Clean existing Pixi enviroment settings
            print(
                "Clearing pixi enviroment settings. This is needed for a clean build."
            )
            env = os.environ
            path_items = os.environ["PATH"].split(os.pathsep)
            filtered_path = [
                path
                for path in path_items
                if not os.path.abspath(path).startswith(str(self.root_dir))
            ]
            env["Path"] = os.pathsep.join(filtered_path)

            pixi_env_vars = [
                "PIXI_PROJECT_ROOT",
                "PIXI_PROJECT_NAME",
                "PIXI_PROJECT_MANIFEST",
                "PIXI_PROJECT_VERSION",
                "PIXI_PROMPT",
                "PIXI_ENVIRONMENT_NAME",
                "PIXI_ENVIRONMENT_PLATFORMS",
                "CONDA_PREFIX",
                "CONDA_DEFAULT_ENV",
                "INIT_CWD",
            ]

            for pixi_var in pixi_env_vars:
                if pixi_var in env:
                    del env[pixi_var]

            # Add json url to the environment. This will be used in the conf.py file
            env["JSON_URL"] = f"{self.baseurl}/{self.json_location}"

            # Build the documentation of the branch
            print("Start sphinx-build.")
            _ = subprocess.run(
                ["pixi", "run", "--frozen", "install"],
                cwd=self.work_dir,
                env=env,
                check=True,
            )
            _ = subprocess.run(
                [
                    "pixi",
                    "run",
                    "--frozen",
                    "sphinx-build",
                    "-M",
                    "html",
                    local_source_dir,
                    local_build_dir,
                ],
                cwd=self.work_dir,
                env=env,
                check=True,
            )

            # Collect the branch documentation and add it to the
            print("Move documentation to correct location.")
            branch_html_dir = local_build_dir / "html"
            shutil.rmtree(self.doc_dir / version, ignore_errors=True)
            shutil.copytree(branch_html_dir, self.doc_dir / version)

    def list_versions(self):
        print(self._get_existing_versions())

    def create_switcher(self):
        self._build_switcher()

    def _build_switcher(self):
        switcher = SwitcherBuilder(self._get_existing_versions(), self.baseurl)
        version_info = switcher.build()

        template = jinja2.Template("""{{ version_info | tojson(indent=4) }}""")
        rendered_document = template.render(version_info=version_info)

        json_path = self.doc_dir / self.json_location
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as fh:
            fh.write(rendered_document)

    def _get_existing_versions(self):
        ignore = ["_static"]
        versions = [
            name
            for name in os.listdir(self.doc_dir)
            if os.path.isdir(self.doc_dir / name) and name not in ignore
        ]
        return versions


class GitWorktreeManager:
    def __init__(self, repo, work_dir, branch_or_tag):
        self.repo = repo
        self.work_dir = work_dir
        self.branch_or_tag = branch_or_tag

    def __enter__(self):
        self.repo.git.execute(
            ["git", "worktree", "add", f"{self.work_dir}", self.branch_or_tag]
        )

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.repo.git.execute(
                ["git", "worktree", "remove", f"{self.work_dir}", "--force"]
            )
        except Exception:
            print("Warning: could not remove the worktree")


class SwitcherBuilder:
    def __init__(self, versions, baseurl):
        self._versions = versions
        self._versions.sort(reverse=True)
        self.baseurl = baseurl

    @property
    def latest_stable_version(self):
        dev_branch = ["master"]
        filtered_versions = [
            version
            for version in self._versions
            if version not in dev_branch
            and not packaging.version.Version(version).is_prerelease
        ]
        latest_version = (
            max(filtered_versions, key=packaging.version.parse)
            if filtered_versions
            else None
        )

        return latest_version

    @property
    def versions(self):
        return self._versions

    def build(self):
        version_info = []
        for version in self.versions:
            version_info += [
                {
                    "name": self._version_to_name(version),
                    "version": version,
                    "url": f"{self.baseurl}/{version}",
                    "preferred": version == self.latest_stable_version,
                }
            ]

        return version_info

    def _version_to_name(self, version):
        name_postfix = ""
        if version == "master":
            name_postfix = "(latest)"
        if version == self.latest_stable_version:
            name_postfix = "(stable)"

        name = " ".join([version, name_postfix])
        return name


if __name__ == "__main__":
    MultiDoc()
