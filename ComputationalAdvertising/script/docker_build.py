#!/usr/bin/env python3
"""
Build Docker images using versions from deps.yaml

Usage:
    python script/docker_build.py download   # Build download stage
    python script/docker_build.py offline    # Build offline stage
    python script/docker_build.py all        # Build both stages
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def load_deps(deps_file: Path) -> dict:
    """Load dependencies from deps.yaml and return version mapping."""
    with open(deps_file) as f:
        data = yaml.safe_load(f)

    versions = {}
    for dep in data.get("dependencies", []):
        name = dep["name"]
        version = dep["version"]
        github = dep.get("github", "")

        # Normalize name to uppercase with underscores for Docker ARG
        arg_name = name.upper().replace("-", "_") + "_VERSION"
        versions[arg_name] = version

        # Store github org/repo for download URLs
        if github:
            repo_arg = name.upper().replace("-", "_") + "_REPO"
            versions[repo_arg] = github

    return versions


def build_docker_image(
    dockerfile: str,
    image_tag: str,
    versions: dict,
    context_dir: Path,
    extra_args: list = None,
) -> int:
    """Build Docker image with version arguments."""
    cmd = ["docker", "build", "-f", dockerfile, "-t", image_tag]

    # Add all version arguments
    for arg_name, value in versions.items():
        cmd.extend(["--build-arg", f"{arg_name}={value}"])

    # Add extra arguments
    if extra_args:
        cmd.extend(extra_args)

    # Add context directory
    cmd.append(str(context_dir))

    print(f"Building {image_tag}...")
    print(f"Command: {' '.join(cmd[:10])}... ({len(versions)} version args)")

    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(description="Build Docker images with deps.yaml versions")
    parser.add_argument(
        "stage",
        choices=["download", "offline", "combined", "all"],
        help="Which stage to build (combined is recommended for CI)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Build without cache",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Number of parallel build jobs (for offline stage)",
    )
    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    deps_file = project_root / "deps.yaml"
    docker_dir = project_root / "docker"

    if not deps_file.exists():
        print(f"Error: {deps_file} not found")
        return 1

    # Load versions
    versions = load_deps(deps_file)
    print(f"Loaded {len(versions)} version variables from deps.yaml")

    # Add build jobs
    versions["BUILD_JOBS"] = str(args.jobs)

    extra_args = []
    if args.no_cache:
        extra_args.append("--no-cache")

    # Build requested stages
    if args.stage in ("download", "all"):
        ret = build_docker_image(
            dockerfile="docker/Dockerfile.download",
            image_tag="computationaladvertising/download-deps:latest",
            versions=versions,
            context_dir=project_root,
            extra_args=extra_args,
        )
        if ret != 0:
            print("Download stage failed!")
            return ret
        print("Download stage completed successfully!")

    if args.stage in ("offline", "all"):
        ret = build_docker_image(
            dockerfile="docker/Dockerfile.offline",
            image_tag="computationaladvertising/build-base:latest",
            versions=versions,
            context_dir=project_root,
            extra_args=extra_args,
        )
        if ret != 0:
            print("Offline stage failed!")
            return ret
        print("Offline stage completed successfully!")

    if args.stage == "combined":
        ret = build_docker_image(
            dockerfile="docker/Dockerfile.base",
            image_tag="computationaladvertising/build-base:latest",
            versions=versions,
            context_dir=project_root,
            extra_args=extra_args,
        )
        if ret != 0:
            print("Combined build failed!")
            return ret
        print("Combined build completed successfully!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
