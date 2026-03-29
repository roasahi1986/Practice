#!/usr/bin/env python3
"""
Check dependency versions against latest releases on GitHub.

Usage:
    python3 script/check_deps.py [--update] [--filter NAME]

Options:
    --update    Update deps.yaml with latest versions (not implemented yet)
    --filter    Only check dependencies matching NAME

Environment:
    GITHUB_TOKEN    GitHub personal access token for higher rate limits (5000/hour vs 60/hour)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, List, Dict

import yaml

# GitHub API token for higher rate limits
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")


def check_rate_limit() -> Optional[Dict]:
    """Check GitHub API rate limit status."""
    url = "https://api.github.com/rate_limit"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("User-Agent", "check-deps-script")
    if GITHUB_TOKEN:
        req.add_header("Authorization", f"token {GITHUB_TOKEN}")

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            return data.get("rate", {})
    except Exception:
        return None


def get_latest_release(repo: str) -> Optional[Dict]:
    """Fetch latest release info from GitHub API."""
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("User-Agent", "check-deps-script")
    if GITHUB_TOKEN:
        req.add_header("Authorization", f"token {GITHUB_TOKEN}")

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            return {
                "tag": data.get("tag_name", ""),
                "name": data.get("name", ""),
                "published_at": data.get("published_at", ""),
                "html_url": data.get("html_url", ""),
            }
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # Try getting latest tag instead (some repos don't use releases)
            return get_latest_tag(repo)
        return None
    except Exception:
        return None


def get_latest_tag(repo: str) -> Optional[Dict]:
    """Fetch latest tag from GitHub API (fallback for repos without releases)."""
    url = f"https://api.github.com/repos/{repo}/tags"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("User-Agent", "check-deps-script")
    if GITHUB_TOKEN:
        req.add_header("Authorization", f"token {GITHUB_TOKEN}")

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            if data:
                return {
                    "tag": data[0].get("name", ""),
                    "name": "",
                    "published_at": "",
                    "html_url": f"https://github.com/{repo}/releases/tag/{data[0].get('name', '')}",
                }
            return None
    except Exception:
        return None


def normalize_version(version: str) -> str:
    """Normalize version string for comparison (remove 'v' prefix, etc.)."""
    v = version.strip().lower()
    # Remove common prefixes
    for prefix in ["v", "release-", "rel-", "llvmorg-"]:
        if v.startswith(prefix):
            v = v[len(prefix):]
    return v


def compare_versions(current: str, latest: str) -> str:
    """Compare two version strings and return status."""
    if current.lower() == "latest":
        return "tracking"

    curr_norm = normalize_version(current)
    latest_norm = normalize_version(latest)

    if curr_norm == latest_norm:
        return "current"
    return "outdated"


def load_deps_config(path: Path) -> List[Dict]:
    """Load dependencies from YAML config file."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("dependencies", [])


def check_dependencies(deps: List[Dict], filter_name: Optional[str] = None) -> List[Dict]:
    """Check all dependencies against their latest versions."""
    results = []

    for dep in deps:
        name = dep.get("name", "")
        if filter_name and filter_name.lower() not in name.lower():
            continue

        github = dep.get("github", "")
        current_version = str(dep.get("version", ""))
        dep_type = dep.get("type", "unknown")

        print(f"Checking {name}...", end=" ", flush=True)

        latest = get_latest_release(github)
        if latest:
            latest_version = latest["tag"]
            status = compare_versions(current_version, latest_version)
            print(f"done ({status})")
        else:
            latest_version = "unknown"
            status = "error"
            print("failed to fetch")

        results.append({
            "name": name,
            "github": github,
            "type": dep_type,
            "current": current_version,
            "latest": latest_version,
            "latest_normalized": normalize_version(latest_version) if latest_version != "unknown" else "",
            "status": status,
            "url": latest.get("html_url", "") if latest else "",
        })

    return results


def print_results(results: List[Dict]) -> None:
    """Print check results in a formatted table."""
    # Calculate column widths
    name_width = max(len(r["name"]) for r in results) if results else 10

    # Separate by status
    outdated = [r for r in results if r["status"] == "outdated"]
    current = [r for r in results if r["status"] == "current"]
    tracking = [r for r in results if r["status"] == "tracking"]
    errors = [r for r in results if r["status"] == "error"]

    # Print header
    print("\n" + "=" * 90)
    print(f"{'Dependency':<{name_width}}  {'Version':<40}  Status")
    print("-" * 90)

    def print_group(items: List[Dict]) -> None:
        for r in sorted(items, key=lambda x: x["name"]):
            status_icon = {
                "current": "✓",
                "outdated": "⚠",
                "tracking": "→",
                "error": "✗",
            }.get(r["status"], "?")

            if r["status"] == "outdated":
                version_str = f"{r['current']} → {r['latest']}"
            elif r["status"] == "current":
                version_str = r["current"]
            elif r["status"] == "tracking":
                version_str = f"latest ({r['latest']})"
            else:
                version_str = f"{r['current']} (fetch failed)"

            print(f"{r['name']:<{name_width}}  {version_str:<40}  {status_icon} {r['status']}")

    if outdated:
        print("\n[Outdated]")
        print_group(outdated)

    if current:
        print("\n[Up to date]")
        print_group(current)

    if tracking:
        print("\n[Tracking latest]")
        print_group(tracking)

    if errors:
        print("\n[Errors]")
        print_group(errors)

    # Summary
    print("\n" + "=" * 90)
    print(f"Summary: {len(current)} current, {len(outdated)} outdated, {len(tracking)} tracking, {len(errors)} errors")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check dependency versions")
    parser.add_argument("--update", action="store_true", help="Update deps.yaml with latest versions")
    parser.add_argument("--filter", type=str, help="Only check dependencies matching this name")
    args = parser.parse_args()

    # Find deps.yaml relative to this script
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    deps_file = repo_root / "deps.yaml"

    if not deps_file.exists():
        print(f"Error: {deps_file} not found")
        return 1

    # Check rate limit first
    rate = check_rate_limit()
    if rate:
        remaining = rate.get("remaining", 0)
        limit = rate.get("limit", 60)
        print(f"GitHub API rate limit: {remaining}/{limit} remaining")
        if remaining == 0:
            import datetime
            reset_time = datetime.datetime.fromtimestamp(rate.get("reset", 0))
            print(f"Rate limit exceeded. Resets at {reset_time}")
            if not GITHUB_TOKEN:
                print("Tip: Set GITHUB_TOKEN env var for higher limits (5000/hour)")
            return 1
        print()

    print(f"Loading dependencies from {deps_file}")
    deps = load_deps_config(deps_file)
    print(f"Found {len(deps)} dependencies\n")

    results = check_dependencies(deps, args.filter)
    print_results(results)

    # Return non-zero if there are outdated dependencies
    outdated_count = sum(1 for r in results if r["status"] == "outdated")
    return 1 if outdated_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
