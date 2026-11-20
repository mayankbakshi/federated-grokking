"""Default output locations for new runs."""

from pathlib import Path


def repo_root():
    return Path(__file__).resolve().parents[1]


def worlds_dir(task):
    return repo_root() / "output" / "worlds" / task


def run_dir(name):
    return repo_root() / "output" / name
