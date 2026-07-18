"""Shared fixtures for the M5 binding gates."""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def cli_binary() -> Path:
    """The release CLI binary, built on demand (cheap when cached).

    The CLI parity gate compares `run_*()` output against the `.npy` files the
    Rust CLI writes, so both sides must come from the same source tree.
    """
    if shutil.which("cargo") is None:
        pytest.skip("cargo not available to build the CLI for the parity gate")
    subprocess.run(
        ["cargo", "build", "--release", "-p", "beamprop"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    binary = REPO_ROOT / "target" / "release" / "beamprop"
    assert binary.exists(), f"CLI binary missing at {binary}"
    return binary
