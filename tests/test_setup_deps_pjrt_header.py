# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
# SPDX-License-Identifier: Apache-2.0
"""The bootstrap must install the pinned PJRT blob, not a stale/dirty header."""

import os
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup_deps.sh"
_HEADER = Path("xla/pjrt/c/pjrt_c_api.h")
_PINNED = b"// pinned PJRT header\n#define PJRT_API_MINOR 114\n"


def _git(directory, *args):
    return subprocess.run(
        ["git", "-C", str(directory), *args], check=True, capture_output=True, text=True, timeout=30
    ).stdout.strip()


@pytest.fixture
def installed_deps(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q")
    _git(origin, "config", "user.name", "Dependency fixture")
    _git(origin, "config", "user.email", "fixture@example.invalid")
    _git(origin, "config", "commit.gpgsign", "false")
    (origin / _HEADER).parent.mkdir(parents=True)
    (origin / _HEADER).write_bytes(_PINNED)
    _git(origin, "add", str(_HEADER))
    _git(origin, "commit", "-qm", "Pinned fixture")
    pin = _git(origin, "rev-parse", "HEAD")
    build = tmp_path / "build"
    build.mkdir()
    for name in ("stablehlo", "llvm-project", "xla"):
        subprocess.run(["git", "clone", "-q", str(origin), str(build / name)], check=True, timeout=30)
    prefix = tmp_path / "prefix"
    for relative in (
        "lib/libabsl_base.a",
        "lib/libprotobuf.a",
        "lib/libStablehloOps.a",
        "lib/cmake/mlir/MLIRConfig.cmake",
    ):
        path = prefix / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    # Pytest may display fixture values on failure; do not carry credentials
    # from the parent environment into this host-only fixture.
    env = {
        **{key: os.environ[key] for key in ("HOME", "PATH", "TMPDIR", "LANG") if key in os.environ},
        "PREFIX": str(prefix),
        "BUILD_DIR": str(build),
        "JOBS": "1",
        "XLA_COMMIT": pin,
        "STABLEHLO_COMMIT": pin,
        "LLVM_COMMIT_OVERRIDE": pin,
        "MACOSX_DEPLOYMENT_TARGET": "15.0",
    }
    return prefix, build, env


def _setup(env, cwd):
    result = subprocess.run(["/bin/bash", str(_SCRIPT)], cwd=cwd, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_existing_pjrt_header_is_refreshed_to_pinned_blob(installed_deps, tmp_path):
    prefix, _, env = installed_deps
    installed = prefix / "include" / _HEADER
    installed.parent.mkdir(parents=True)
    installed.write_bytes(b"// stale PJRT API 90\n")
    _setup(env, tmp_path)
    assert installed.read_bytes() == _PINNED


def test_dirty_checkout_header_is_preserved_but_not_installed(installed_deps, tmp_path):
    prefix, build, env = installed_deps
    checkout_header = build / "xla" / _HEADER
    dirty = b"// user edits must remain here, not in the installation\n"
    checkout_header.write_bytes(dirty)
    _setup(env, tmp_path)
    assert checkout_header.read_bytes() == dirty
    assert (prefix / "include" / _HEADER).read_bytes() == _PINNED


def test_current_pinned_header_is_not_rewritten(installed_deps, tmp_path):
    prefix, _, env = installed_deps
    installed = prefix / "include" / _HEADER
    installed.parent.mkdir(parents=True)
    installed.write_bytes(_PINNED)
    before = installed.stat().st_mtime_ns
    _setup(env, tmp_path)
    assert installed.read_bytes() == _PINNED
    assert installed.stat().st_mtime_ns == before
