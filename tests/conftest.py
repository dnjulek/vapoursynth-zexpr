"""Pytest configuration and shared fixtures for zexpr tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import vapoursynth as vs

REPO_ROOT = Path(__file__).resolve().parent.parent


def _default_lib() -> Path:
    if sys.platform == "win32":
        return REPO_ROOT / "zig-out" / "bin" / "zexpr.dll"
    if sys.platform == "darwin":
        return REPO_ROOT / "zig-out" / "lib" / "libzexpr.dylib"
    return REPO_ROOT / "zig-out" / "lib" / "libzexpr.so"


DEFAULT_LIB = _default_lib()


def is_sde() -> bool:
    """Best-effort detection of Intel SDE emulation (never load akarin under SDE)."""
    for key in os.environ:
        if key.startswith("SDE_") or key == "PIN_CRT_TZDATA":
            return True
    try:
        maps = Path("/proc/self/maps").read_text(errors="ignore")
        if "sde" in maps.lower() or "pinbin" in maps.lower():
            return True
    except OSError:
        pass
    return False


@pytest.fixture(scope="session")
def zexpr_lib() -> Path:
    env_path = os.environ.get("ZEXPR_LIB")
    path = Path(env_path) if env_path else DEFAULT_LIB
    if not path.is_file():
        pytest.fail(f"zexpr plugin library not found: {path}")
    return path


@pytest.fixture(scope="session")
def core(zexpr_lib: Path) -> vs.Core:
    c = vs.core
    if not hasattr(c, "zexpr"):
        c.std.LoadPlugin(str(zexpr_lib))
    loaded = Path(c.zexpr.plugin_path).resolve()
    expected = zexpr_lib.resolve()
    if loaded != expected:
        pytest.fail(
            f"zexpr autoloaded from {loaded}, not requested {expected}; "
            "uninstall vapoursynth-zexpr from the test venv"
        )
    return c
