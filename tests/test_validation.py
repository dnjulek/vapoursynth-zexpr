"""Phase 5 tests: Validation errors, subprocess-isolated hostile inputs, and post-LoadPlugin execmem fallback (S11/Y13/Z2)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import vapoursynth as vs

from conftest import is_sde
from helpers import make_clip


@pytest.mark.parametrize(
    "expr",
    [
        "x +",                # stack underflow
        "x y",                # stack leftover (2 values)
        "unknown_op",         # unknown token
        "a@",                 # uninitialized variable
        "x[1,2,3]",           # bad spatial syntax
        "x[+1,0]",            # leading '+' in relative offset rejected
        "x[1_0,0]",           # underscore in relative offset rejected
        "1_000",              # underscore in numeric literal rejected
        "-0x10",              # signed hex literal rejected
        "+1",                 # leading '+' literal rejected
        "src99",              # out-of-range clip reference
        "x y +",              # y referenced with only 1 clip
        " ".join(["x"] * 1025) + " sort1025 drop1024",  # Z3: sortN > 1024 rejected (with enough items on stack)
        "x polyval",                                     # missing polyval degree
        "x polyvalx",                                    # non-numeric polyval degree
        "x polyeval-1",                                  # negative polyval degree
        "2 1 x polyval2",                                # polyval stack underflow
        "x y lerp",                                      # lerp stack underflow (needs 3 args)
        " ".join(["1"] * 1026) + " x polyval1025",       # polyval degree > 1024 rejected
    ],
)
def test_invalid_expressions_rejected_in_create(core: vs.Core, expr: str) -> None:
    c = make_clip(core, width=16, height=8, fmt=vs.GRAY8, length=1)
    with pytest.raises(vs.Error):
        core.zexpr.Expr([c], [expr])


def test_invalid_arguments_and_formats_rejected(core: vs.Core) -> None:
    c8 = make_clip(core, width=16, height=8, fmt=vs.GRAY8, length=1)
    c_small = make_clip(core, width=8, height=8, fmt=vs.GRAY8, length=1)
    c_yuv = make_clip(core, width=16, height=8, fmt=vs.YUV420P8, length=1)
    g24_fmt = core.query_video_format(vs.GRAY, vs.INTEGER, 24, 0, 0).id
    c24 = core.std.BlankClip(width=16, height=8, format=g24_fmt, length=1)

    # Mismatched dimensions
    with pytest.raises(vs.Error):
        core.zexpr.Expr([c8, c_small], ["x y +"])

    # Mismatched plane count / subsampling
    with pytest.raises(vs.Error):
        core.zexpr.Expr([c8, c_yuv], ["x y +"])

    # 17..31-bit integer rejected (only 8..16 and 32-bit integer supported)
    with pytest.raises(vs.Error):
        core.zexpr.Expr([c24], ["x"])
    with pytest.raises(vs.Error):
        core.zexpr.Expr([c8], ["x"], format=g24_fmt)

    # Bad opt / boundary
    with pytest.raises(vs.Error):
        core.zexpr.Expr([c8], ["x"], opt=2)
    with pytest.raises(vs.Error):
        core.zexpr.Expr([c8], ["x"], boundary=2)

    # Too many expressions (> numPlanes)
    with pytest.raises(vs.Error):
        core.zexpr.Expr([c8], ["x", "x"])

    # Output format color family / subsampling mismatch (D14)
    with pytest.raises(vs.Error):
        core.zexpr.Expr([c_yuv], ["x"], format=vs.YUV444P8)
    with pytest.raises(vs.Error):
        core.zexpr.Expr([c8], ["x"], format=vs.RGB24)

    # Undefined / invalid format ids rejected cleanly
    with pytest.raises(vs.Error, match="invalid format id"):
        core.zexpr.Expr([c8], ["x"], format=0)
    with pytest.raises(vs.Error, match="invalid format id"):
        core.zexpr.Expr([c8], ["x"], format=12345)

    # Variable format input accepted (Phase C dynamic mode)
    c16 = make_clip(core, width=16, height=8, fmt=vs.GRAY16, length=1, seed=42)
    var_fmt = core.std.Splice([c8, c16], mismatch=True)
    var_out = core.zexpr.Expr([var_fmt], ["x"])
    assert var_out.format.id == vs.NONE
    assert np.array_equal(np.asarray(var_out.get_frame(0)[0]), np.asarray(c8.get_frame(0)[0]))
    assert np.array_equal(np.asarray(var_out.get_frame(1)[0]), np.asarray(c16.get_frame(0)[0]))


@pytest.mark.skipif(
    os.environ.get("ZEXPR_CPU_LEVEL") == "0",
    reason="Scalar interpreter mode explicitly requested via ZEXPR_CPU_LEVEL=0",
)
def test_jit_active_without_execmem_fallback(core: vs.Core) -> None:
    """L2: Fail if the platform denied executable memory and silently fell back to scalar."""
    warnings: list[str] = []

    def log_cb(msg_type: vs.MessageType, msg: str) -> None:
        if msg_type == vs.MESSAGE_TYPE_WARNING:
            warnings.append(msg)

    handle = core.add_log_handler(log_cb)
    try:
        c = core.std.BlankClip(width=32, height=8, format=vs.GRAY8, color=[100], length=1)
        out = core.zexpr.Expr([c], ["x[-1,0] x + 2 /"])
        assert np.all(np.asarray(out.get_frame(0)[0]) == 100)
    finally:
        core.remove_log_handler(handle)

    fallback_warnings = [w for w in warnings if "executable memory" in w.lower() or "scalar" in w.lower()]
    assert not fallback_warnings, f"JIT unexpectedly fell back to scalar interpreter: {fallback_warnings}"


@pytest.mark.skipif(sys.platform != "linux", reason="LD_PRELOAD mmap interposition test is Linux-only")
@pytest.mark.skipif(is_sde(), reason="Subprocess tests are skipped under Intel SDE")
def test_execmem_denied_fallback_logs_single_warning_and_matches_oracle(
    zexpr_lib: Path,
    tmp_path: Path,
) -> None:
    """S11/Y13/Z2: Post-LoadPlugin mmap(PROT_EXEC) denial falls back to scalar interpreter with 1 mtWarning."""
    c_src = tmp_path / "deny_execmem.c"
    so_path = tmp_path / "libdeny_execmem.so"
    c_src.write_text(
        """
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdlib.h>
#include <sys/mman.h>

typedef void *(*mmap_fn)(void *, size_t, int, int, int, off_t);

void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t offset) {
    static mmap_fn real_mmap = NULL;
    if (!real_mmap) {
        real_mmap = (mmap_fn)dlsym(RTLD_NEXT, "mmap");
    }
    if ((prot & PROT_EXEC) && getenv("ZEXPR_DENY_EXECMEM_ACTIVE") != NULL) {
        errno = EACCES;
        return MAP_FAILED;
    }
    return real_mmap(addr, len, prot, flags, fd, offset);
}

#ifdef __GLIBC__
typedef void *(*mmap64_fn)(void *, size_t, int, int, int, off64_t);

void *mmap64(void *addr, size_t len, int prot, int flags, int fd, off64_t offset) {
    static mmap64_fn real_mmap64 = NULL;
    if (!real_mmap64) {
        real_mmap64 = (mmap64_fn)dlsym(RTLD_NEXT, "mmap64");
    }
    if ((prot & PROT_EXEC) && getenv("ZEXPR_DENY_EXECMEM_ACTIVE") != NULL) {
        errno = EACCES;
        return MAP_FAILED;
    }
    return real_mmap64(addr, len, prot, flags, fd, offset);
}
#endif
"""
    )
    subprocess.run(
        ["cc", "-shared", "-fPIC", "-O2", str(c_src), "-o", str(so_path), "-ldl"],
        check=True,
    )

    script = f"""
import os, pathlib, sys
import numpy as np
import vapoursynth as vs

core = vs.core
warnings = []
def log_cb(msg_type, msg):
    if msg_type == vs.MESSAGE_TYPE_WARNING:
        warnings.append(msg)
core.add_log_handler(log_cb)

# 1. LoadPlugin BEFORE enabling PROT_EXEC denial (Y13/Z2); allow matching autoloaded wheel (L3)
if not hasattr(core, "zexpr"):
    core.std.LoadPlugin({str(zexpr_lib)!r})
loaded_path = pathlib.Path(core.zexpr.plugin_path).resolve()
expected_path = pathlib.Path({str(zexpr_lib)!r}).resolve()
assert loaded_path == expected_path, (
    f"zexpr autoloaded from {{loaded_path}}, not requested {{expected_path}}; "
    "uninstall vapoursynth-zexpr from the test venv"
)

# 2. Now deny PROT_EXEC mmap allocations
os.environ["ZEXPR_DENY_EXECMEM_ACTIVE"] = "1"

c = core.std.BlankClip(width=32, height=8, format=vs.YUV420P8, color=[100, 128, 160], length=1)
# Use a non-LUT spatial expression across all 3 planes: must emit exactly ONE warning per filter!
out = core.zexpr.Expr([c], ["x[-1,0] x + 2 /", "x 10 +", "x 5 -"])
f = out.get_frame(0)
assert np.all(np.asarray(f[0]) == 100)
assert np.all(np.asarray(f[1]) == 138)
assert np.all(np.asarray(f[2]) == 155)
assert len(warnings) == 1, f"expected 1 mtWarning, got {{len(warnings)}}: {{warnings}}"
assert "executable memory" in warnings[0].lower() or "scalar" in warnings[0].lower()

# 3. Also verify ZEXPR_CPU_LEVEL=0 produces ZERO warnings
warnings.clear()
os.environ["ZEXPR_CPU_LEVEL"] = "0"
out2 = core.zexpr.Expr([c], ["x[-1,0] x + 2 /"])
assert np.all(np.asarray(out2.get_frame(0)[0]) == 100)
assert len(warnings) == 0, f"expected 0 warnings at ZEXPR_CPU_LEVEL=0, got {{warnings}}"
print("EXECMEM_FALLBACK_OK")
"""
    env = os.environ.copy()
    env.pop("ZEXPR_CPU_LEVEL", None)
    env["LD_PRELOAD"] = str(so_path)
    proc = subprocess.run(
        [sys.executable, "-B", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "EXECMEM_FALLBACK_OK" in proc.stdout
