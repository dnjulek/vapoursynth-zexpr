import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from packaging import tags

LIBRARY_NAMES = frozenset({"libzexpr.so", "libzexpr.dylib", "zexpr.dll"})

# (zig_target, python_platform_tag); -Dtarget implies -Dcpu=baseline in Zig
TARGETS = {
    "aarch64-linux-gnu": ("aarch64-linux-gnu.2.17", "manylinux_2_17_aarch64"),
    "aarch64-linux-musl": ("aarch64-linux-musl", "musllinux_1_2_aarch64"),
    "x86_64-linux-gnu": ("x86_64-linux-gnu.2.17", "manylinux_2_17_x86_64"),
    "x86_64-linux-musl": ("x86_64-linux-musl", "musllinux_1_2_x86_64"),
    "aarch64-macos": ("aarch64-macos.11.0", "macosx_11_0_arm64"),
    "x86_64-macos": ("x86_64-macos.11.0", "macosx_11_0_x86_64"),
    "x86_64-windows": ("x86_64-windows", "win_amd64"),
}


class CustomHook(BuildHookInterface[Any]):
    @property
    def target_dir(self) -> Path:
        return Path(self.root) / "vapoursynth" / "plugins"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        build_data["pure_python"] = False
        shutil.rmtree(self.target_dir, ignore_errors=True)
        self.target_dir.mkdir(parents=True, exist_ok=True)

        cmd = [sys.executable, "-m", "ziglang", "build", "-Doptimize=ReleaseFast"]
        if ztarget := os.environ.get("ZTARGET"):
            if ztarget not in TARGETS:
                raise ValueError(f"Unsupported ZTARGET={ztarget!r}")
            zig_target, py_tag = TARGETS[ztarget]
            build_data["tag"] = f"py3-none-{py_tag}"
            cmd.append(f"-Dtarget={zig_target}")
        else:
            build_data["tag"] = f"py3-none-{next(tags.platform_tags())}"
            cmd.append("-Dcpu=baseline")

        with tempfile.TemporaryDirectory(prefix="zexpr-wheel-") as tmpdir:
            prefix = Path(tmpdir)
            subprocess.run([*cmd, "--prefix", str(prefix)], check=True, cwd=self.root)
            libs = [
                p
                for sub in ("lib", "bin")
                if (prefix / sub).is_dir()
                for p in (prefix / sub).iterdir()
                if p.is_file() and p.name in LIBRARY_NAMES
            ]
            if len(libs) != 1:
                raise RuntimeError(f"Expected 1 plugin library under {prefix}, found: {libs}")
            shutil.copy2(libs[0], self.target_dir / libs[0].name)

    def finalize(self, version: str, build_data: dict[str, Any], artifact_path: str) -> None:
        shutil.rmtree(self.target_dir, ignore_errors=True)
        try:
            self.target_dir.parent.rmdir()
        except OSError:
            pass
