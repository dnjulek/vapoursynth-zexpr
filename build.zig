//! Build script for the `zexpr` VapourSynth plugin and its host-free test suite.

const std = @import("std");
const asmjit_build = @import("build/asmjit.zig");
const build_zon = @import("build.zig.zon");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const strip = b.option(bool, "strip", "Strip debug symbols from the plugin library") orelse (optimize != .debug);
    const win_dllmain = b.option(bool, "win-dllmain", "Link Windows DllMain teardown hook (disable only for negative-control leak tests)") orelse true;

    const asmjit_bundle = asmjit_build.create(b, target, optimize);

    // Plugin metadata options (version as SemanticVersion).
    const zon_options = b.addOptions();
    zon_options.addOption(
        std.SemanticVersion,
        "version",
        std.SemanticVersion.parse(build_zon.version) catch @panic("invalid version in build.zig.zon"),
    );

    // Plugin shared library (`zig-out/lib/libzexpr.so`).
    const vs_dep = b.dependency("vapoursynth", .{
        .target = target,
        .optimize = optimize,
    });

    const plugin_mod = b.createModule(.{
        .root_source_file = b.path("src/zexpr.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
    });
    plugin_mod.addImport("vapoursynth", vs_dep.module("vapoursynth"));
    plugin_mod.addOptions("zon", zon_options);
    asmjit_build.wireShim(b, asmjit_bundle, plugin_mod);
    if (target.result.os.tag == .windows and win_dllmain) {
        plugin_mod.addCSourceFile(.{
            .file = b.path("src/vs/win_dllmain.c"),
            .flags = &.{"-std=c11"},
        });
    }

    const lib = b.addLibrary(.{
        .name = "zexpr",
        .linkage = .dynamic,
        .root_module = plugin_mod,
    });
    lib.use_llvm = true;
    if (target.result.ofmt == .elf) {
        lib.use_lld = true;
        lib.setVersionScript(b.path("build/exports.map"));
    }

    b.installArtifact(lib);

    // Host-free unit test binary (no VapourSynth dependency).
    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/tests.zig"),
        .target = target,
        .optimize = optimize,
    });
    asmjit_build.wireShim(b, asmjit_bundle, test_mod);
    asmjit_build.wireAbiProbe(b, asmjit_bundle, test_mod);

    const unit_tests = b.addTest(.{
        .name = "zexpr-tests",
        .root_module = test_mod,
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run host-free unit, ABI, and JIT smoke tests");
    test_step.dependOn(&run_unit_tests.step);

    // Install the host-free test binary for direct SDE runs (`zig build install-tests`).
    const install_tests_step = b.step("install-tests", "Install the host-free test binary to bin/zexpr-tests");
    const install_tests = b.addInstallArtifact(unit_tests, .{});
    install_tests_step.dependOn(&install_tests.step);

    // Standalone CLI tool (`zig build cli` -> `zig-out/bin/zexpr-cli`, zero VapourSynth dependency).
    const cli_mod = b.createModule(.{
        .root_source_file = b.path("src/cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    asmjit_build.wireShim(b, asmjit_bundle, cli_mod);

    const cli_exe = b.addExecutable(.{
        .name = "zexpr-cli",
        .root_module = cli_mod,
    });
    const install_cli = b.addInstallArtifact(cli_exe, .{});
    const cli_step = b.step("cli", "Build standalone zexpr-cli for disasm and SDE profiling");
    cli_step.dependOn(&install_cli.step);
}
