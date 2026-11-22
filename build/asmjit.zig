//! Repo-owned build wiring for upstream AsmJit, the C ABI shim,
//! and `b.addTranslateC` bindings.

const std = @import("std");

pub const Bundle = struct {
    lib: *std.Build.Step.Compile,
    asmjit_dep: *std.Build.Dependency,
    zasmjit_c: *std.Build.Module,
    target: std.Build.ResolvedTarget,
    optimize: std.lang.Optimize,
};

pub fn shimCppFlags(optimize: std.lang.Optimize) []const []const u8 {
    return if (optimize == .debug)
        &.{
            "-std=c++20",
            "-fvisibility=hidden",
            "-fno-exceptions",
            "-fno-rtti",
            "-DASMJIT_STATIC",
            "-DASMJIT_NO_FOREIGN",
            "-DASMJIT_BUILD_DEBUG",
        }
    else
        &.{
            "-std=c++20",
            "-fvisibility=hidden",
            "-fno-exceptions",
            "-fno-rtti",
            "-DASMJIT_STATIC",
            "-DASMJIT_NO_FOREIGN",
            "-DASMJIT_BUILD_RELEASE",
        };
}

pub fn create(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.lang.Optimize,
) Bundle {
    const asmjit_dep = b.dependency("asmjit", .{});

    const asmjit_mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .pic = if (target.result.ofmt == .elf) true else null,
        .link_libc = true,
        .link_libcpp = true,
    });
    asmjit_mod.addIncludePath(asmjit_dep.path("."));

    var flags: std.ArrayList([]const u8) = .empty;
    defer flags.deinit(b.allocator);

    flags.appendSlice(b.allocator, &.{
        "-std=c++20",
        "-fvisibility=hidden",
        "-fno-exceptions",
        "-fno-rtti",
        "-fno-math-errno",
        "-fno-threadsafe-statics",
        "-fno-trapping-math",
        "-fno-finite-math-only",
        "-DASMJIT_STATIC",
        "-DASMJIT_NO_FOREIGN",
        if (optimize == .debug) "-DASMJIT_BUILD_DEBUG" else "-DASMJIT_BUILD_RELEASE",
    }) catch @panic("OOM");

    if (!target.result.os.tag.isDarwin()) {
        flags.append(b.allocator, "-fno-semantic-interposition") catch @panic("OOM");
    }

    asmjit_mod.addCSourceFiles(.{
        .root = asmjit_dep.path("."),
        .files = &upstream_sources,
        .flags = flags.items,
    });

    const lib = b.addLibrary(.{
        .name = "asmjit",
        .linkage = .static,
        .root_module = asmjit_mod,
    });

    const tc = b.addTranslateC(.{
        .root_source_file = b.path("src/asmjit_shim/zasmjit.h"),
        .target = target,
        .optimize = optimize,
    });

    return .{
        .lib = lib,
        .asmjit_dep = asmjit_dep,
        .zasmjit_c = tc.createModule(),
        .target = target,
        .optimize = optimize,
    };
}

pub fn wireShim(b: *std.Build, bundle: Bundle, mod: *std.Build.Module) void {
    mod.linkLibrary(bundle.lib);
    mod.link_libc = true;
    mod.link_libcpp = true;
    mod.addIncludePath(bundle.asmjit_dep.path("."));
    mod.addIncludePath(b.path("src/asmjit_shim"));
    mod.addCSourceFile(.{
        .file = b.path("src/asmjit_shim/zasmjit.cpp"),
        .flags = shimCppFlags(bundle.optimize),
    });
    mod.addImport("zasmjit_c", bundle.zasmjit_c);
}

pub fn wireAbiProbe(b: *std.Build, bundle: Bundle, mod: *std.Build.Module) void {
    mod.addIncludePath(b.path("src/asmjit_shim"));
    mod.addCSourceFile(.{
        .file = b.path("src/asmjit_shim/abi_probe.cpp"),
        .flags = shimCppFlags(bundle.optimize),
    });
}

const upstream_sources = [_][]const u8{
    // axl
    "asmjit/axl/arena.cpp",
    "asmjit/axl/arena_bit_set.cpp",
    "asmjit/axl/arena_hash.cpp",
    "asmjit/axl/arena_vector.cpp",
    // core (including pauth.cpp)
    "asmjit/core/arch_traits.cpp",
    "asmjit/core/assembler.cpp",
    "asmjit/core/builder.cpp",
    "asmjit/core/code_holder.cpp",
    "asmjit/core/code_writer.cpp",
    "asmjit/core/compiler.cpp",
    "asmjit/core/const_pool.cpp",
    "asmjit/core/cpu_info.cpp",
    "asmjit/core/debug_utils.cpp",
    "asmjit/core/emit_helper.cpp",
    "asmjit/core/emitter.cpp",
    "asmjit/core/emitter_utils.cpp",
    "asmjit/core/environment.cpp",
    "asmjit/core/error.cpp",
    "asmjit/core/error_handler.cpp",
    "asmjit/core/formatter.cpp",
    "asmjit/core/func.cpp",
    "asmjit/core/func_args_context.cpp",
    "asmjit/core/inst.cpp",
    "asmjit/core/inst_db.cpp",
    "asmjit/core/jit_allocator.cpp",
    "asmjit/core/jit_runtime.cpp",
    "asmjit/core/logger.cpp",
    "asmjit/core/pauth.cpp",
    "asmjit/core/ra_local.cpp",
    "asmjit/core/ra_pass.cpp",
    "asmjit/core/ra_stack.cpp",
    "asmjit/core/string.cpp",
    "asmjit/core/target.cpp",
    "asmjit/core/type.cpp",
    "asmjit/core/virt_mem.cpp",
    // arm
    "asmjit/arm/a64_assembler.cpp",
    "asmjit/arm/a64_builder.cpp",
    "asmjit/arm/a64_compiler.cpp",
    "asmjit/arm/a64_emit_helper.cpp",
    "asmjit/arm/a64_formatter.cpp",
    "asmjit/arm/a64_func.cpp",
    "asmjit/arm/a64_inst_api.cpp",
    "asmjit/arm/a64_inst_db.cpp",
    "asmjit/arm/a64_ra_pass.cpp",
    "asmjit/arm/arm_formatter.cpp",
    // x86
    "asmjit/x86/x86_assembler.cpp",
    "asmjit/x86/x86_builder.cpp",
    "asmjit/x86/x86_compiler.cpp",
    "asmjit/x86/x86_emit_helper.cpp",
    "asmjit/x86/x86_formatter.cpp",
    "asmjit/x86/x86_func.cpp",
    "asmjit/x86/x86_inst_api.cpp",
    "asmjit/x86/x86_inst_db.cpp",
    "asmjit/x86/x86_ra_pass.cpp",
    // ujit
    "asmjit/ujit/uni_compiler_a64.cpp",
    "asmjit/ujit/uni_compiler_x86.cpp",
    "asmjit/ujit/vec_const_table.cpp",
};
