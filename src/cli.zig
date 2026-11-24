//! Standalone `zexpr-cli` tool (zero VapourSynth / NumPy dependency) for inspecting
//! IR/disassembly and running standalone plane kernels under Intel SDE `-mix`/`-ast`/`-nhm`.

const std = @import("std");
const ast = @import("expr/ast.zig");
const builtins_mod = @import("expr/builtins.zig");
const parser = @import("expr/parser.zig");
const lower_mod = @import("expr/lower.zig");
const optimize_mod = @import("expr/optimize.zig");
const analysis_mod = @import("expr/analysis.zig");
const interp = @import("expr/interp.zig");
const aj = @import("jit/asmjit.zig");
const cpu_mod = @import("jit/cpu.zig");
const codegen = @import("jit/codegen.zig");

fn parseFormat(s: []const u8) ?ast.SampleFormat {
    if (std.mem.eql(u8, s, "u8")) return .{ .is_float = false, .bits_per_sample = 8 };
    if (std.mem.eql(u8, s, "u10")) return .{ .is_float = false, .bits_per_sample = 10 };
    if (std.mem.eql(u8, s, "u16")) return .{ .is_float = false, .bits_per_sample = 16 };
    if (std.mem.eql(u8, s, "u32") or std.mem.eql(u8, s, "i32")) return .{ .is_float = false, .bits_per_sample = 32 };
    if (std.mem.eql(u8, s, "f16")) return .{ .is_float = true, .bits_per_sample = 16 };
    if (std.mem.eql(u8, s, "f32")) return .{ .is_float = true, .bits_per_sample = 32 };
    return null;
}

fn parseSize(s: []const u8) ?struct { w: usize, h: usize } {
    const x_pos = std.mem.findScalar(u8, s, 'x') orelse return null;
    const w = std.fmt.parseInt(usize, s[0..x_pos], 10) catch return null;
    const h = std.fmt.parseInt(usize, s[x_pos + 1 ..], 10) catch return null;
    if (w == 0 or h == 0 or w > 65536 or h > 65536) return null;
    const pixels = std.math.mul(usize, w, h) catch return null;
    if (pixels > 64_000_000) return null;
    return .{ .w = w, .h = h };
}

pub fn main(init: std.process.Init) u8 {
    run(init) catch |err| {
        std.debug.print("zexpr-cli error: {s}\n", .{@errorName(err)});
        return 1;
    };
    return 0;
}

fn run(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var out_buf: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writerStreaming(init.io, &out_buf);
    const stdout = &stdout_writer.interface;

    if (args.len < 2) {
        try stdout.print("Usage: zexpr-cli <run|disasm> [--format u8|u10|u16|u32|f16|f32] [--size WxH] [--opt 0|1] [--boundary 0|1] [--plane 0|1|2] [--color-family gray|rgb|yuv] [--range limited|full] [--canonical-nan] <expr>\n", .{});
        try stdout.flush();
        return error.InvalidArgs;
    }

    const cmd = args[1];
    const is_disasm = std.mem.eql(u8, cmd, "disasm");
    const is_run = std.mem.eql(u8, cmd, "run");
    if (!is_disasm and !is_run) return error.InvalidArgs;

    var fmt: ast.SampleFormat = .{ .is_float = true, .bits_per_sample = 32 };
    var width: usize = 256;
    var height: usize = 16;
    var opt_int = false;
    var boundary: ast.BoundaryMode = .clamp;
    var plane_idx: u8 = 0;
    var color_family: builtins_mod.ColorFamily = .gray;
    var opt_full_range: ?bool = null;
    var canonical_nan = false;
    var expr_str: ?[]const u8 = null;

    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--")) {
            i += 1;
            if (i >= args.len or expr_str != null or i + 1 != args.len) return error.InvalidArgs;
            expr_str = args[i];
            break;
        } else if (std.mem.eql(u8, a, "--format")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            fmt = parseFormat(args[i]) orelse return error.InvalidFormat;
        } else if (std.mem.eql(u8, a, "--size")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            const sz = parseSize(args[i]) orelse return error.InvalidSize;
            width = sz.w;
            height = sz.h;
        } else if (std.mem.eql(u8, a, "--opt")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            if (std.mem.eql(u8, args[i], "1")) {
                opt_int = true;
            } else if (std.mem.eql(u8, args[i], "0")) {
                opt_int = false;
            } else {
                return error.InvalidArgs;
            }
        } else if (std.mem.eql(u8, a, "--boundary")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            if (std.mem.eql(u8, args[i], "1")) {
                boundary = .mirror;
            } else if (std.mem.eql(u8, args[i], "0")) {
                boundary = .clamp;
            } else {
                return error.InvalidArgs;
            }
        } else if (std.mem.eql(u8, a, "--plane")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            if (std.mem.eql(u8, args[i], "0")) {
                plane_idx = 0;
            } else if (std.mem.eql(u8, args[i], "1")) {
                plane_idx = 1;
            } else if (std.mem.eql(u8, args[i], "2")) {
                plane_idx = 2;
            } else {
                return error.InvalidArgs;
            }
        } else if (std.mem.eql(u8, a, "--color-family")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            if (std.mem.eql(u8, args[i], "gray")) {
                color_family = .gray;
            } else if (std.mem.eql(u8, args[i], "rgb")) {
                color_family = .rgb;
            } else if (std.mem.eql(u8, args[i], "yuv")) {
                color_family = .yuv;
            } else {
                return error.InvalidArgs;
            }
        } else if (std.mem.eql(u8, a, "--range")) {
            i += 1;
            if (i >= args.len) return error.MissingFlagValue;
            if (std.mem.eql(u8, args[i], "limited")) {
                opt_full_range = false;
            } else if (std.mem.eql(u8, args[i], "full")) {
                opt_full_range = true;
            } else {
                return error.InvalidArgs;
            }
        } else if (std.mem.eql(u8, a, "--canonical-nan")) {
            canonical_nan = true;
        } else if (std.mem.startsWith(u8, a, "--")) {
            return error.UnknownFlag;
        } else {
            if (expr_str != null) return error.InvalidArgs;
            expr_str = a;
        }
    }

    const expr = expr_str orelse return error.MissingExpression;
    const cpu = cpu_mod.detect();
    const profile = interp.TargetProfile.fromCpuWithOracleFma(cpu, cpu_mod.readOracleFmaOverride());
    const fmts = [_]ast.SampleFormat{ fmt, fmt, fmt, fmt };
    const is_full = if (fmt.is_float)
        true
    else
        opt_full_range orelse (color_family == .rgb);
    const meta: builtins_mod.ClipMeta = .{
        .is_float = fmt.is_float,
        .bits_per_sample = fmt.bits_per_sample,
        .color_family = color_family,
        .plane_idx = plane_idx,
        .is_full_range = is_full,
    };
    const metas = [_]builtins_mod.ClipMeta{ meta, meta, meta, meta };
    const builtin_ctx: builtins_mod.ResolveContext = .{
        .src_metas = &metas,
        .dst_meta = meta,
        .opt_int = opt_int,
    };

    var prog = try parser.parse(gpa, expr, .{
        .mode = .expr,
        .num_clips = 4,
        .default_boundary = boundary,
        .opt_int = opt_int,
        .builtin_ctx = &builtin_ctx,
    });
    defer prog.deinit(gpa);

    try lower_mod.lower(gpa, &prog, profile);
    try optimize_mod.optimize(gpa, &prog, profile, .{
        .enable_value_changing = true,
        .enable_fma = profile.use_fma,
    });

    var info = try analysis_mod.analyze(gpa, &prog, &fmts, fmt, @intCast(width), @intCast(height), .{
        .has_fast_u8_lut = aj.Builder.cpuSupportsFastLutU8(cpu),
        .f32_lanes = cpu.f32_lanes,
    });
    defer info.deinit(gpa);

    if (is_disasm) {
        if (cpu.isa == .scalar) return error.UnsupportedFeature;
        var rt = try aj.Runtime.init();
        defer rt.deinit();

        var compiled = try codegen.compilePlane(gpa, &rt, cpu, &prog, &info, .{
            .width = width,
            .height = height,
            .src_fmts = &fmts,
            .dst_fmt = fmt,
            .capture_disasm = true,
        });
        defer compiled.deinit(gpa, &rt);

        try stdout.print("; ISA={s} vec_bytes={d} code_size={d}\n", .{ cpu.isaName(), cpu.vec_bytes, compiled.kernel.code_size });
        if (compiled.disasm) |d| {
            try stdout.writeAll(d);
        }
        try stdout.flush();
        return;
    }

    // Execute plane kernel (or scalar oracle when ZEXPR_CPU_LEVEL=0) on deterministic aligned buffers.
    const bps: usize = fmt.bytesPerSample();
    const stride_bytes: usize = std.mem.alignForward(usize, width * bps, 64);
    const total_bytes = stride_bytes * height;

    const src0 = try gpa.alignedAlloc(u8, .@"64", total_bytes);
    defer gpa.free(src0);
    const src1 = try gpa.alignedAlloc(u8, .@"64", total_bytes);
    defer gpa.free(src1);
    const src2 = try gpa.alignedAlloc(u8, .@"64", total_bytes);
    defer gpa.free(src2);
    const src3 = try gpa.alignedAlloc(u8, .@"64", total_bytes);
    defer gpa.free(src3);
    const dst = try gpa.alignedAlloc(u8, .@"64", total_bytes);
    defer gpa.free(dst);
    @memset(dst, 0);

    const props = try gpa.alloc(f32, prog.props.items.len);
    defer gpa.free(props);
    @memset(props, std.math.nan(f32));

    if (fmt.is_float and fmt.bits_per_sample == 32) {
        const s0: [*]f32 = @ptrCast(@alignCast(src0.ptr));
        const s1: [*]f32 = @ptrCast(@alignCast(src1.ptr));
        const s2: [*]f32 = @ptrCast(@alignCast(src2.ptr));
        const s3: [*]f32 = @ptrCast(@alignCast(src3.ptr));
        const elems = total_bytes / 4;
        for (0..elems) |idx| {
            s0[idx] = @as(f32, @floatFromInt(idx & 255)) / 255.0;
            s1[idx] = @as(f32, @floatFromInt((idx * 3 + 7) & 255)) / 255.0;
            s2[idx] = @as(f32, @floatFromInt((idx * 5 + 13) & 255)) / 255.0;
            s3[idx] = @as(f32, @floatFromInt((idx * 7 + 29) & 255)) / 255.0;
        }
    } else {
        for (0..total_bytes) |idx| {
            src0[idx] = @intCast(idx & 255);
            src1[idx] = @intCast((idx * 3 + 7) & 255);
            src2[idx] = @intCast((idx * 5 + 13) & 255);
            src3[idx] = @intCast((idx * 7 + 29) & 255);
        }
    }

    const srcs = [_][*]const u8{ src0.ptr, src1.ptr, src2.ptr, src3.ptr };
    const strides = [_]isize{ @intCast(stride_bytes), @intCast(stride_bytes), @intCast(stride_bytes), @intCast(stride_bytes) };
    var code_size: usize = 0;

    if (cpu.isa == .scalar) {
        try interp.evalPlane(gpa, &prog, profile, .{
            .dst = dst.ptr,
            .dst_stride = @intCast(stride_bytes),
            .dst_fmt = fmt,
            .srcs = &srcs,
            .src_strides = &strides,
            .src_fmts = &fmts,
            .width = width,
            .height = height,
            .frame_n = 0,
            .props = props,
        });
    } else {
        var rt = try aj.Runtime.init();
        defer rt.deinit();

        var compiled = try codegen.compilePlane(gpa, &rt, cpu, &prog, &info, .{
            .width = width,
            .height = height,
            .src_fmts = &fmts,
            .dst_fmt = fmt,
            .capture_disasm = false,
        });
        defer compiled.deinit(gpa, &rt);
        code_size = compiled.kernel.code_size;
        compiled.kernel.func(
            dst.ptr,
            @intCast(stride_bytes),
            &srcs,
            &strides,
            0,
            if (props.len > 0) props.ptr else null,
        );
    }

    var hash: u64 = 14695981039346656037;
    if (canonical_nan and fmt.is_float and fmt.bits_per_sample == 16) {
        for (0..height) |y| {
            const row = dst[y * stride_bytes ..][0 .. width * 2];
            for (0..width) |x| {
                const raw = std.mem.readInt(u16, row[x * 2 ..][0..2], .little);
                const is_nan = (raw & 0x7C00) == 0x7C00 and (raw & 0x03FF) != 0;
                const norm: u16 = if (is_nan) 0x7E00 else raw;
                var bytes: [2]u8 = undefined;
                std.mem.writeInt(u16, &bytes, norm, .little);
                for (bytes) |byte| {
                    hash ^= byte;
                    hash *%= 1099511628211;
                }
            }
        }
    } else if (canonical_nan and fmt.is_float and fmt.bits_per_sample == 32) {
        for (0..height) |y| {
            const row = dst[y * stride_bytes ..][0 .. width * 4];
            for (0..width) |x| {
                const raw = std.mem.readInt(u32, row[x * 4 ..][0..4], .little);
                const is_nan = (raw & 0x7F80_0000) == 0x7F80_0000 and (raw & 0x007F_FFFF) != 0;
                const norm: u32 = if (is_nan) 0x7FC0_0000 else raw;
                var bytes: [4]u8 = undefined;
                std.mem.writeInt(u32, &bytes, norm, .little);
                for (bytes) |byte| {
                    hash ^= byte;
                    hash *%= 1099511628211;
                }
            }
        }
    } else {
        for (0..height) |y| {
            const row = dst[y * stride_bytes ..][0 .. width * bps];
            for (row) |byte| {
                hash ^= byte;
                hash *%= 1099511628211;
            }
        }
    }
    try stdout.print("ok isa={s} size={d}x{d} code_size={d} fnv64=0x{x:0>16}\n", .{
        cpu.isaName(), width, height, code_size, hash,
    });
    try stdout.flush();
}
