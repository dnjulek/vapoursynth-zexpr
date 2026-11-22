//! Mandatory two-phase semantic lowering (`src/expr/lower.zig`).
//!
//! Runs unconditionally before optional optimizer passes to define the baseline D5/D1-D15 IR:
//! - Phase A (fixpoint loop, no power expansion):
//!   1. Fold all-constant subtrees: C `libm` (`powf`/`expf`/`logf`/`sinf`/`cosf`/`tanf`/`atan2f`/`sqrtf`)
//!      strictly for transcendentals & `sqrt`; `interp.zig`'s exact helpers for every other op (T2/U4/Y9).
//!   2. Chained constant-exponent powers `(a ** b) ** c -> a ** (b * c)` (wrapping `a` in `abs(a)`
//!      when `b` is an even integer and `c` is non-integer, T3).
//!   3. Trivial power rules: `x ** 0 -> 1`, `x ** 1 -> x`, `0 ** e -> 0`, `1 ** e -> 1` (Q1).
//! - Phase B (power expansion):
//!   1. `x ** (n/2) -> sqrt(x ** n)` and `x ** (n/4) -> sqrt(sqrt(x ** n))`.
//!   2. `x ** -N -> 1 / (x ** N)` for positive integer `N`.
//!   3. `x ** N` for positive integer `N <= 64`: iterative balanced split `x^ceil(N/2) * x^floor(N/2)`.
//!   4. `x ** N` for integer `N > 64`: odd `N -> x * (x ** (N - 1))`, even `N -> abs(x) ** N`.
//! - `Op.mod` stays atomic and is never lowered here (R3).

const std = @import("std");
const builtin = @import("builtin");
const ast = @import("ast.zig");
const interp = @import("interp.zig");

extern "c" fn powf(x: f32, y: f32) f32;
extern "c" fn expf(x: f32) f32;
extern "c" fn exp2f(x: f32) f32;
extern "c" fn logf(x: f32) f32;
extern "c" fn log2f(x: f32) f32;
extern "c" fn sinf(x: f32) f32;
extern "c" fn cosf(x: f32) f32;
extern "c" fn tanf(x: f32) f32;
extern "c" fn atanf(x: f32) f32;
extern "c" fn asinf(x: f32) f32;
extern "c" fn acosf(x: f32) f32;
extern "c" fn atan2f(y: f32, x: f32) f32;
extern "c" fn sqrtf(x: f32) f32;

const UnaryLibmFn = *const fn (f32) callconv(.c) f32;
const BinaryLibmFn = *const fn (f32, f32) callconv(.c) f32;

// E1/A3/CI2: Debug and portable baseline builds link Zig's compiler_rt, whose local copies of
// expf/exp2f/logf/log2f/sinf/cosf/tanf/sqrtf shadow host libm at link time. Resolving via
// `dlsym(RTLD_DEFAULT, ...)` (Linux/macOS) or `GetProcAddress(GetModuleHandleA("ucrtbase.dll"), ...)`
// (Windows) binds to the host process's libm (matching `std.Expr` across all builds), falling back
// to the linked symbol when unavailable (e.g. static musl).
const use_host_libm = builtin.link_libc and
    (builtin.os.tag == .linux or builtin.os.tag == .macos or builtin.os.tag == .windows);

extern "c" fn dlsym(handle: ?*anyopaque, symbol: [*:0]const u8) callconv(.c) ?*anyopaque;
extern "kernel32" fn GetModuleHandleA(module_name: ?[*:0]const u8) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn GetProcAddress(module: *anyopaque, proc_name: [*:0]const u8) callconv(.winapi) ?*anyopaque;

fn resolveHostLibmSymbol(comptime name: [:0]const u8) ?*anyopaque {
    if (comptime builtin.os.tag == .windows) {
        const mod = GetModuleHandleA("ucrtbase.dll") orelse return null;
        return GetProcAddress(mod, name.ptr);
    }
    const rtld_default: ?*anyopaque = if (comptime builtin.os.tag == .macos)
        @ptrFromInt(@as(usize, @bitCast(@as(isize, -2))))
    else
        null;
    return dlsym(rtld_default, name.ptr);
}

const LibmCache = struct {
    sqrtf: ?UnaryLibmFn = null,
    expf: ?UnaryLibmFn = null,
    exp2f: ?UnaryLibmFn = null,
    logf: ?UnaryLibmFn = null,
    log2f: ?UnaryLibmFn = null,
    sinf: ?UnaryLibmFn = null,
    cosf: ?UnaryLibmFn = null,
    tanf: ?UnaryLibmFn = null,
    atanf: ?UnaryLibmFn = null,
    asinf: ?UnaryLibmFn = null,
    acosf: ?UnaryLibmFn = null,
    powf: ?BinaryLibmFn = null,
    atan2f: ?BinaryLibmFn = null,
};

var libm_cache: LibmCache = .{};

fn callHostLibm1(comptime name: [:0]const u8, slot: *?UnaryLibmFn, comptime fallback: UnaryLibmFn, x: f32) f32 {
    if (comptime !use_host_libm) return fallback(x);
    const f = @atomicLoad(?UnaryLibmFn, slot, .acquire) orelse blk: {
        const resolved: UnaryLibmFn = if (resolveHostLibmSymbol(name)) |ptr|
            @ptrCast(@alignCast(ptr))
        else
            fallback;
        @atomicStore(?UnaryLibmFn, slot, resolved, .release);
        break :blk resolved;
    };
    return f(x);
}

fn callHostLibm2(comptime name: [:0]const u8, slot: *?BinaryLibmFn, comptime fallback: BinaryLibmFn, a: f32, b: f32) f32 {
    if (comptime !use_host_libm) return fallback(a, b);
    const f = @atomicLoad(?BinaryLibmFn, slot, .acquire) orelse blk: {
        const resolved: BinaryLibmFn = if (resolveHostLibmSymbol(name)) |ptr|
            @ptrCast(@alignCast(ptr))
        else
            fallback;
        @atomicStore(?BinaryLibmFn, slot, resolved, .release);
        break :blk resolved;
    };
    return f(a, b);
}

/// Fold a constant unary op: uses host C `libm` strictly for transcendentals & `sqrt` (T2/U4/Y9, E1),
/// and `interp.zig`'s exact helpers for every other op (including clamped `rsqrt`, E3).
pub fn foldConstUnary(profile: interp.TargetProfile, op: ast.UnaryOp, a: interp.EvalVal, opt_int: bool) interp.EvalVal {
    switch (op) {
        .sqrt => return .{ .f = callHostLibm1("sqrtf", &libm_cache.sqrtf, &sqrtf, a.asF32()) },
        .exp => return .{ .f = callHostLibm1("expf", &libm_cache.expf, &expf, a.asF32()) },
        .exp2 => return .{ .f = callHostLibm1("exp2f", &libm_cache.exp2f, &exp2f, a.asF32()) },
        .log => return .{ .f = callHostLibm1("logf", &libm_cache.logf, &logf, a.asF32()) },
        .log2 => return .{ .f = callHostLibm1("log2f", &libm_cache.log2f, &log2f, a.asF32()) },
        .sin => return .{ .f = callHostLibm1("sinf", &libm_cache.sinf, &sinf, a.asF32()) },
        .cos => return .{ .f = callHostLibm1("cosf", &libm_cache.cosf, &cosf, a.asF32()) },
        .tan => return .{ .f = callHostLibm1("tanf", &libm_cache.tanf, &tanf, a.asF32()) },
        .atan => return .{ .f = callHostLibm1("atanf", &libm_cache.atanf, &atanf, a.asF32()) },
        .asin => return .{ .f = callHostLibm1("asinf", &libm_cache.asinf, &asinf, a.asF32()) },
        .acos => return .{ .f = callHostLibm1("acosf", &libm_cache.acosf, &acosf, a.asF32()) },
        else => return interp.evalUnaryVal(profile, op, a, opt_int),
    }
}

/// Fold a constant binary op: uses host C `libm` strictly for `pow` and `atan2` (T2/U4/Y9, E1),
/// and `interp.zig`'s exact helpers for every other op (including shared unfused `modF32`).
pub fn foldConstBinary(profile: interp.TargetProfile, op: ast.BinaryOp, a: interp.EvalVal, b: interp.EvalVal, opt_int: bool) interp.EvalVal {
    switch (op) {
        .pow => return .{ .f = callHostLibm2("powf", &libm_cache.powf, &powf, a.asF32(), b.asF32()) },
        .atan2 => return .{ .f = callHostLibm2("atan2f", &libm_cache.atan2f, &atan2f, a.asF32(), b.asF32()) },
        else => return interp.evalBinaryVal(profile, op, a, b, opt_int),
    }
}

/// Rounded-up `+1 ULP` reciprocal of `hi - lo` for `linstep` (F3/E2/A1).
/// Ensures `(hi - lo) * inv >= 1.0` so the upper endpoint reaches `1.0` after clamping,
/// shared by both Phase A constant folding and Phase B runtime expansion.
fn linstepReciprocal(lo_v: f32, hi_v: f32) ?f32 {
    const span = hi_v - lo_v;
    if (!std.math.isFinite(lo_v) or !std.math.isFinite(hi_v) or !std.math.isFinite(span) or span == 0.0) {
        return null;
    }
    var inv: f32 = 1.0 / span;
    if (std.math.isFinite(inv) and span * inv < 1.0) {
        inv = @bitCast(@as(u32, @bitCast(inv)) + 1);
    }
    return if (std.math.isFinite(inv)) inv else null;
}

pub fn foldConstTernary(profile: interp.TargetProfile, op: ast.TernaryOp, a: interp.EvalVal, b: interp.EvalVal, c: interp.EvalVal, opt_int: bool) interp.EvalVal {
    if (op == .linstep and !opt_int) {
        const lo_v = a.asF32();
        const hi_v = b.asF32();
        const x_v = c.asF32();
        if (linstepReciprocal(lo_v, hi_v)) |inv| {
            const raw_t = (x_v - lo_v) * inv;
            const clamped_lo = interp.evalBinaryF32(profile, .max, raw_t, 0.0);
            return .{ .f = interp.evalBinaryF32(profile, .min, clamped_lo, 1.0) };
        }
    }
    return interp.evalTernaryVal(profile, op, a, b, c, opt_int);
}

fn nodeConstVal(node: ast.Node) ?interp.EvalVal {
    return switch (node) {
        .const_f32 => |v| .{ .f = v },
        .const_i32 => |v| .{ .i = v },
        else => null,
    };
}

fn constNode(val: interp.EvalVal) ast.Node {
    return switch (val) {
        .f => |v| .{ .const_f32 = v },
        .i => |v| .{ .const_i32 = v },
    };
}

fn isFiniteInteger(v: f32) bool {
    return std.math.isFinite(v) and @trunc(v) == v;
}

fn isEvenInteger(v: f32) bool {
    if (!isFiniteInteger(v)) return false;
    const half = v * 0.5;
    return @trunc(half) == half;
}

fn remapIdx(map: []const ast.NodeIdx, old: ast.NodeIdx) ast.NodeIdx {
    return map[old];
}

/// Run mandatory two-phase semantic lowering in place on `prog`.
pub fn lower(allocator: std.mem.Allocator, prog: *ast.Program, profile: interp.TargetProfile) std.mem.Allocator.Error!void {
    // Phase A: fixpoint loop (no power expansion).
    var iterations: usize = 0;
    while (iterations < 32) : (iterations += 1) {
        const changed = try runPhaseAOnce(allocator, prog, profile);
        if (!changed) break;
    }

    // Phase B: D5 power expansion pass.
    try runPhaseB(allocator, prog);

    // Final Phase A cleanup in case Phase B exposed `x ** 1` or new constant folds
    // (e.g., `x ** 0.5 -> sqrt(x ** 1) -> sqrt(x)`).
    _ = try runPhaseAOnce(allocator, prog, profile);
}

fn runPhaseAOnce(allocator: std.mem.Allocator, prog: *ast.Program, profile: interp.TargetProfile) std.mem.Allocator.Error!bool {
    const old_nodes = prog.nodes.items;
    var new_nodes: std.ArrayList(ast.Node) = .empty;
    errdefer new_nodes.deinit(allocator);
    try new_nodes.ensureTotalCapacity(allocator, old_nodes.len);

    const remap = try allocator.alloc(ast.NodeIdx, old_nodes.len);
    defer allocator.free(remap);

    var changed = false;

    for (old_nodes, 0..) |node, old_idx| {
        const out_idx: ast.NodeIdx = switch (node) {
            .const_f32, .const_i32, .load_clip, .load_rel, .load_prop, .coord_x, .coord_y, .frame_n, .plane_width, .plane_height, .arg_sort_elem => blk: {
                const idx: ast.NodeIdx = @intCast(new_nodes.items.len);
                try new_nodes.append(allocator, node);
                break :blk idx;
            },
            .load_abs => |la| blk: {
                const idx: ast.NodeIdx = @intCast(new_nodes.items.len);
                try new_nodes.append(allocator, .{ .load_abs = .{
                    .clip = la.clip,
                    .x = remapIdx(remap, la.x),
                    .y = remapIdx(remap, la.y),
                    .mode = la.mode,
                } });
                break :blk idx;
            },
            .arg_extrema => |ae| blk: {
                // Remap extra_args in place after loop; preserve node.
                const idx: ast.NodeIdx = @intCast(new_nodes.items.len);
                try new_nodes.append(allocator, .{ .arg_extrema = ae });
                break :blk idx;
            },
            .unary => |u| blk: {
                const a = remapIdx(remap, u.a);
                if (nodeConstVal(new_nodes.items[a])) |ca| {
                    changed = true;
                    const folded = foldConstUnary(profile, u.op, ca, prog.opt_int);
                    const idx: ast.NodeIdx = @intCast(new_nodes.items.len);
                    try new_nodes.append(allocator, constNode(folded));
                    break :blk idx;
                }
                if (u.op == .to_f32 and new_nodes.items[a] == .unary and new_nodes.items[a].unary.op == .to_f32) {
                    changed = true;
                    break :blk a;
                }
                const idx: ast.NodeIdx = @intCast(new_nodes.items.len);
                try new_nodes.append(allocator, .{ .unary = .{ .op = u.op, .a = a } });
                break :blk idx;
            },
            .binary => |b| blk: {
                const a = remapIdx(remap, b.a);
                const rhs = remapIdx(remap, b.b);
                const ca = nodeConstVal(new_nodes.items[a]);
                const cb = nodeConstVal(new_nodes.items[rhs]);

                if (ca != null and cb != null) {
                    changed = true;
                    const folded = foldConstBinary(profile, b.op, ca.?, cb.?, prog.opt_int);
                    const idx: ast.NodeIdx = @intCast(new_nodes.items.len);
                    try new_nodes.append(allocator, constNode(folded));
                    break :blk idx;
                }

                if (b.op == .pow) {
                    // Chained constant-exponent powers: (inner_a ** c1) ** c2 -> inner_a ** (c1 * c2)
                    // (wrapping inner_a in abs() if c1 is an even integer and c2 is non-integer).
                    if (cb) |cb_val| {
                        const c2 = cb_val.asF32();
                        if (new_nodes.items[a] == .binary and new_nodes.items[a].binary.op == .pow) {
                            const inner = new_nodes.items[a].binary;
                            if (nodeConstVal(new_nodes.items[inner.b])) |c1_val| {
                                const c1 = c1_val.asF32();
                                if (std.math.isFinite(c1) and std.math.isFinite(c2)) {
                                    changed = true;
                                    var base = inner.a;
                                    if (isEvenInteger(c1) and !isFiniteInteger(c2)) {
                                        const abs_idx: ast.NodeIdx = @intCast(new_nodes.items.len);
                                        try new_nodes.append(allocator, .{ .unary = .{ .op = .abs, .a = base } });
                                        base = abs_idx;
                                    }
                                    const exp_idx: ast.NodeIdx = @intCast(new_nodes.items.len);
                                    try new_nodes.append(allocator, .{ .const_f32 = c1 * c2 });
                                    const pow_idx: ast.NodeIdx = @intCast(new_nodes.items.len);
                                    try new_nodes.append(allocator, .{ .binary = .{ .op = .pow, .a = base, .b = exp_idx } });
                                    break :blk pow_idx;
                                }
                            }
                        }

                        // Trivial constant-exponent rules: x ** 0 -> 1, x ** 1 -> x.
                        if (c2 == 0.0) {
                            changed = true;
                            const idx: ast.NodeIdx = @intCast(new_nodes.items.len);
                            try new_nodes.append(allocator, .{ .const_f32 = 1.0 });
                            break :blk idx;
                        }
                        if (c2 == 1.0) {
                            changed = true;
                            if (prog.opt_int) {
                                // V3: `pow` always produces f32 even under `opt=1`.
                                const idx: ast.NodeIdx = @intCast(new_nodes.items.len);
                                try new_nodes.append(allocator, .{ .unary = .{ .op = .to_f32, .a = a } });
                                break :blk idx;
                            }
                            break :blk a;
                        }
                    }

                    // Trivial constant-base rules (non-constant exponent e, Q1):
                    // 0 ** e -> 0, 1 ** e -> 1.
                    if (ca) |ca_val| {
                        const base_f = ca_val.asF32();
                        if (base_f == 0.0) {
                            changed = true;
                            const idx: ast.NodeIdx = @intCast(new_nodes.items.len);
                            try new_nodes.append(allocator, .{ .const_f32 = 0.0 });
                            break :blk idx;
                        }
                        if (base_f == 1.0) {
                            changed = true;
                            const idx: ast.NodeIdx = @intCast(new_nodes.items.len);
                            try new_nodes.append(allocator, .{ .const_f32 = 1.0 });
                            break :blk idx;
                        }
                    }
                }

                const idx: ast.NodeIdx = @intCast(new_nodes.items.len);
                try new_nodes.append(allocator, .{ .binary = .{ .op = b.op, .a = a, .b = rhs } });
                break :blk idx;
            },
            .ternary => |t| blk: {
                const a = remapIdx(remap, t.a);
                const b = remapIdx(remap, t.b);
                const c = remapIdx(remap, t.c);
                const ca = nodeConstVal(new_nodes.items[a]);
                const cb = nodeConstVal(new_nodes.items[b]);
                const cc = nodeConstVal(new_nodes.items[c]);
                if (ca != null and cb != null and cc != null) {
                    changed = true;
                    const folded = foldConstTernary(profile, t.op, ca.?, cb.?, cc.?, prog.opt_int);
                    const idx: ast.NodeIdx = @intCast(new_nodes.items.len);
                    try new_nodes.append(allocator, constNode(folded));
                    break :blk idx;
                }
                const idx: ast.NodeIdx = @intCast(new_nodes.items.len);
                try new_nodes.append(allocator, .{ .ternary = .{ .op = t.op, .a = a, .b = b, .c = c } });
                break :blk idx;
            },
        };
        remap[old_idx] = out_idx;
    }

    for (prog.extra_args.items) |*arg| {
        arg.* = remapIdx(remap, arg.*);
    }
    prog.root = remapIdx(remap, prog.root);

    prog.nodes.deinit(allocator);
    prog.nodes = new_nodes;
    return changed;
}

/// Phase B: D5 power expansion pass (`n/2`, `n/4`, `-N`, balanced `N <= 64`, odd/even `N > 64`).
fn runPhaseB(allocator: std.mem.Allocator, prog: *ast.Program) std.mem.Allocator.Error!void {
    const old_nodes = prog.nodes.items;
    var new_nodes: std.ArrayList(ast.Node) = .empty;
    errdefer new_nodes.deinit(allocator);
    try new_nodes.ensureTotalCapacity(allocator, old_nodes.len);

    const remap = try allocator.alloc(ast.NodeIdx, old_nodes.len);
    defer allocator.free(remap);

    for (old_nodes, 0..) |node, old_idx| {
        const out_idx: ast.NodeIdx = switch (node) {
            .const_f32, .const_i32, .load_clip, .load_rel, .load_prop, .coord_x, .coord_y, .frame_n, .plane_width, .plane_height, .arg_extrema, .arg_sort_elem => blk: {
                const idx: ast.NodeIdx = @intCast(new_nodes.items.len);
                try new_nodes.append(allocator, node);
                break :blk idx;
            },
            .load_abs => |la| blk: {
                const idx: ast.NodeIdx = @intCast(new_nodes.items.len);
                try new_nodes.append(allocator, .{ .load_abs = .{
                    .clip = la.clip,
                    .x = remapIdx(remap, la.x),
                    .y = remapIdx(remap, la.y),
                    .mode = la.mode,
                } });
                break :blk idx;
            },
            .unary => |u| blk: {
                const idx: ast.NodeIdx = @intCast(new_nodes.items.len);
                try new_nodes.append(allocator, .{ .unary = .{ .op = u.op, .a = remapIdx(remap, u.a) } });
                break :blk idx;
            },
            .ternary => |t| blk: {
                const a = remapIdx(remap, t.a);
                const b = remapIdx(remap, t.b);
                const c = remapIdx(remap, t.c);
                if (t.op == .linstep) {
                    // E2: expand `linstep(lo, hi, x)` after Phase A constant folding so both literal
                    // bounds (`16 235 x linstep`) and constant-expression bounds (`16 257 * 235 257 * x linstep`)
                    // get the exact-endpoint `+1 ULP` reciprocal rule.
                    const lo = a;
                    const hi = b;
                    const x = c;
                    const raw_t: ast.NodeIdx = raw_blk: {
                        if (!prog.opt_int) {
                            if (nodeConstVal(new_nodes.items[lo])) |lo_c| {
                                if (nodeConstVal(new_nodes.items[hi])) |hi_c| {
                                    if (linstepReciprocal(lo_c.asF32(), hi_c.asF32())) |inv| {
                                        const diff_idx: ast.NodeIdx = @intCast(new_nodes.items.len);
                                        try new_nodes.append(allocator, .{ .binary = .{ .op = .sub, .a = x, .b = lo } });
                                        const inv_idx: ast.NodeIdx = @intCast(new_nodes.items.len);
                                        try new_nodes.append(allocator, .{ .const_f32 = inv });
                                        const mul_idx: ast.NodeIdx = @intCast(new_nodes.items.len);
                                        try new_nodes.append(allocator, .{ .binary = .{ .op = .mul, .a = diff_idx, .b = inv_idx } });
                                        break :raw_blk mul_idx;
                                    }
                                }
                            }
                        }
                        const num_idx: ast.NodeIdx = @intCast(new_nodes.items.len);
                        try new_nodes.append(allocator, .{ .binary = .{ .op = .sub, .a = x, .b = lo } });
                        const den_idx: ast.NodeIdx = @intCast(new_nodes.items.len);
                        try new_nodes.append(allocator, .{ .binary = .{ .op = .sub, .a = hi, .b = lo } });
                        const div_idx: ast.NodeIdx = @intCast(new_nodes.items.len);
                        try new_nodes.append(allocator, .{ .binary = .{ .op = .div, .a = num_idx, .b = den_idx } });
                        break :raw_blk div_idx;
                    };
                    const zero_idx: ast.NodeIdx = @intCast(new_nodes.items.len);
                    try new_nodes.append(allocator, .{ .const_f32 = 0.0 });
                    const one_idx: ast.NodeIdx = @intCast(new_nodes.items.len);
                    try new_nodes.append(allocator, .{ .const_f32 = 1.0 });
                    const clamped_lo: ast.NodeIdx = @intCast(new_nodes.items.len);
                    try new_nodes.append(allocator, .{ .binary = .{ .op = .max, .a = raw_t, .b = zero_idx } });
                    const clamped: ast.NodeIdx = @intCast(new_nodes.items.len);
                    try new_nodes.append(allocator, .{ .binary = .{ .op = .min, .a = clamped_lo, .b = one_idx } });
                    break :blk clamped;
                }
                const idx: ast.NodeIdx = @intCast(new_nodes.items.len);
                try new_nodes.append(allocator, .{ .ternary = .{
                    .op = t.op,
                    .a = a,
                    .b = b,
                    .c = c,
                } });
                break :blk idx;
            },
            .binary => |b| blk: {
                const a = remapIdx(remap, b.a);
                const rhs = remapIdx(remap, b.b);
                if (b.op == .pow) {
                    if (nodeConstVal(new_nodes.items[rhs])) |cb_val| {
                        const exp_f = cb_val.asF32();
                        if (std.math.isFinite(exp_f)) {
                            // V3: `pow` always evaluates in f32 even under `opt=1` so expanded
                            // integer multiplications do not wrap in i32.
                            var base_node = a;
                            if (prog.opt_int and exp_f != 0.0) {
                                const cvt_idx: ast.NodeIdx = @intCast(new_nodes.items.len);
                                try new_nodes.append(allocator, .{ .unary = .{ .op = .to_f32, .a = a } });
                                base_node = cvt_idx;
                            }
                            if (try emitExpandedPow(allocator, &new_nodes, base_node, exp_f)) |expanded| {
                                break :blk expanded;
                            }
                        }
                    }
                }
                const idx: ast.NodeIdx = @intCast(new_nodes.items.len);
                try new_nodes.append(allocator, .{ .binary = .{ .op = b.op, .a = a, .b = rhs } });
                break :blk idx;
            },
        };
        remap[old_idx] = out_idx;
    }

    for (prog.extra_args.items) |*arg| {
        arg.* = remapIdx(remap, arg.*);
    }
    prog.root = remapIdx(remap, prog.root);

    prog.nodes.deinit(allocator);
    prog.nodes = new_nodes;
}

fn emitExpandedPow(
    allocator: std.mem.Allocator,
    nodes: *std.ArrayList(ast.Node),
    base: ast.NodeIdx,
    exp_f: f32,
) std.mem.Allocator.Error!?ast.NodeIdx {
    if (exp_f == 0.0) {
        const idx: ast.NodeIdx = @intCast(nodes.items.len);
        try nodes.append(allocator, .{ .const_f32 = 1.0 });
        return idx;
    }
    if (exp_f == 1.0) return base;

    // Integer exponent?
    if (isFiniteInteger(exp_f)) {
        return try emitIntPow(allocator, nodes, base, exp_f);
    }

    // Half-integer `n / 2` -> `sqrt(base ** n)` (matches std.Expr for any finite integer n).
    const exp2 = exp_f * 2.0;
    if (isFiniteInteger(exp2)) {
        const inner = (try emitIntPow(allocator, nodes, base, exp2)) orelse return null;
        const idx: ast.NodeIdx = @intCast(nodes.items.len);
        try nodes.append(allocator, .{ .unary = .{ .op = .sqrt, .a = inner } });
        return idx;
    }

    // Quarter-integer `n / 4` -> `sqrt(sqrt(base ** n))`.
    const exp4 = exp_f * 4.0;
    if (isFiniteInteger(exp4)) {
        const inner = (try emitIntPow(allocator, nodes, base, exp4)) orelse return null;
        const sq1: ast.NodeIdx = @intCast(nodes.items.len);
        try nodes.append(allocator, .{ .unary = .{ .op = .sqrt, .a = inner } });
        const sq2: ast.NodeIdx = @intCast(nodes.items.len);
        try nodes.append(allocator, .{ .unary = .{ .op = .sqrt, .a = sq1 } });
        return sq2;
    }

    return null;
}

fn emitIntPow(
    allocator: std.mem.Allocator,
    nodes: *std.ArrayList(ast.Node),
    base: ast.NodeIdx,
    exp_f: f32,
) std.mem.Allocator.Error!?ast.NodeIdx {
    if (exp_f < 0.0) {
        const pos = (try emitPositiveIntPow(allocator, nodes, base, -exp_f)) orelse return null;
        const one: ast.NodeIdx = @intCast(nodes.items.len);
        try nodes.append(allocator, .{ .const_f32 = 1.0 });
        const div: ast.NodeIdx = @intCast(nodes.items.len);
        try nodes.append(allocator, .{ .binary = .{ .op = .div, .a = one, .b = pos } });
        return div;
    }
    return try emitPositiveIntPow(allocator, nodes, base, exp_f);
}

fn emitPositiveIntPow(
    allocator: std.mem.Allocator,
    nodes: *std.ArrayList(ast.Node),
    base: ast.NodeIdx,
    exp_f: f32,
) std.mem.Allocator.Error!?ast.NodeIdx {
    if (exp_f == 0.0) {
        const idx: ast.NodeIdx = @intCast(nodes.items.len);
        try nodes.append(allocator, .{ .const_f32 = 1.0 });
        return idx;
    }
    if (exp_f == 1.0) return base;

    // Positive integer N <= 64: iterative balanced split x^ceil(N/2) * x^floor(N/2) with local memo.
    if (exp_f <= 64.0) {
        const n: u32 = @intFromFloat(exp_f);
        var memo: [65]?ast.NodeIdx = @splat(null);
        memo[1] = base;

        // Iterative work stack (max depth for N <= 64 is 7).
        var work: [32]u32 = undefined;
        var sp: usize = 0;
        work[sp] = n;
        sp += 1;

        while (sp > 0) {
            const cur = work[sp - 1];
            if (memo[cur] != null) {
                sp -= 1;
                continue;
            }
            const hi = (cur + 1) / 2;
            const lo = cur / 2;
            if (memo[hi] == null) {
                work[sp] = hi;
                sp += 1;
                continue;
            }
            if (memo[lo] == null) {
                work[sp] = lo;
                sp += 1;
                continue;
            }
            const mul_idx: ast.NodeIdx = @intCast(nodes.items.len);
            try nodes.append(allocator, .{ .binary = .{ .op = .mul, .a = memo[hi].?, .b = memo[lo].? } });
            memo[cur] = mul_idx;
            sp -= 1;
        }
        return memo[n].?;
    }

    // Integer N > 64: odd N -> x * (abs(x) ** (N - 1)) / x * (x ** (N - 1)), even N -> abs(x) ** N.
    if (isEvenInteger(exp_f)) {
        const abs_idx: ast.NodeIdx = @intCast(nodes.items.len);
        try nodes.append(allocator, .{ .unary = .{ .op = .abs, .a = base } });
        const exp_idx: ast.NodeIdx = @intCast(nodes.items.len);
        try nodes.append(allocator, .{ .const_f32 = exp_f });
        const pow_idx: ast.NodeIdx = @intCast(nodes.items.len);
        try nodes.append(allocator, .{ .binary = .{ .op = .pow, .a = abs_idx, .b = exp_idx } });
        return pow_idx;
    } else {
        // V8: Odd integer N > 64 -> `x * (x ** (N - 1))`, expanding `x ** 64` for N == 65
        // and `abs(x) ** (N - 1)` for larger odd N (matches std.Expr).
        const even_pow = (try emitPositiveIntPow(allocator, nodes, base, exp_f - 1.0)) orelse return null;
        const mul_idx: ast.NodeIdx = @intCast(nodes.items.len);
        try nodes.append(allocator, .{ .binary = .{ .op = .mul, .a = base, .b = even_pow } });
        return mul_idx;
    }
}

test "D1-D15 mandatory lowering and oracle contracts" {
    const parser = @import("parser.zig");
    const allocator = std.testing.allocator;
    const x86: interp.TargetProfile = .{ .use_fma = false, .isa_family = .x86 };

    const evalConst = struct {
        fn run(alloc: std.mem.Allocator, expr: []const u8, prof: interp.TargetProfile) !f32 {
            var p = try parser.parse(alloc, expr, .{ .num_clips = 1 });
            defer p.deinit(alloc);
            try lower(alloc, &p, prof);
            return try interp.evalScalar(alloc, &p, prof, .{ .width = 16, .height = 16, .frame_n = 0, .props = &.{} });
        }
    }.run;

    // D1: 023 == 23, 0xFFFFFFFF == -1, opt=0 evaluates in f32 (16777217 1 + 16777216 - == 0).
    try std.testing.expectEqual(@as(f32, 23.0), try evalConst(allocator, "023", x86));
    try std.testing.expectEqual(@as(f32, -1.0), try evalConst(allocator, "0xFFFFFFFF", x86));
    try std.testing.expectEqual(@as(f32, 0.0), try evalConst(allocator, "16777217 1 + 16777216 -", x86));

    // D2: constant `-1 sqrt` folds via libm sqrtf -> NaN.
    try std.testing.expect(std.math.isNan(try evalConst(allocator, "-1 sqrt", x86)));

    // D3: unfused `%` constant folding matches runtime (`0.7 0.1 % == 0.0`, `2.0 0.2 % == 0.0`, `1e10 3 % == 0.0`, `-0 3 % == +0.0`).
    try std.testing.expectEqual(@as(f32, 0.0), try evalConst(allocator, "0.7 0.1 %", x86));
    try std.testing.expectEqual(@as(f32, 0.0), try evalConst(allocator, "2.0 0.2 %", x86));
    try std.testing.expectEqual(@as(f32, 0.0), try evalConst(allocator, "1e10 3 %", x86));
    try std.testing.expectEqual(@as(u32, 0x0000_0000), @as(u32, @bitCast(try evalConst(allocator, "-0 3 %", x86))));

    // D5: all-constant pow folds via libm powf (`1.1 5 pow == 1.6105102`, `0.1 0.2 + 7 pow`, `0 -1 pow == +inf`, `0 0 pow == 1`).
    try std.testing.expectEqual(powf(1.1, 5.0), try evalConst(allocator, "1.1 5 pow", x86));
    try std.testing.expectEqual(powf(@as(f32, 0.1) + @as(f32, 0.2), 7.0), try evalConst(allocator, "0.1 0.2 + 7 pow", x86));
    try std.testing.expectEqual(std.math.inf(f32), try evalConst(allocator, "0 -1 pow", x86));
    try std.testing.expectEqual(@as(f32, 1.0), try evalConst(allocator, "0 0 pow", x86));

    // Non-powf libm constant folds match process libm (portable across glibc & musl, U4).
    // Prevent LLVM from constant-folding the reference call in ReleaseFast with a different libm model.
    var v_exp: f32 = 1.25;
    var v_log: f32 = 0.0;
    var v_sin: f32 = 0.75;
    std.mem.doNotOptimizeAway(&v_exp);
    std.mem.doNotOptimizeAway(&v_log);
    std.mem.doNotOptimizeAway(&v_sin);
    try std.testing.expectEqual(callHostLibm1("expf", &libm_cache.expf, &expf, v_exp), try evalConst(allocator, "1.25 exp", x86));
    try std.testing.expectEqual(callHostLibm1("logf", &libm_cache.logf, &logf, v_log), try evalConst(allocator, "0 log", x86));
    try std.testing.expectEqual(callHostLibm1("sinf", &libm_cache.sinf, &sinf, v_sin), try evalConst(allocator, "0.75 sin", x86));
    try std.testing.expectEqual(@as(f32, 1.0), try evalConst(allocator, "48 257 * 167 257 * 42919 linstep", x86));
    try std.testing.expectEqual(interp.evalUnaryF32(x86, .rsqrt, 0.0), try evalConst(allocator, "0 rsqrt", x86));
    try std.testing.expectEqual(interp.evalUnaryF32(x86, .rsqrt, std.math.inf(f32)), try evalConst(allocator, "1 0 / rsqrt", x86));

    // D9: constant fold `2.5 round == 2` (NOT libm roundf -> 3, Y9).
    try std.testing.expectEqual(@as(f32, 2.0), try evalConst(allocator, "2.5 round", x86));

    // D12: constant `0 0 / not == 0.0` (ordered v <= 0.0).
    try std.testing.expectEqual(@as(f32, 0.0), try evalConst(allocator, "0 0 / not", x86));

    // Runtime D5 tests on a 1x1 f32 plane:
    const evalOnePixel = struct {
        fn run(alloc: std.mem.Allocator, expr: []const u8, x_val: f32, prof: interp.TargetProfile) !f32 {
            var p = try parser.parse(alloc, expr, .{ .num_clips = 1 });
            defer p.deinit(alloc);
            try lower(alloc, &p, prof);
            var src = [1]f32{x_val};
            var dst = [1]f32{0.0};
            const srcs = [_][*]const u8{@ptrCast(&src)};
            const strides = [_]isize{4};
            const fmts = [_]ast.SampleFormat{.{ .is_float = true, .bits_per_sample = 32 }};
            try interp.evalPlane(alloc, &p, prof, .{
                .dst = @ptrCast(&dst),
                .dst_stride = 4,
                .dst_fmt = fmts[0],
                .srcs = &srcs,
                .src_strides = &strides,
                .src_fmts = &fmts,
                .width = 1,
                .height = 1,
                .frame_n = 0,
                .props = &.{},
            });
            return dst[0];
        }
    }.run;

    // T3: chained power simplifies before expansion: `x 1e20 * 2 pow 0.5 pow` on x = -0.25 -> 2.5e19 (no overflow!).
    try std.testing.expectEqual(@as(f32, 2.5e19), try evalOnePixel(allocator, "x 1e20 * 2 pow 0.5 pow", -0.25, x86));
    // Q1: runtime `0 x -1 * pow == 0`, `1 x 0 0 / + pow == 1`.
    try std.testing.expectEqual(@as(f32, 0.0), try evalOnePixel(allocator, "0 x -1 * pow", 2.0, x86));
    try std.testing.expectEqual(@as(f32, 1.0), try evalOnePixel(allocator, "1 x 0 0 / + pow", 2.0, x86));
    // Negative-base integer powers (N <= 64 and N > 64).
    try std.testing.expectEqual(@as(f32, -8.0), try evalOnePixel(allocator, "x 3 pow", -2.0, x86));
    try std.testing.expectEqual(@as(f32, 16.0), try evalOnePixel(allocator, "x 4 pow", -2.0, x86));
    try std.testing.expectEqual(@as(f32, -1.0), try evalOnePixel(allocator, "x 65 pow", -1.0, x86));
    try std.testing.expectEqual(@as(f32, 1.0), try evalOnePixel(allocator, "x 66 pow", -1.0, x86));
    // Runtime D2/D12/D13/D15 twins.
    try std.testing.expectEqual(@as(f32, 0.0), try evalOnePixel(allocator, "x sqrt", -1.0, x86));
    try std.testing.expectEqual(@as(f32, 0.0), try evalOnePixel(allocator, "x not", std.math.nan(f32), x86));
    try std.testing.expect(@abs((try evalOnePixel(allocator, "x log", 0.0, x86)) - (-87.33654)) < 1e-3);
    try std.testing.expectEqual(@as(f32, 0.0), try evalOnePixel(allocator, "x 0 >", std.math.nan(f32), x86));
    try std.testing.expectEqual(@as(f32, 42.0), try evalOnePixel(allocator, "x 10 42 ?", std.math.nan(f32), x86));

    // E2/A1: runtime-x Phase B expansion hits exact endpoints for constant-expression bounds,
    // and all-constant linstep/smoothstep folds match runtime bit-for-bit.
    try std.testing.expectEqual(@as(f32, 0.0), try evalOnePixel(allocator, "48 257 * 167 257 * x linstep", 12336.0, x86));
    try std.testing.expectEqual(@as(f32, 1.0), try evalOnePixel(allocator, "48 257 * 167 257 * x linstep", 42919.0, x86));
    try std.testing.expectEqual(
        try evalOnePixel(allocator, "77 199 x linstep", 124.0, x86),
        try evalConst(allocator, "77 199 124 linstep", x86),
    );
    try std.testing.expectEqual(
        try evalOnePixel(allocator, "241 81 x smoothstep", 199.0, x86),
        try evalConst(allocator, "241 81 199 smoothstep", x86),
    );
    try std.testing.expectEqual(
        try evalOnePixel(allocator, "0.2 0.7 x linstep", 0.5, x86),
        try evalConst(allocator, "0.2 0.7 0.5 linstep", x86),
    );
}
