//! Dimension-specialized SIMD JIT compiler for plane expressions,
//! including B1/D6 compile-time row-split relative access, D7/U6 dynamic gather,
//! `f16` (F16C / baseline NEON), and `opt=1` `i32` evaluation mode.

const std = @import("std");
const ast = @import("../expr/ast.zig");
const analysis_mod = @import("../expr/analysis.zig");
const interp = @import("../expr/interp.zig");
const aj = @import("asmjit.zig");
const cpu_mod = @import("cpu.zig");
const pixel = @import("pixel.zig");
const math_jit = @import("math.zig");
const int_path = @import("int_path.zig");

pub const CompileSpec = struct {
    width: usize,
    height: usize,
    src_fmts: []const ast.SampleFormat,
    dst_fmt: ast.SampleFormat,
    capture_disasm: bool = false,
};

pub const CompileResult = struct {
    kernel: aj.CompiledKernel,
    disasm: ?[]u8 = null,

    pub fn deinit(self: *CompileResult, allocator: std.mem.Allocator, rt: *aj.Runtime) void {
        rt.release(self.kernel.func);
        if (self.disasm) |d| allocator.free(d);
        self.* = undefined;
    }
};

pub const CompileError = aj.JitError || std.mem.Allocator.Error || error{
    UnsupportedFeature,
};

const RelRowKey = struct {
    clip: ast.ClipId,
    dy: i32,
    mode: ast.BoundaryMode,
};

const XMode = union(enum) {
    /// Compile-time constant column `col` (left edge / right edge + tail using B1 window loads).
    const_col: usize,
    /// Branchless interior vector loop (all relative horizontal accesses are in bounds).
    interior,
    /// Fallback runtime loop for huge left-edge `|dx|`; holds `left_end` so `eff_dx >= 0` loads directly.
    dynamic_left: usize,
    /// Fallback runtime loop for huge right-edge `|dx|`; holds `right_start` so `eff_dx <= 0` loads directly.
    dynamic_right: usize,
};

const Invariance = enum(u8) {
    frame = 0,
    row = 1,
    pixel = 2,
};

/// True iff `c` is a finite normal power of two whose reciprocal `1/c` is also normal
/// (so `a / c == a * (1 / c)` is bit-exact for every float32 value).
fn isExactPow2Divisor(c: f32) bool {
    const bits: u32 = @bitCast(c);
    const mant = bits & 0x007F_FFFF;
    const exp = (bits >> 23) & 0xFF;
    return mant == 0 and exp >= 2 and exp <= 253;
}

/// Returns integer period `C` iff `node` is `X % C` where `C` is a positive integer dividing `lanes`
/// and `width <= 2^24` (so `X % C` is bit-identical to the constant lane vector `[0%C, 1%C, ...]`).
fn periodicCoordXMod(
    prog: *const ast.Program,
    const_f32_vals: []const ?f32,
    node: ast.Node,
    width: usize,
    lanes: usize,
) ?usize {
    if (prog.opt_int or width > (1 << 24)) return null;
    if (node != .binary or node.binary.op != .mod) return null;
    if (prog.nodes.items[node.binary.a] != .coord_x) return null;
    const c_val: f32 = if (const_f32_vals.len == prog.nodes.items.len and const_f32_vals[node.binary.b] != null)
        const_f32_vals[node.binary.b].?
    else switch (prog.nodes.items[node.binary.b]) {
        .const_f32 => |v| v,
        .const_i32 => |v| @floatFromInt(v),
        else => return null,
    };
    if (!std.math.isFinite(c_val) or c_val < 1.0 or c_val > @as(f32, @floatFromInt(lanes)) or @trunc(c_val) != c_val) return null;
    const ci: usize = @intFromFloat(c_val);
    if (ci == 0 or (lanes % ci) != 0) return null;
    return ci;
}

fn computeInvariance(
    allocator: std.mem.Allocator,
    prog: *const ast.Program,
    info: *const analysis_mod.Analysis,
    const_f32_vals: []const ?f32,
    width: usize,
    lanes: usize,
) std.mem.Allocator.Error![]Invariance {
    const n = prog.nodes.items.len;
    const inv = try allocator.alloc(Invariance, n);
    @memset(inv, .pixel);
    for (prog.nodes.items, 0..) |node, idx| {
        if (idx != prog.root and info.use_counts.items[idx] == 0) continue;
        inv[idx] = switch (node) {
            .const_f32, .const_i32, .frame_n, .load_prop, .plane_width, .plane_height => .frame,
            .coord_y => .row,
            .coord_x, .load_clip, .load_rel, .arg_extrema, .arg_sort_elem => .pixel,
            .load_abs => |la| @fromBackingInt(@max(@backingInt(inv[la.x]), @backingInt(inv[la.y]))),
            .unary => |u| inv[u.a],
            .binary => |b| if (periodicCoordXMod(prog, const_f32_vals, node, width, lanes) != null)
                .frame
            else
                @fromBackingInt(@max(@backingInt(inv[b.a]), @backingInt(inv[b.b]))),
            .ternary => |t| @fromBackingInt(@max(@max(@backingInt(inv[t.a]), @backingInt(inv[t.b])), @backingInt(inv[t.c]))),
        };
    }
    return inv;
}

/// Fold pure compile-time `.frame` nodes (built only from literals and `plane_width`/`plane_height`,
/// never `N` or frame props) using the exact target profile so e.g. `width 2 /` and `height 2 /`
/// become compile-time `f32` constants (and RIP-relative const-memory operands on high-pressure loops).
fn computeCompileTimeF32Consts(
    allocator: std.mem.Allocator,
    prog: *const ast.Program,
    info: *const analysis_mod.Analysis,
    spec: CompileSpec,
    profile: interp.TargetProfile,
) std.mem.Allocator.Error![]?f32 {
    const n = prog.nodes.items.len;
    const vals = try allocator.alloc(?f32, n);
    @memset(vals, null);
    if (prog.opt_int) return vals;

    for (prog.nodes.items, 0..) |node, idx| {
        if (idx != prog.root and info.use_counts.items[idx] == 0) continue;
        if (info.value_types.items[idx] != .f32) continue;
        vals[idx] = switch (node) {
            .const_f32 => |v| v,
            .const_i32 => |v| @floatFromInt(v),
            .plane_width => @floatFromInt(spec.width),
            .plane_height => @floatFromInt(spec.height),
            .unary => |u| blk: {
                const va = vals[u.a] orelse break :blk null;
                const r = interp.evalUnaryF32(profile, u.op, va);
                break :blk if (std.math.isFinite(r)) r else null;
            },
            .binary => |b| blk: {
                const va = vals[b.a] orelse break :blk null;
                const vb = vals[b.b] orelse break :blk null;
                const r = interp.evalBinaryF32(profile, b.op, va, vb);
                break :blk if (std.math.isFinite(r)) r else null;
            },
            .ternary => |t| blk: {
                const va = vals[t.a] orelse break :blk null;
                const vb = vals[t.b] orelse break :blk null;
                const vc = vals[t.c] orelse break :blk null;
                const r = interp.evalTernaryF32(profile, t.op, va, vb, vc);
                break :blk if (std.math.isFinite(r)) r else null;
            },
            else => null,
        };
    }
    return vals;
}

const RpnStats = struct {
    peak_live: u32 = 0,
    has_transcendental: bool = false,
};

/// Estimate peak simultaneous live `.pixel` values in RPN order (excluding hoisted/const nodes)
/// to decide whether a long DAG should use Sethi-Ullman reordering (`peak_live > 10`) or
/// lockstep 2x-interleaved evaluation (`peak_live <= 6` on 16-reg AVX2, `<= 8` on 32-reg ISAs).
fn estimateRpnStats(
    allocator: std.mem.Allocator,
    prog: *const ast.Program,
    info: *const analysis_mod.Analysis,
    node_inv: []const Invariance,
    const_f32_vals: []const ?f32,
) std.mem.Allocator.Error!RpnStats {
    const n_nodes = prog.nodes.items.len;
    if (prog.root >= n_nodes) return .{};

    const rem = try allocator.alloc(u32, n_nodes);
    defer allocator.free(rem);
    @memset(rem, 0);
    rem[prog.root] = 1;

    var ri: usize = prog.root + 1;
    while (ri > 0) {
        ri -= 1;
        if (rem[ri] == 0) continue;
        if (node_inv[ri] != .pixel or const_f32_vals[ri] != null) continue;
        switch (prog.nodes.items[ri]) {
            .load_abs => |la| {
                rem[la.x] += 1;
                rem[la.y] += 1;
            },
            .unary => |u| rem[u.a] += 1,
            .binary => |bin| {
                rem[bin.a] += 1;
                rem[bin.b] += 1;
            },
            .ternary => |t| {
                rem[t.a] += 1;
                rem[t.b] += 1;
                rem[t.c] += 1;
            },
            else => {},
        }
    }

    var live_now: u32 = 0;
    var peak: u32 = 0;
    var has_trans = false;
    const isTracked = struct {
        fn run(p: *const ast.Program, inv: []const Invariance, cf: []const ?f32, i: ast.NodeIdx) bool {
            if (inv[i] != .pixel or cf[i] != null) return false;
            return switch (p.nodes.items[i]) {
                .const_f32, .const_i32, .coord_x, .coord_y, .frame_n, .load_prop, .plane_width, .plane_height => false,
                else => true,
            };
        }
    }.run;

    for (prog.nodes.items[0 .. prog.root + 1], 0..) |node, i| {
        const idx: ast.NodeIdx = @intCast(i);
        if (rem[idx] == 0 or !isTracked(prog, node_inv, const_f32_vals, idx)) continue;
        switch (node) {
            .unary => |u| switch (u.op) {
                .exp, .exp2, .log, .log2, .sin, .cos, .tan, .atan, .asin, .acos => has_trans = true,
                else => {},
            },
            .binary => |bin| switch (bin.op) {
                .pow, .atan2 => has_trans = true,
                else => {},
            },
            else => {},
        }
        live_now += 1;
        if (live_now > peak) peak = live_now;
        switch (node) {
            .load_abs => |la| {
                for ([_]ast.NodeIdx{ la.x, la.y }) |c| {
                    if (isTracked(prog, node_inv, const_f32_vals, c) and rem[c] > 0) {
                        rem[c] -= 1;
                        if (rem[c] == 0 and live_now > 0) live_now -= 1;
                    }
                }
            },
            .unary => |u| {
                if (isTracked(prog, node_inv, const_f32_vals, u.a) and rem[u.a] > 0) {
                    rem[u.a] -= 1;
                    if (rem[u.a] == 0 and live_now > 0) live_now -= 1;
                }
            },
            .binary => |bin| {
                for ([_]ast.NodeIdx{ bin.a, bin.b }) |c| {
                    if (isTracked(prog, node_inv, const_f32_vals, c) and rem[c] > 0) {
                        rem[c] -= 1;
                        if (rem[c] == 0 and live_now > 0) live_now -= 1;
                    }
                }
            },
            .ternary => |t| {
                for ([_]ast.NodeIdx{ t.a, t.b, t.c }) |c| {
                    if (isTracked(prog, node_inv, const_f32_vals, c) and rem[c] > 0) {
                        rem[c] -= 1;
                        if (rem[c] == 0 and live_now > 0) live_now -= 1;
                    }
                }
            },
            else => {},
        }
    }
    _ = info;
    return .{ .peak_live = peak, .has_transcendental = has_trans };
}

/// Compile a lowered/optimized plane expression into a native SIMD kernel.
pub fn compilePlane(
    allocator: std.mem.Allocator,
    rt: *aj.Runtime,
    cpu: cpu_mod.CpuInfo,
    prog: *const ast.Program,
    info: *const analysis_mod.Analysis,
    spec: CompileSpec,
) CompileError!CompileResult {
    if (cpu.isa == .scalar) return error.UnsupportedFeature;
    if (info.uses_f16 and !cpu.has_f16c and cpu.isa != .neon) return error.UnsupportedFeature;

    var b = try aj.Builder.init(rt, cpu, spec.capture_disasm);
    defer b.deinit();

    try b.beginFunc();

    if (spec.height == 0 or spec.width == 0) {
        b.retVoid();
        b.endFunc();
        return try finishCompile(allocator, rt, &b, spec.capture_disasm);
    }

    // Kernel arguments: (dst, dst_stride, srcs, src_strides, frame_n, props)
    const dst_ptr = b.argGp(0);
    const dst_stride = b.argGp(1);
    const srcs_ptr = b.argGp(2);
    const src_strides_ptr = b.argGp(3);
    const frame_n_gp = b.argGp(4);
    const props_ptr = b.argGp(5);

    const n_clips = spec.src_fmts.len;
    const src_row_ptrs = try allocator.alloc(?aj.Reg, n_clips);
    defer allocator.free(src_row_ptrs);
    @memset(src_row_ptrs, null);

    const base_src_ptrs = try allocator.alloc(?aj.Reg, n_clips);
    defer allocator.free(base_src_ptrs);
    @memset(base_src_ptrs, null);

    const src_stride_regs = try allocator.alloc(?aj.Reg, n_clips);
    defer allocator.free(src_stride_regs);
    @memset(src_stride_regs, null);

    const src_stride_vecs = try allocator.alloc(?aj.Reg, n_clips);
    defer allocator.free(src_stride_vecs);
    @memset(src_stride_vecs, null);

    // Collect unique (clip, dy != 0, mode) row keys and clips needing base_src_ptr.
    var rel_row_keys: std.ArrayList(RelRowKey) = .empty;
    defer rel_row_keys.deinit(allocator);

    const need_base = try allocator.alloc(bool, n_clips);
    defer allocator.free(need_base);
    @memset(need_base, false);

    const need_stride_vec = try allocator.alloc(bool, n_clips);
    defer allocator.free(need_stride_vec);
    @memset(need_stride_vec, false);

    for (prog.nodes.items, 0..) |node, idx| {
        if (idx != prog.root and info.use_counts.items[idx] == 0) continue;
        switch (node) {
            .load_rel => |rel| {
                if (rel.dy != 0) {
                    need_base[rel.clip] = true;
                    const key: RelRowKey = .{ .clip = rel.clip, .dy = rel.dy, .mode = rel.mode };
                    var found = false;
                    for (rel_row_keys.items) |k| {
                        if (k.clip == key.clip and k.dy == key.dy and k.mode == key.mode) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) try rel_row_keys.append(allocator, key);
                }
            },
            .load_abs => |la| {
                need_base[la.clip] = true;
                need_stride_vec[la.clip] = true;
            },
            else => {},
        }
    }

    // Hoist per-frame clip row pointers, base pointers & strides above the y loop.
    for (info.used_clips.items) |clip| {
        const ptr_reg = b.newGp64();
        const stride_reg = b.newGp64();
        const byte_off: i32 = @intCast(@as(usize, clip) * @sizeOf(usize));
        b.gpLoad(ptr_reg, aj.Mem.ptr(srcs_ptr, byte_off), 8, false);
        b.gpLoad(stride_reg, aj.Mem.ptr(src_strides_ptr, byte_off), 8, false);
        src_row_ptrs[clip] = ptr_reg;
        src_stride_regs[clip] = stride_reg;
        if (need_base[clip]) {
            const base_reg = b.newGp64();
            b.gpMov(base_reg, ptr_reg);
            base_src_ptrs[clip] = base_reg;
        }
        if (need_stride_vec[clip]) {
            const sv = b.newVec();
            b.vecBroadcastGp32(sv, stride_reg);
            src_stride_vecs[clip] = sv;
        }
    }

    // Hoist per-frame invariants (`N`, frame properties, lane offsets for `X`).
    var v_frame_n_f32: ?aj.Reg = null;
    var v_frame_n_i32: ?aj.Reg = null;
    if (info.uses_n) {
        const tmp_i = b.newVec();
        b.vecBroadcastGp32(tmp_i, frame_n_gp);
        if (prog.opt_int) {
            v_frame_n_i32 = tmp_i;
        } else {
            b.vecOp2(.cvt_i32_to_f32, tmp_i, tmp_i);
            v_frame_n_f32 = tmp_i;
        }
    }

    const prop_regs = try allocator.alloc(aj.Reg, prog.props.items.len);
    defer allocator.free(prop_regs);
    for (prog.props.items, 0..) |_, pid| {
        const preg = b.newVec();
        b.vecLoad(.broadcast_u32, preg, aj.Mem.ptr(props_ptr, @intCast(pid * 4)));
        prop_regs[pid] = preg;
    }

    // Phase 3 (LICM): fold compile-time width/height expressions, classify node invariance
    // (including periodic `X % C` where `C | f32_lanes`, or `C | (2 * f32_lanes)` for 2-phase hoisting),
    // and hoist .frame/.row nodes.
    const lanes: usize = cpu.f32_lanes;
    const has_horiz_rel = info.max_left > 0 or info.max_right > 0;
    b.uint_f32_bias_loads = cpu.prefer_alt_widen_load and info.live_node_count <= neon_bias_load_max_nodes;
    const use_u8_interior = info.u8_plan.eligible and !info.uses_abs_access;
    const use_u16_interior = !use_u8_interior and info.u16_plan.eligible and !info.uses_abs_access;
    const use_int16_interior = !use_u8_interior and !use_u16_interior and info.int16_plan.eligible and !info.uses_abs_access;
    const use_int32_interior = !use_u8_interior and !use_u16_interior and !use_int16_interior and info.int32_plan.eligible and !info.uses_abs_access;
    const base_unroll_x2 = !use_u8_interior and !use_u16_interior and !use_int16_interior and !use_int32_interior and !info.uses_abs_access and info.live_node_count <= 40 and info.expr_cost <= 64;
    const enable_licm = !info.u8_plan.eligible and !info.u16_plan.eligible and !info.int16_plan.eligible and !info.int32_plan.eligible;

    const profile = interp.TargetProfile.fromCpu(cpu);
    const const_f32_vals = try computeCompileTimeF32Consts(allocator, prog, info, spec, profile);
    defer allocator.free(const_f32_vals);

    var has_2x_periodic = false;
    if (enable_licm and !has_horiz_rel and !prog.opt_int and !info.uses_abs_access and spec.width <= (1 << 24)) {
        for (prog.nodes.items, 0..) |node, idx| {
            if (idx != prog.root and info.use_counts.items[idx] == 0) continue;
            if (periodicCoordXMod(prog, const_f32_vals, node, spec.width, lanes) == null and
                periodicCoordXMod(prog, const_f32_vals, node, spec.width, lanes * 2) != null)
            {
                has_2x_periodic = true;
                break;
            }
        }
    }

    var node_inv = try computeInvariance(allocator, prog, info, const_f32_vals, spec.width, if (has_2x_periodic) lanes * 2 else lanes);
    defer allocator.free(node_inv);

    const rpn_stats = try estimateRpnStats(allocator, prog, info, node_inv, const_f32_vals);
    const max_interleave_live: u32 = if (cpu.has_avx512 or cpu.isa == .neon) 8 else 6;
    const can_interleave_x2 = !use_u8_interior and !use_u16_interior and !use_int16_interior and !use_int32_interior and
        !info.uses_abs_access and !has_horiz_rel and !prog.opt_int and !rpn_stats.has_transcendental and
        info.live_node_count > 40 and rpn_stats.peak_live <= max_interleave_live;
    // On 32-register ISAs (AArch64 NEON and AVX-512), narrow serial chains (e.g. Mandelbrot,
    // peak_live <= 5) can interleave 4 independent vectors with zero spills.
    const can_interleave_x4 = cpu.has32VecRegs() and can_interleave_x2 and rpn_stats.peak_live <= 5 and !has_2x_periodic;
    const can_unroll_x2 = base_unroll_x2 or can_interleave_x2;

    if (has_2x_periodic and node_inv[prog.root] == .pixel and !can_unroll_x2) {
        has_2x_periodic = false;
        allocator.free(node_inv);
        node_inv = try computeInvariance(allocator, prog, info, const_f32_vals, spec.width, lanes);
    }
    const periodic_lanes: usize = if (has_2x_periodic) lanes * 2 else lanes;

    const n_nodes = prog.nodes.items.len;
    const hoisted_slots = try allocator.alloc(?RegSlot, n_nodes);
    defer allocator.free(hoisted_slots);
    @memset(hoisted_slots, null);

    const hoisted_slots_hi = try allocator.alloc(?RegSlot, if (has_2x_periodic) n_nodes else 0);
    defer allocator.free(hoisted_slots_hi);
    if (has_2x_periodic) @memset(hoisted_slots_hi, null);

    // Precompute frame-invariant row pointers for contiguous `X const_y x[]` absolute loads.
    const abs_row_ptrs = try allocator.alloc(?aj.Reg, if (info.uses_abs_access and spec.width <= (1 << 24)) n_nodes else 0);
    defer allocator.free(abs_row_ptrs);
    if (abs_row_ptrs.len == n_nodes) {
        @memset(abs_row_ptrs, null);
        for (prog.nodes.items, 0..) |node, idx| {
            if (idx != prog.root and info.use_counts.items[idx] == 0) continue;
            if (node != .load_abs) continue;
            const la = node.load_abs;
            if (prog.nodes.items[la.x] != .coord_x) continue;
            if (const_f32_vals[la.y]) |cy| {
                const sy = interp.absCoordToIndex(profile, cy, @intCast(spec.height), la.mode);
                const rptr = b.newGp64();
                if (sy == 0) {
                    b.gpMov(rptr, base_src_ptrs[la.clip].?);
                } else {
                    b.gpMovImm(rptr, @intCast(sy));
                    b.gpOp3(.mul, rptr, rptr, src_stride_regs[la.clip].?);
                    b.gpOp3(.add, rptr, rptr, base_src_ptrs[la.clip].?);
                }
                abs_row_ptrs[idx] = rptr;
            }
        }
    }

    // Track `.frame`/`.row` nodes consumed as boolean conditions by `.pixel` select/logic trees
    // so we hoist their `val > 0` mask once per frame/row instead of re-emitting `vcmpps` per pixel.
    const need_hoisted_mask = try allocator.alloc(bool, n_nodes);
    defer allocator.free(need_hoisted_mask);
    @memset(need_hoisted_mask, false);
    if (enable_licm and !prog.opt_int) {
        var mask_work: std.ArrayList(ast.NodeIdx) = .empty;
        defer mask_work.deinit(allocator);
        for (prog.nodes.items, 0..) |node, idx| {
            if (idx != prog.root and info.use_counts.items[idx] == 0) continue;
            if (node_inv[idx] != .pixel) continue;
            if (node == .ternary and node.ternary.op == .select and info.value_types.items[idx] == .f32) {
                try mask_work.append(allocator, node.ternary.a);
                while (mask_work.items.len > 0) {
                    const ci = mask_work.pop().?;
                    if (node_inv[ci] != .pixel) {
                        if (const_f32_vals[ci] == null) need_hoisted_mask[ci] = true;
                        continue;
                    }
                    if (info.use_counts.items[ci] != 1) continue;
                    switch (prog.nodes.items[ci]) {
                        .binary => |bin| switch (bin.op) {
                            .and_op, .or_op, .xor_op => {
                                try mask_work.append(allocator, bin.a);
                                try mask_work.append(allocator, bin.b);
                            },
                            else => {},
                        },
                        else => {},
                    }
                }
            }
        }
    }

    const hoisted_masks = try allocator.alloc(?aj.Reg, n_nodes);
    defer allocator.free(hoisted_masks);
    @memset(hoisted_masks, null);

    const hoisted_masks_hi = try allocator.alloc(?aj.Reg, if (has_2x_periodic) n_nodes else 0);
    defer allocator.free(hoisted_masks_hi);
    if (has_2x_periodic) @memset(hoisted_masks_hi, null);

    if (enable_licm) {
        const frame_env = BodyEnv{
            .dst_ptr = dst_ptr,
            .x_gp = undefined,
            .src_row_ptrs = src_row_ptrs,
            .base_src_ptrs = base_src_ptrs,
            .src_stride_vecs = src_stride_vecs,
            .rel_row_keys = rel_row_keys.items,
            .rel_row_ptrs = &.{},
            .prop_regs = prop_regs,
            .v_x = null,
            .v_x_i32 = null,
            .v_y = null,
            .v_frame_n_f32 = v_frame_n_f32,
            .v_frame_n_i32 = v_frame_n_i32,
            .node_inv = node_inv,
            .hoisted_slots = hoisted_slots,
            .hoisted_masks = hoisted_masks,
            .const_f32_vals = const_f32_vals,
            .abs_row_ptrs = abs_row_ptrs,
        };
        try evalHoistedStage(allocator, &b, prog, info, spec, frame_env, node_inv, hoisted_slots, hoisted_masks, need_hoisted_mask, .frame, periodic_lanes, 0);
        if (has_2x_periodic) {
            var frame_env_hi = frame_env;
            frame_env_hi.hoisted_slots = hoisted_slots_hi;
            frame_env_hi.hoisted_masks = hoisted_masks_hi;
            try evalHoistedStage(allocator, &b, prog, info, spec, frame_env_hi, node_inv, hoisted_slots_hi, hoisted_masks_hi, need_hoisted_mask, .frame, periodic_lanes, lanes);
        }
    }

    const interior_step: usize = if (use_u8_interior)
        cpu.vec_bytes
    else if (use_u16_interior or use_int16_interior)
        lanes * 2
    else
        lanes;

    // Decide compile-time row-split boundaries (B1 / S5 / Y2: sized for the interior loop's OWN pixel step!).
    // G3: align `u16` interior start to `interior_step` (32B on AVX2 / 64B on AVX-512) when an interior loop exists.
    const right_margin: usize = (interior_step - 1) + @as(usize, @intCast(info.max_right));
    const aligned_left_u16: usize = std.mem.alignForward(usize, @intCast(info.max_left), interior_step);
    const left_align: usize = if (use_u16_interior and spec.width > aligned_left_u16 + right_margin)
        interior_step
    else
        lanes;
    const left_end: usize = if (has_horiz_rel)
        @min(std.mem.alignForward(usize, @intCast(info.max_left), left_align), spec.width)
    else
        0;
    const interior_end: usize = if (has_horiz_rel and spec.width > right_margin)
        @max(left_end, spec.width - right_margin)
    else
        left_end;
    const interior_iters: usize = if (interior_end > left_end)
        (interior_end - left_end + interior_step - 1) / interior_step
    else
        0;
    const right_start: usize = left_end + interior_iters * interior_step;
    const left_vecs: usize = left_end / lanes;
    const right_vecs: usize = if (has_horiz_rel and spec.width > right_start)
        (spec.width - right_start + lanes - 1) / lanes
    else
        0;
    // V12: only the oversized edge strips use a compact `.dynamic_left`/`.dynamic_right` loop;
    // the interior `[left_end, right_start)` ALWAYS keeps the branchless u8 / i16x2 / f32x2 fast path!
    const left_use_dynamic = has_horiz_rel and left_vecs > 8;
    const right_use_dynamic = has_horiz_rel and right_vecs > 8;
    const can_unroll_u8_x2 = use_u8_interior and info.live_node_count <= 40;
    const can_unroll_u16_x2 = use_u16_interior and info.live_node_count <= 40;
    const can_unroll_i16_x2 = use_int16_interior and (info.int16_plan.final_scale_op == .none or cpu.isa == .neon) and info.live_node_count <= 48;
    const can_unroll_i32_x2 = use_int32_interior and info.live_node_count <= 48;
    // Short transcendental chains (exp/log/pow polynomials) are latency-bound: interleave the two
    // unrolled vectors on AVX2 too, up to peak_live 8 even on 16 YMM registers (Zen 3 silicon:
    // guided_a +10%, adg_mask +10..18%, exp p16 +11%, fine_sharp p16 +10% despite 4 -> 19 spills),
    // and on NEON (Neoverse-N2 silicon: adg_mask +9..11%, sigmoid_inv +7%; Apple M1 within noise).
    const interleave_x2 = can_interleave_x2 or ((cpu.vec_bytes >= 32 or cpu.isa == .neon) and base_unroll_x2 and rpn_stats.has_transcendental and rpn_stats.peak_live <= 8 and !has_horiz_rel and !prog.opt_int);
    const want_running_ptrs = (cpu.isa == .neon) and (info.used_clips.items.len + rel_row_keys.items.len <= 8);
    const want_pair_mem = want_running_ptrs;

    const run_src_ptrs = try allocator.alloc(?aj.Reg, n_clips);
    defer allocator.free(run_src_ptrs);
    @memset(run_src_ptrs, null);
    const run_rel_ptrs = try allocator.alloc(aj.Reg, rel_row_keys.items.len);
    defer allocator.free(run_rel_ptrs);

    const need_x_vec_i32 = left_use_dynamic or right_use_dynamic or (info.uses_x and prog.opt_int);
    // Only materialize runtime `v_x` if some unhoisted `.pixel` node actually consumes `.coord_x`.
    var need_x_vec_f32 = false;
    if (info.uses_x and !prog.opt_int) {
        if (!enable_licm) {
            need_x_vec_f32 = true;
        } else {
            for (prog.nodes.items, 0..) |node, idx| {
                if (idx != prog.root and info.use_counts.items[idx] == 0) continue;
                if (node_inv[idx] != .pixel) continue;
                const uses_coord_x = switch (node) {
                    .coord_x => idx == prog.root,
                    .unary => |u| prog.nodes.items[u.a] == .coord_x,
                    .binary => |bin| prog.nodes.items[bin.a] == .coord_x or prog.nodes.items[bin.b] == .coord_x,
                    .ternary => |t| prog.nodes.items[t.a] == .coord_x or prog.nodes.items[t.b] == .coord_x or prog.nodes.items[t.c] == .coord_x,
                    .load_abs => |la| prog.nodes.items[la.x] == .coord_x or prog.nodes.items[la.y] == .coord_x,
                    else => false,
                };
                if (uses_coord_x) {
                    need_x_vec_f32 = true;
                    break;
                }
            }
        }
    }

    const need_v_x = need_x_vec_f32 or (info.uses_x and prog.opt_int);

    var v_lane_offsets_f32: ?aj.Reg = null;
    var v_lane_offsets_i32: ?aj.Reg = null;
    var v_lane_step_f32: ?aj.Reg = null;
    var v_lane_step_i32: ?aj.Reg = null;
    if (need_x_vec_f32) {
        var lane_f: [16]f32 = @splat(0.0);
        for (0..lanes) |i| lane_f[i] = @floatFromInt(i);
        v_lane_offsets_f32 = b.constBytes(std.mem.sliceAsBytes(lane_f[0..lanes]));
        v_lane_step_f32 = b.constF32Splat(@floatFromInt(lanes));
    }
    if (need_x_vec_i32) {
        var lane_i: [16]i32 = @splat(0);
        for (0..lanes) |i| lane_i[i] = @intCast(i);
        v_lane_offsets_i32 = b.constBytes(std.mem.sliceAsBytes(lane_i[0..lanes]));
        v_lane_step_i32 = b.constI32Splat(@intCast(lanes));
    }

    // Outer y loop: `for (y = 0; y < height; y += 1)`
    const y_gp = b.newGp64();
    b.gpMovImm(y_gp, 0);
    const y_loop = b.newLabel();
    b.bind(y_loop);

    // Per-row hoisted `Y` vector.
    var v_y: ?aj.Reg = null;
    if (info.uses_y) {
        const vy = b.newVec();
        b.vecBroadcastGp32(vy, y_gp);
        if (!prog.opt_int) {
            b.vecOp2(.cvt_i32_to_f32, vy, vy);
        }
        v_y = vy;
    }

    // Per-row hoisted vertical relative source row pointers (D6 / Y27).
    const rel_row_ptrs = try allocator.alloc(aj.Reg, rel_row_keys.items.len);
    defer allocator.free(rel_row_ptrs);
    for (rel_row_keys.items, 0..) |key, ki| {
        const row_ptr = b.newGp64();
        if (spec.height <= 1) {
            b.gpMov(row_ptr, base_src_ptrs[key.clip].?);
        } else {
            const h_i32: i32 = @intCast(spec.height);
            const h_i64: i64 = h_i32;
            const eff_dy: i64 = interp.clampRelOffset(key.dy, h_i32);
            const sy = b.newGp64();
            switch (key.mode) {
                .clamp => {
                    const zero_gp = b.newGp64();
                    const max_y_gp = b.newGp64();
                    b.gpMovImm(zero_gp, 0);
                    b.gpMovImm(max_y_gp, h_i64 - 1);
                    b.gpOp3Imm(.add, sy, y_gp, eff_dy);
                    b.gpOp3(.smax, sy, sy, zero_gp);
                    b.gpOp3(.smin, sy, sy, max_y_gp);
                },
                .mirror => {
                    // D6: compile-time clamp dy to [-H, H], then reflect once.
                    b.gpOp3Imm(.add, sy, y_gp, eff_dy);

                    const zero_gp = b.newGp64();
                    b.gpMovImm(zero_gp, 0);
                    const neg_one = b.newGp64();
                    b.gpMovImm(neg_one, -1);
                    const ref_lo = b.newGp64();
                    b.gpOp3(.sub, ref_lo, neg_one, sy);
                    b.gpCmov(.slt, sy, sy, zero_gp, ref_lo);

                    const h_gp = b.newGp64();
                    b.gpMovImm(h_gp, h_i64);
                    const two_h_m1 = b.newGp64();
                    b.gpMovImm(two_h_m1, 2 * h_i64 - 1);
                    const ref_hi = b.newGp64();
                    b.gpOp3(.sub, ref_hi, two_h_m1, sy);
                    b.gpCmov(.sge, sy, sy, h_gp, ref_hi);
                },
            }
            b.gpOp3(.mul, row_ptr, sy, src_stride_regs[key.clip].?);
            b.gpOp3(.add, row_ptr, row_ptr, base_src_ptrs[key.clip].?);
        }
        rel_row_ptrs[ki] = row_ptr;
    }

    var common_env = BodyEnv{
        .dst_ptr = dst_ptr,
        .x_gp = undefined,
        .src_row_ptrs = src_row_ptrs,
        .base_src_ptrs = base_src_ptrs,
        .src_stride_vecs = src_stride_vecs,
        .rel_row_keys = rel_row_keys.items,
        .rel_row_ptrs = rel_row_ptrs,
        .prop_regs = prop_regs,
        .v_x = null,
        .v_x_i32 = null,
        .v_y = v_y,
        .v_frame_n_f32 = v_frame_n_f32,
        .v_frame_n_i32 = v_frame_n_i32,
        .node_inv = node_inv,
        .hoisted_slots = hoisted_slots,
        .hoisted_slots_hi = hoisted_slots_hi,
        .hoisted_masks = hoisted_masks,
        .hoisted_masks_hi = hoisted_masks_hi,
        .const_f32_vals = const_f32_vals,
        .abs_row_ptrs = abs_row_ptrs,
        .rpn_peak_live = rpn_stats.peak_live,
    };

    if (enable_licm) {
        try evalHoistedStage(allocator, &b, prog, info, spec, common_env, node_inv, hoisted_slots, hoisted_masks, need_hoisted_mask, .row, periodic_lanes, 0);
        common_env.hoisted_slots = hoisted_slots;
        common_env.hoisted_masks = hoisted_masks;
        if (has_2x_periodic) {
            var row_env_hi = common_env;
            row_env_hi.hoisted_slots = hoisted_slots_hi;
            row_env_hi.hoisted_masks = hoisted_masks_hi;
            try evalHoistedStage(allocator, &b, prog, info, spec, row_env_hi, node_inv, hoisted_slots_hi, hoisted_masks_hi, need_hoisted_mask, .row, periodic_lanes, lanes);
            common_env.hoisted_slots_hi = hoisted_slots_hi;
            common_env.hoisted_masks_hi = hoisted_masks_hi;
        }
    }

    const dst_bps_row: usize = spec.dst_fmt.bytesPerSample();
    const makeRunningEnv = struct {
        fn init(
            builder: *aj.Builder,
            base_env: BodyEnv,
            used_clips: []const ast.ClipId,
            src_fmts: []const ast.SampleFormat,
            dst_bps: usize,
            out_src: []?aj.Reg,
            out_rel: []aj.Reg,
            start_col: usize,
            pair_mem: bool,
        ) BodyEnv {
            var renv = base_env;
            const cd = builder.newGp64();
            if (start_col == 0) {
                builder.gpMov(cd, base_env.dst_ptr);
            } else {
                builder.gpOp3Imm(.add, cd, base_env.dst_ptr, @intCast(start_col * dst_bps));
            }
            renv.dst_ptr = cd;
            for (used_clips) |clip| {
                const cs = builder.newGp64();
                const bps = src_fmts[clip].bytesPerSample();
                if (start_col == 0) {
                    builder.gpMov(cs, base_env.src_row_ptrs[clip].?);
                } else {
                    builder.gpOp3Imm(.add, cs, base_env.src_row_ptrs[clip].?, @intCast(start_col * bps));
                }
                out_src[clip] = cs;
            }
            for (base_env.rel_row_keys, 0..) |rk, ki| {
                const cr = builder.newGp64();
                const bps = src_fmts[rk.clip].bytesPerSample();
                if (start_col == 0) {
                    builder.gpMov(cr, base_env.rel_row_ptrs[ki]);
                } else {
                    builder.gpOp3Imm(.add, cr, base_env.rel_row_ptrs[ki], @intCast(start_col * bps));
                }
                out_rel[ki] = cr;
            }
            renv.src_row_ptrs = out_src;
            renv.rel_row_ptrs = out_rel;
            renv.use_running_ptrs = true;
            renv.use_pair_mem = pair_mem;
            return renv;
        }
    }.init;

    const stepRunningEnv = struct {
        fn step(
            builder: *aj.Builder,
            renv: BodyEnv,
            used_clips: []const ast.ClipId,
            src_fmts: []const ast.SampleFormat,
            dst_bps: usize,
            step_px: usize,
        ) void {
            if (!renv.use_running_ptrs) return;
            builder.gpOp3Imm(.add, renv.dst_ptr, renv.dst_ptr, @intCast(step_px * dst_bps));
            for (used_clips) |clip| {
                const bps = src_fmts[clip].bytesPerSample();
                builder.gpOp3Imm(.add, renv.src_row_ptrs[clip].?, renv.src_row_ptrs[clip].?, @intCast(step_px * bps));
            }
            for (renv.rel_row_keys, 0..) |rk, ki| {
                const bps = src_fmts[rk.clip].bytesPerSample();
                builder.gpOp3Imm(.add, renv.rel_row_ptrs[ki], renv.rel_row_ptrs[ki], @intCast(step_px * bps));
            }
        }
    }.step;

    if (!has_horiz_rel) {
        // Single runtime vector loop across the whole row (with u8 / u16 / 2x-unrolled i16 / 2x/4x-unrolled f32/opt1 fast paths).
        var env = common_env;
        if (need_v_x) {
            const vx = b.newVec();
            b.vecOp2(.mov, vx, if (prog.opt_int) v_lane_offsets_i32.? else v_lane_offsets_f32.?);
            env.v_x = vx;
        }

        const x_gp = b.newGp64();
        b.gpMovImm(x_gp, 0);
        env.x_gp = x_gp;

        const root_hoisted_u8 = enable_licm and hoisted_slots[prog.root] != null and
            !spec.dst_fmt.is_float and spec.dst_fmt.bits_per_sample == 8 and
            (lanes == 4 or lanes == 8 or lanes == 16);
        const root_hoisted_u16 = enable_licm and hoisted_slots[prog.root] != null and
            !spec.dst_fmt.is_float and spec.dst_fmt.bits_per_sample > 8 and spec.dst_fmt.bits_per_sample <= 16 and
            (lanes == 4 or lanes == 8 or lanes == 16);
        const root_hoisted_f32 = enable_licm and hoisted_slots[prog.root] != null and
            spec.dst_fmt.is_float and spec.dst_fmt.bits_per_sample == 32 and
            has_2x_periodic and lanes == 4;

        if (root_hoisted_u8) {
            // Root is row/frame-invariant (e.g. `deblock_qed` mask via periodic `X % 8`):
            // pack once per row via the exact D4 store path, broadcast/reload, and stream raw stores!
            const root_reg = hoisted_slots[prog.root].?.reg;
            const packed_vec = b.newVec();
            const step_bytes: usize = if (lanes == 4) 16 else if (lanes == 8) 32 else 64;
            if (has_2x_periodic) {
                const root_hi = hoisted_slots_hi[prog.root].?.reg;
                // `has_2x_periodic` requires `!prog.opt_int`, so the root value type is always `.f32`.
                std.debug.assert(info.value_types.items[prog.root] == .f32);
                pixel.emitStoreF32x2(&b, aj.Mem.ptr(dst_ptr, 0), aj.Mem.ptr(dst_ptr, @intCast(lanes)), root_reg, root_hi, spec.dst_fmt, false);
                if (lanes == 4) {
                    b.vecLoad(.broadcast_u64, packed_vec, aj.Mem.ptr(dst_ptr, 0));
                } else if (lanes == 8) {
                    // AVX2 stride is guaranteed a multiple of 32 bytes: duplicate 16B -> 32B once per row.
                    b.vecLoad(.u128, packed_vec, aj.Mem.ptr(dst_ptr, 0));
                    b.vecStore(.u128, aj.Mem.ptr(dst_ptr, 16), packed_vec);
                    b.vecLoad(.u_vec, packed_vec, aj.Mem.ptr(dst_ptr, 0));
                } else {
                    // AVX-512 stride is guaranteed a multiple of 64 bytes: duplicate 32B -> 64B once per row.
                    b.vecLoad(.u256, packed_vec, aj.Mem.ptr(dst_ptr, 0));
                    b.vecStore(.u256, aj.Mem.ptr(dst_ptr, 32), packed_vec);
                    b.vecLoad(.u_vec, packed_vec, aj.Mem.ptr(dst_ptr, 0));
                }
            } else {
                if (info.value_types.items[prog.root] == .i32) {
                    pixel.emitStoreI32(&b, aj.Mem.ptr(dst_ptr, 0), root_reg, spec.dst_fmt);
                } else {
                    pixel.emitStoreF32(&b, aj.Mem.ptr(dst_ptr, 0), root_reg, spec.dst_fmt);
                }
                if (lanes == 4) {
                    b.vecLoad(.broadcast_u32, packed_vec, aj.Mem.ptr(dst_ptr, 0));
                } else if (lanes == 8) {
                    b.vecLoad(.broadcast_u64, packed_vec, aj.Mem.ptr(dst_ptr, 0));
                } else {
                    // AVX-512 frame stride is guaranteed a multiple of 64 bytes: widen 16B -> 32B -> 64B once per row.
                    b.vecLoad(.u128, packed_vec, aj.Mem.ptr(dst_ptr, 0));
                    b.vecStore(.u128, aj.Mem.ptr(dst_ptr, 16), packed_vec);
                    b.vecLoad(.u256, packed_vec, aj.Mem.ptr(dst_ptr, 0));
                    b.vecStore(.u256, aj.Mem.ptr(dst_ptr, 32), packed_vec);
                    b.vecLoad(.u_vec, packed_vec, aj.Mem.ptr(dst_ptr, 0));
                }
            }
            if (want_running_ptrs) {
                const cur_dst = b.newGp64();
                b.gpMov(cur_dst, dst_ptr);
                const x_loop = b.newLabel();
                b.bind(x_loop);
                b.vecStore(.u_vec, aj.Mem.ptr(cur_dst, 0), packed_vec);
                b.gpOp3Imm(.add, cur_dst, cur_dst, @intCast(step_bytes));
                b.gpOp3Imm(.add, x_gp, x_gp, @intCast(step_bytes));
                b.jccImm(.ult, x_gp, @intCast(spec.width), x_loop);
            } else {
                const x_loop = b.newLabel();
                b.bind(x_loop);
                b.vecStore(.u_vec, aj.Mem.indexed(dst_ptr, x_gp, 0, 0), packed_vec);
                b.gpOp3Imm(.add, x_gp, x_gp, @intCast(step_bytes));
                b.jccImm(.ult, x_gp, @intCast(spec.width), x_loop);
            }
        } else if (root_hoisted_u16) {
            // Row/frame-invariant u9..u16 root: pack once per row via exact D4 store path,
            // widen to a full 16/32/64-byte vector (safe within 32B / AVX-512 64B stride), and stream raw stores.
            const root_reg = hoisted_slots[prog.root].?.reg;
            const packed_vec = b.newVec();
            const step_px: usize = if (lanes == 4) 8 else if (lanes == 8) 16 else 32;
            if (has_2x_periodic) {
                const root_hi = hoisted_slots_hi[prog.root].?.reg;
                // `has_2x_periodic` requires `!prog.opt_int`, so the root value type is always `.f32`.
                std.debug.assert(info.value_types.items[prog.root] == .f32);
                pixel.emitStoreF32x2(&b, aj.Mem.ptr(dst_ptr, 0), aj.Mem.ptr(dst_ptr, @intCast(lanes * 2)), root_reg, root_hi, spec.dst_fmt, false);
                b.vecLoad(.u_vec, packed_vec, aj.Mem.ptr(dst_ptr, 0));
            } else {
                if (info.value_types.items[prog.root] == .i32) {
                    pixel.emitStoreI32(&b, aj.Mem.ptr(dst_ptr, 0), root_reg, spec.dst_fmt);
                } else {
                    pixel.emitStoreF32(&b, aj.Mem.ptr(dst_ptr, 0), root_reg, spec.dst_fmt);
                }
                if (lanes == 4) {
                    b.vecLoad(.broadcast_u64, packed_vec, aj.Mem.ptr(dst_ptr, 0));
                } else if (lanes == 8) {
                    b.vecLoad(.u128, packed_vec, aj.Mem.ptr(dst_ptr, 0));
                    b.vecStore(.u128, aj.Mem.ptr(dst_ptr, 16), packed_vec);
                    b.vecLoad(.u_vec, packed_vec, aj.Mem.ptr(dst_ptr, 0));
                } else {
                    b.vecLoad(.u256, packed_vec, aj.Mem.ptr(dst_ptr, 0));
                    b.vecStore(.u256, aj.Mem.ptr(dst_ptr, 32), packed_vec);
                    b.vecLoad(.u_vec, packed_vec, aj.Mem.ptr(dst_ptr, 0));
                }
            }
            if (want_running_ptrs) {
                const cur_dst = b.newGp64();
                b.gpMov(cur_dst, dst_ptr);
                const x_loop = b.newLabel();
                b.bind(x_loop);
                b.vecStore(.u_vec, aj.Mem.ptr(cur_dst, 0), packed_vec);
                b.gpOp3Imm(.add, cur_dst, cur_dst, @intCast(step_px * 2));
                b.gpOp3Imm(.add, x_gp, x_gp, @intCast(step_px));
                b.jccImm(.ult, x_gp, @intCast(spec.width), x_loop);
            } else {
                const x_loop = b.newLabel();
                b.bind(x_loop);
                b.vecStore(.u_vec, aj.Mem.indexed(dst_ptr, x_gp, 1, 0), packed_vec);
                b.gpOp3Imm(.add, x_gp, x_gp, @intCast(step_px));
                b.jccImm(.ult, x_gp, @intCast(spec.width), x_loop);
            }
        } else if (root_hoisted_f32) {
            const root_lo = hoisted_slots[prog.root].?.reg;
            const root_hi = hoisted_slots_hi[prog.root].?.reg;
            const step2 = lanes * 2;
            const unroll_end = (spec.width / step2) * step2;
            const cur_dst = b.newGp64();
            b.gpMov(cur_dst, dst_ptr);
            if (unroll_end > 0) {
                const x_loop = b.newLabel();
                b.bind(x_loop);
                b.vecStoreUVecx2(aj.Mem.ptr(cur_dst, 0), root_lo, root_hi);
                b.gpOp3Imm(.add, cur_dst, cur_dst, @intCast(step2 * 4));
                b.gpOp3Imm(.add, x_gp, x_gp, @intCast(step2));
                b.jccImm(.ult, x_gp, @intCast(unroll_end), x_loop);
            }
            if (unroll_end < spec.width) {
                if (spec.width - unroll_end > lanes) {
                    b.vecStoreUVecx2(aj.Mem.ptr(cur_dst, 0), root_lo, root_hi);
                } else {
                    b.vecStore(.u_vec, aj.Mem.ptr(cur_dst, 0), root_lo);
                }
            }
        } else if (use_u8_interior) {
            const renv = if (want_running_ptrs)
                makeRunningEnv(&b, env, info.used_clips.items, spec.src_fmts, dst_bps_row, run_src_ptrs, run_rel_ptrs, 0, want_pair_mem)
            else
                env;
            const step2 = interior_step * 2;
            const unroll_end = if (can_unroll_u8_x2 and spec.width >= step2) (spec.width / step2) * step2 else 0;
            if (unroll_end > 0) {
                const x_loop2 = b.newLabel();
                b.alignCode(32);
                b.bind(x_loop2);
                try int_path.emitU8InteriorBodyX2(allocator, &b, prog, info, spec.width, renv.dst_ptr, x_gp, renv.src_row_ptrs, renv);
                stepRunningEnv(&b, renv, info.used_clips.items, spec.src_fmts, dst_bps_row, step2);
                b.gpOp3Imm(.add, x_gp, x_gp, @intCast(step2));
                b.jccImm(.ult, x_gp, @intCast(unroll_end), x_loop2);
            }
            if (unroll_end < spec.width) {
                const x_loop = b.newLabel();
                b.alignCode(32);
                b.bind(x_loop);
                try int_path.emitU8InteriorBody(allocator, &b, prog, info, spec.width, renv.dst_ptr, x_gp, renv.src_row_ptrs, renv);
                stepRunningEnv(&b, renv, info.used_clips.items, spec.src_fmts, dst_bps_row, interior_step);
                b.gpOp3Imm(.add, x_gp, x_gp, @intCast(interior_step));
                b.jccImm(.ult, x_gp, @intCast(spec.width), x_loop);
            }
        } else if (use_u16_interior) {
            const renv = if (want_running_ptrs)
                makeRunningEnv(&b, env, info.used_clips.items, spec.src_fmts, dst_bps_row, run_src_ptrs, run_rel_ptrs, 0, want_pair_mem)
            else
                env;
            const step2 = interior_step * 2;
            const unroll_end = if (can_unroll_u16_x2 and spec.width >= step2) (spec.width / step2) * step2 else 0;
            if (unroll_end > 0) {
                const x_loop2 = b.newLabel();
                b.alignCode(32);
                b.bind(x_loop2);
                try int_path.emitU16InteriorBodyX2(allocator, &b, prog, info, spec.width, spec.dst_fmt, renv.dst_ptr, x_gp, renv.src_row_ptrs, renv);
                stepRunningEnv(&b, renv, info.used_clips.items, spec.src_fmts, dst_bps_row, step2);
                b.gpOp3Imm(.add, x_gp, x_gp, @intCast(step2));
                b.jccImm(.ult, x_gp, @intCast(unroll_end), x_loop2);
            }
            if (unroll_end < spec.width) {
                const x_loop = b.newLabel();
                b.alignCode(32);
                b.bind(x_loop);
                try int_path.emitU16InteriorBody(allocator, &b, prog, info, spec.width, spec.dst_fmt, renv.dst_ptr, x_gp, renv.src_row_ptrs, renv);
                stepRunningEnv(&b, renv, info.used_clips.items, spec.src_fmts, dst_bps_row, interior_step);
                b.gpOp3Imm(.add, x_gp, x_gp, @intCast(interior_step));
                b.jccImm(.ult, x_gp, @intCast(spec.width), x_loop);
            }
        } else if (use_int16_interior) {
            const renv = if (want_running_ptrs)
                makeRunningEnv(&b, env, info.used_clips.items, spec.src_fmts, dst_bps_row, run_src_ptrs, run_rel_ptrs, 0, want_pair_mem)
            else
                env;
            const step2 = lanes * 4;
            const unroll_end = if (can_unroll_i16_x2 and spec.width >= step2) (spec.width / step2) * step2 else 0;
            if (unroll_end > 0) {
                const x_loop2 = b.newLabel();
                b.alignCode(32);
                b.bind(x_loop2);
                try int_path.emitInteriorBodyX2(allocator, &b, prog, info, spec.width, renv.dst_ptr, x_gp, renv.src_row_ptrs, renv);
                stepRunningEnv(&b, renv, info.used_clips.items, spec.src_fmts, dst_bps_row, step2);
                b.gpOp3Imm(.add, x_gp, x_gp, @intCast(step2));
                b.jccImm(.ult, x_gp, @intCast(unroll_end), x_loop2);
            }
            if (unroll_end < spec.width) {
                const x_loop = b.newLabel();
                b.alignCode(32);
                b.bind(x_loop);
                try int_path.emitInteriorBody(allocator, &b, prog, info, spec.width, renv.dst_ptr, x_gp, renv.src_row_ptrs, renv);
                stepRunningEnv(&b, renv, info.used_clips.items, spec.src_fmts, dst_bps_row, interior_step);
                b.gpOp3Imm(.add, x_gp, x_gp, @intCast(interior_step));
                b.jccImm(.ult, x_gp, @intCast(spec.width), x_loop);
            }
        } else if (use_int32_interior) {
            const renv = if (want_running_ptrs)
                makeRunningEnv(&b, env, info.used_clips.items, spec.src_fmts, dst_bps_row, run_src_ptrs, run_rel_ptrs, 0, want_pair_mem)
            else
                env;
            const step2 = lanes * 2;
            const unroll_end = if (can_unroll_i32_x2 and spec.width >= step2) (spec.width / step2) * step2 else 0;
            if (unroll_end > 0) {
                const x_loop2 = b.newLabel();
                b.alignCode(32);
                b.bind(x_loop2);
                try int_path.emitI32InteriorBodyX2(allocator, &b, prog, info, spec.width, spec.dst_fmt, renv.dst_ptr, renv.src_row_ptrs, renv);
                stepRunningEnv(&b, renv, info.used_clips.items, spec.src_fmts, dst_bps_row, step2);
                b.gpOp3Imm(.add, x_gp, x_gp, @intCast(step2));
                b.jccImm(.ult, x_gp, @intCast(unroll_end), x_loop2);
            }
            if (unroll_end < spec.width) {
                const x_loop = b.newLabel();
                b.alignCode(32);
                b.bind(x_loop);
                try int_path.emitI32InteriorBody(allocator, &b, prog, info, spec.width, spec.dst_fmt, renv.dst_ptr, renv.src_row_ptrs, renv);
                stepRunningEnv(&b, renv, info.used_clips.items, spec.src_fmts, dst_bps_row, lanes);
                b.gpOp3Imm(.add, x_gp, x_gp, @intCast(lanes));
                b.jccImm(.ult, x_gp, @intCast(spec.width), x_loop);
            }
        } else if (can_interleave_x4 and spec.width >= lanes * 4) {
            const renv = if (want_running_ptrs)
                makeRunningEnv(&b, env, info.used_clips.items, spec.src_fmts, dst_bps_row, run_src_ptrs, run_rel_ptrs, 0, want_pair_mem)
            else
                env;
            const step4 = lanes * 4;
            const unroll_end = (spec.width / step4) * step4;
            if (unroll_end > 0) {
                const x_loop4 = b.newLabel();
                b.alignCode(32);
                b.bind(x_loop4);
                try emitDagBodyX4(allocator, &b, prog, info, spec, renv, v_lane_step_f32, v_lane_step_i32);
                if (renv.v_x) |vx| {
                    b.vecOp3(.add_f32, vx, vx, b.constF32Splat(@floatFromInt(step4)));
                }
                stepRunningEnv(&b, renv, info.used_clips.items, spec.src_fmts, dst_bps_row, step4);
                b.gpOp3Imm(.add, x_gp, x_gp, @intCast(step4));
                b.jccImm(.ult, x_gp, @intCast(unroll_end), x_loop4);
            }
            if (unroll_end < spec.width) {
                const rem = spec.width - unroll_end;
                if (rem > lanes * 2) {
                    try emitDagBodyX2(allocator, &b, prog, info, spec, renv, v_lane_step_f32, v_lane_step_i32, true);
                    if (renv.v_x) |vx| {
                        b.vecOp3(.add_f32, vx, vx, b.constF32Splat(@floatFromInt(lanes * 2)));
                    }
                    stepRunningEnv(&b, renv, info.used_clips.items, spec.src_fmts, dst_bps_row, lanes * 2);
                    b.gpOp3Imm(.add, x_gp, x_gp, @intCast(lanes * 2));
                    if (rem - lanes * 2 > lanes) {
                        try emitDagBodyX2(allocator, &b, prog, info, spec, renv, v_lane_step_f32, v_lane_step_i32, true);
                    } else {
                        try emitDagBody(allocator, &b, prog, info, spec, renv, .interior);
                    }
                } else if (rem > lanes) {
                    try emitDagBodyX2(allocator, &b, prog, info, spec, renv, v_lane_step_f32, v_lane_step_i32, true);
                } else {
                    try emitDagBody(allocator, &b, prog, info, spec, renv, .interior);
                }
            }
        } else if ((can_unroll_x2 and spec.width >= lanes * 2) or has_2x_periodic) {
            const renv = if (want_running_ptrs)
                makeRunningEnv(&b, env, info.used_clips.items, spec.src_fmts, dst_bps_row, run_src_ptrs, run_rel_ptrs, 0, want_pair_mem)
            else
                env;
            const step2 = lanes * 2;
            const unroll_end = (spec.width / step2) * step2;
            if (unroll_end > 0) {
                const x_loop2 = b.newLabel();
                b.alignCode(32);
                b.bind(x_loop2);
                try emitDagBodyX2(allocator, &b, prog, info, spec, renv, v_lane_step_f32, v_lane_step_i32, interleave_x2);
                if (renv.v_x) |vx| {
                    if (prog.opt_int) {
                        b.vecOp3(.add_i32, vx, vx, b.constI32Splat(@intCast(step2)));
                    } else {
                        b.vecOp3(.add_f32, vx, vx, b.constF32Splat(@floatFromInt(step2)));
                    }
                }
                stepRunningEnv(&b, renv, info.used_clips.items, spec.src_fmts, dst_bps_row, step2);
                b.gpOp3Imm(.add, x_gp, x_gp, @intCast(step2));
                b.jccImm(.ult, x_gp, @intCast(unroll_end), x_loop2);
            }

            if (unroll_end < spec.width) {
                if (spec.width - unroll_end > lanes) {
                    // Remainder spans both phase 0 and phase 1; `width > unroll_end + lanes` guarantees
                    // the 32B/64B-aligned stride accommodates all `2 * lanes` pixels of every src/dst format.
                    try emitDagBodyX2(allocator, &b, prog, info, spec, renv, v_lane_step_f32, v_lane_step_i32, interleave_x2);
                } else {
                    try emitDagBody(allocator, &b, prog, info, spec, renv, .interior);
                }
            }
        } else {
            std.debug.assert(!has_2x_periodic);
            const renv = if (want_running_ptrs and !info.uses_abs_access)
                makeRunningEnv(&b, env, info.used_clips.items, spec.src_fmts, dst_bps_row, run_src_ptrs, run_rel_ptrs, 0, want_pair_mem)
            else
                env;
            const x_loop = b.newLabel();
            b.alignCode(32);
            b.bind(x_loop);

            try emitDagBody(allocator, &b, prog, info, spec, renv, .interior);

            if (renv.v_x) |vx| {
                if (prog.opt_int) {
                    b.vecOp3(.add_i32, vx, vx, v_lane_step_i32.?);
                } else {
                    b.vecOp3(.add_f32, vx, vx, v_lane_step_f32.?);
                }
            }

            stepRunningEnv(&b, renv, info.used_clips.items, spec.src_fmts, dst_bps_row, lanes);
            b.gpOp3Imm(.add, x_gp, x_gp, @intCast(lanes));
            b.jccImm(.ult, x_gp, @intCast(spec.width), x_loop);
        }
    } else {
        // B1 / S5 / V12 three-part dimension-specialized row split: Left Edge | Interior | Right Edge.

        // 1. Left edge: unrolled B1 window loads when <= 8 vecs, else compact `.dynamic_left` loop.
        if (left_end > 0) {
            if (left_use_dynamic) {
                var env = common_env;
                if (need_v_x) {
                    const vx = b.newVec();
                    b.vecOp2(.mov, vx, if (prog.opt_int) v_lane_offsets_i32.? else v_lane_offsets_f32.?);
                    env.v_x = vx;
                }
                const vxi = b.newVec();
                b.vecOp2(.mov, vxi, v_lane_offsets_i32.?);
                env.v_x_i32 = vxi;
                const x_gp = b.newGp64();
                b.gpMovImm(x_gp, 0);
                env.x_gp = x_gp;

                const left_loop = b.newLabel();
                b.bind(left_loop);
                try emitDagBody(allocator, &b, prog, info, spec, env, .{ .dynamic_left = left_end });
                if (env.v_x) |vx| {
                    if (prog.opt_int) {
                        b.vecOp3(.add_i32, vx, vx, v_lane_step_i32.?);
                    } else {
                        b.vecOp3(.add_f32, vx, vx, v_lane_step_f32.?);
                    }
                }
                b.vecOp3(.add_i32, vxi, vxi, v_lane_step_i32.?);
                b.gpOp3Imm(.add, x_gp, x_gp, @intCast(lanes));
                b.jccImm(.ult, x_gp, @intCast(left_end), left_loop);
            } else {
                var col: usize = 0;
                while (col < left_end) : (col += lanes) {
                    var env = common_env;
                    if (need_v_x) {
                        env.v_x = makeConstXVec(&b, col, lanes, prog.opt_int);
                    }
                    try emitDagBody(allocator, &b, prog, info, spec, env, .{ .const_col = col });
                }
            }
        }

        // 2. Branchless interior vector loop (u8 / u16 / i16 2x-unrolled / f32/opt1 2x-unrolled).
        if (right_start > left_end) {
            var env = common_env;
            if (need_v_x) {
                const vx = b.newVec();
                b.vecOp2(.mov, vx, makeConstXVec(&b, left_end, lanes, prog.opt_int));
                env.v_x = vx;
            }
            const x_gp = b.newGp64();
            b.gpMovImm(x_gp, @intCast(left_end));
            env.x_gp = x_gp;
            const renv = if (want_running_ptrs and !info.uses_abs_access)
                makeRunningEnv(&b, env, info.used_clips.items, spec.src_fmts, dst_bps_row, run_src_ptrs, run_rel_ptrs, left_end, want_pair_mem)
            else
                env;

            const span = right_start - left_end;
            if (use_u8_interior) {
                const step2 = interior_step * 2;
                const unroll_end = if (can_unroll_u8_x2 and span >= step2) left_end + (span / step2) * step2 else left_end;
                if (unroll_end > left_end) {
                    const x_loop2 = b.newLabel();
                    b.alignCode(32);
                    b.bind(x_loop2);
                    try int_path.emitU8InteriorBodyX2(allocator, &b, prog, info, spec.width, renv.dst_ptr, x_gp, renv.src_row_ptrs, renv);
                    stepRunningEnv(&b, renv, info.used_clips.items, spec.src_fmts, dst_bps_row, step2);
                    b.gpOp3Imm(.add, x_gp, x_gp, @intCast(step2));
                    b.jccImm(.ult, x_gp, @intCast(unroll_end), x_loop2);
                }
                if (unroll_end < right_start) {
                    if (unroll_end == left_end) {
                        const x_loop = b.newLabel();
                        b.alignCode(32);
                        b.bind(x_loop);
                        try int_path.emitU8InteriorBody(allocator, &b, prog, info, spec.width, renv.dst_ptr, x_gp, renv.src_row_ptrs, renv);
                        stepRunningEnv(&b, renv, info.used_clips.items, spec.src_fmts, dst_bps_row, interior_step);
                        b.gpOp3Imm(.add, x_gp, x_gp, @intCast(interior_step));
                        b.jccImm(.ult, x_gp, @intCast(right_start), x_loop);
                    } else {
                        try int_path.emitU8InteriorBody(allocator, &b, prog, info, spec.width, renv.dst_ptr, x_gp, renv.src_row_ptrs, renv);
                        stepRunningEnv(&b, renv, info.used_clips.items, spec.src_fmts, dst_bps_row, interior_step);
                        b.gpOp3Imm(.add, x_gp, x_gp, @intCast(interior_step));
                    }
                }
            } else if (use_u16_interior) {
                const step2 = interior_step * 2;
                const unroll_end = if (can_unroll_u16_x2 and span >= step2) left_end + (span / step2) * step2 else left_end;
                if (unroll_end > left_end) {
                    const x_loop2 = b.newLabel();
                    b.alignCode(32);
                    b.bind(x_loop2);
                    try int_path.emitU16InteriorBodyX2(allocator, &b, prog, info, spec.width, spec.dst_fmt, renv.dst_ptr, x_gp, renv.src_row_ptrs, renv);
                    stepRunningEnv(&b, renv, info.used_clips.items, spec.src_fmts, dst_bps_row, step2);
                    b.gpOp3Imm(.add, x_gp, x_gp, @intCast(step2));
                    b.jccImm(.ult, x_gp, @intCast(unroll_end), x_loop2);
                }
                if (unroll_end < right_start) {
                    if (unroll_end == left_end) {
                        const x_loop = b.newLabel();
                        b.alignCode(32);
                        b.bind(x_loop);
                        try int_path.emitU16InteriorBody(allocator, &b, prog, info, spec.width, spec.dst_fmt, renv.dst_ptr, x_gp, renv.src_row_ptrs, renv);
                        stepRunningEnv(&b, renv, info.used_clips.items, spec.src_fmts, dst_bps_row, interior_step);
                        b.gpOp3Imm(.add, x_gp, x_gp, @intCast(interior_step));
                        b.jccImm(.ult, x_gp, @intCast(right_start), x_loop);
                    } else {
                        try int_path.emitU16InteriorBody(allocator, &b, prog, info, spec.width, spec.dst_fmt, renv.dst_ptr, x_gp, renv.src_row_ptrs, renv);
                        stepRunningEnv(&b, renv, info.used_clips.items, spec.src_fmts, dst_bps_row, interior_step);
                        b.gpOp3Imm(.add, x_gp, x_gp, @intCast(interior_step));
                    }
                }
            } else if (use_int16_interior) {
                const step2 = lanes * 4;
                const unroll_end = if (can_unroll_i16_x2 and span >= step2) left_end + (span / step2) * step2 else left_end;
                if (unroll_end > left_end) {
                    const x_loop2 = b.newLabel();
                    b.alignCode(32);
                    b.bind(x_loop2);
                    try int_path.emitInteriorBodyX2(allocator, &b, prog, info, spec.width, renv.dst_ptr, x_gp, renv.src_row_ptrs, renv);
                    stepRunningEnv(&b, renv, info.used_clips.items, spec.src_fmts, dst_bps_row, step2);
                    b.gpOp3Imm(.add, x_gp, x_gp, @intCast(step2));
                    b.jccImm(.ult, x_gp, @intCast(unroll_end), x_loop2);
                }
                if (unroll_end < right_start) {
                    const x_loop = b.newLabel();
                    b.alignCode(32);
                    b.bind(x_loop);
                    try int_path.emitInteriorBody(allocator, &b, prog, info, spec.width, renv.dst_ptr, x_gp, renv.src_row_ptrs, renv);
                    stepRunningEnv(&b, renv, info.used_clips.items, spec.src_fmts, dst_bps_row, interior_step);
                    b.gpOp3Imm(.add, x_gp, x_gp, @intCast(interior_step));
                    b.jccImm(.ult, x_gp, @intCast(right_start), x_loop);
                }
            } else if (use_int32_interior) {
                const step2 = lanes * 2;
                const unroll_end = if (can_unroll_i32_x2 and span >= step2) left_end + (span / step2) * step2 else left_end;
                if (unroll_end > left_end) {
                    const x_loop2 = b.newLabel();
                    b.alignCode(32);
                    b.bind(x_loop2);
                    try int_path.emitI32InteriorBodyX2(allocator, &b, prog, info, spec.width, spec.dst_fmt, renv.dst_ptr, renv.src_row_ptrs, renv);
                    stepRunningEnv(&b, renv, info.used_clips.items, spec.src_fmts, dst_bps_row, step2);
                    b.gpOp3Imm(.add, x_gp, x_gp, @intCast(step2));
                    b.jccImm(.ult, x_gp, @intCast(unroll_end), x_loop2);
                }
                if (unroll_end < right_start) {
                    const x_loop = b.newLabel();
                    b.alignCode(32);
                    b.bind(x_loop);
                    try int_path.emitI32InteriorBody(allocator, &b, prog, info, spec.width, spec.dst_fmt, renv.dst_ptr, renv.src_row_ptrs, renv);
                    stepRunningEnv(&b, renv, info.used_clips.items, spec.src_fmts, dst_bps_row, lanes);
                    b.gpOp3Imm(.add, x_gp, x_gp, @intCast(lanes));
                    b.jccImm(.ult, x_gp, @intCast(right_start), x_loop);
                }
            } else if (can_unroll_x2 and span >= lanes * 2) {
                const step2 = lanes * 2;
                const unroll_end = left_end + (span / step2) * step2;
                const x_loop2 = b.newLabel();
                b.alignCode(32);
                b.bind(x_loop2);
                try emitDagBodyX2(allocator, &b, prog, info, spec, renv, v_lane_step_f32, v_lane_step_i32, interleave_x2);
                if (renv.v_x) |vx| {
                    if (prog.opt_int) {
                        b.vecOp3(.add_i32, vx, vx, b.constI32Splat(@intCast(step2)));
                    } else {
                        b.vecOp3(.add_f32, vx, vx, b.constF32Splat(@floatFromInt(step2)));
                    }
                }
                stepRunningEnv(&b, renv, info.used_clips.items, spec.src_fmts, dst_bps_row, step2);
                b.gpOp3Imm(.add, x_gp, x_gp, @intCast(step2));
                b.jccImm(.ult, x_gp, @intCast(unroll_end), x_loop2);

                if (unroll_end < right_start) {
                    const x_loop = b.newLabel();
                    b.alignCode(32);
                    b.bind(x_loop);
                    try emitDagBody(allocator, &b, prog, info, spec, renv, .interior);
                    if (renv.v_x) |vx| {
                        if (prog.opt_int) {
                            b.vecOp3(.add_i32, vx, vx, v_lane_step_i32.?);
                        } else {
                            b.vecOp3(.add_f32, vx, vx, v_lane_step_f32.?);
                        }
                    }
                    stepRunningEnv(&b, renv, info.used_clips.items, spec.src_fmts, dst_bps_row, lanes);
                    b.gpOp3Imm(.add, x_gp, x_gp, @intCast(lanes));
                    b.jccImm(.ult, x_gp, @intCast(right_start), x_loop);
                }
            } else {
                const x_loop = b.newLabel();
                b.alignCode(32);
                b.bind(x_loop);
                try emitDagBody(allocator, &b, prog, info, spec, renv, .interior);
                if (renv.v_x) |vx| {
                    if (prog.opt_int) {
                        b.vecOp3(.add_i32, vx, vx, v_lane_step_i32.?);
                    } else {
                        b.vecOp3(.add_f32, vx, vx, v_lane_step_f32.?);
                    }
                }
                stepRunningEnv(&b, renv, info.used_clips.items, spec.src_fmts, dst_bps_row, interior_step);
                b.gpOp3Imm(.add, x_gp, x_gp, @intCast(interior_step));
                b.jccImm(.ult, x_gp, @intCast(right_start), x_loop);
            }
        }

        // 3. Right edge + tail: unrolled B1 window loads when <= 8 vecs, else compact `.dynamic_right` loop.
        if (right_start < spec.width) {
            if (right_use_dynamic) {
                var env = common_env;
                if (need_v_x) {
                    const vx = b.newVec();
                    b.vecOp2(.mov, vx, makeConstXVec(&b, right_start, lanes, prog.opt_int));
                    env.v_x = vx;
                }
                const vxi = b.newVec();
                b.vecOp2(.mov, vxi, makeConstXVec(&b, right_start, lanes, true));
                env.v_x_i32 = vxi;
                const x_gp = b.newGp64();
                b.gpMovImm(x_gp, @intCast(right_start));
                env.x_gp = x_gp;

                const right_loop = b.newLabel();
                b.bind(right_loop);
                try emitDagBody(allocator, &b, prog, info, spec, env, .{ .dynamic_right = right_start });
                if (env.v_x) |vx| {
                    if (prog.opt_int) {
                        b.vecOp3(.add_i32, vx, vx, v_lane_step_i32.?);
                    } else {
                        b.vecOp3(.add_f32, vx, vx, v_lane_step_f32.?);
                    }
                }
                b.vecOp3(.add_i32, vxi, vxi, v_lane_step_i32.?);
                b.gpOp3Imm(.add, x_gp, x_gp, @intCast(lanes));
                b.jccImm(.ult, x_gp, @intCast(spec.width), right_loop);
            } else {
                var col = right_start;
                while (col < spec.width) : (col += lanes) {
                    var env = common_env;
                    if (need_v_x) {
                        env.v_x = makeConstXVec(&b, col, lanes, prog.opt_int);
                    }
                    try emitDagBody(allocator, &b, prog, info, spec, env, .{ .const_col = col });
                }
            }
        }
    }

    // Advance row pointers by stride.
    b.gpOp3(.add, dst_ptr, dst_ptr, dst_stride);
    for (info.used_clips.items) |clip| {
        b.gpOp3(.add, src_row_ptrs[clip].?, src_row_ptrs[clip].?, src_stride_regs[clip].?);
    }

    b.gpOp3Imm(.add, y_gp, y_gp, 1);
    b.jccImm(.ult, y_gp, @intCast(spec.height), y_loop);

    b.retVoid();
    b.endFunc();
    return try finishCompile(allocator, rt, &b, spec.capture_disasm);
}

fn makeConstXVec(b: *aj.Builder, col: usize, lanes: usize, opt_int: bool) aj.Reg {
    if (opt_int) {
        var vals: [16]i32 = @splat(0);
        for (0..lanes) |i| vals[i] = @intCast(col + i);
        return b.constBytes(std.mem.sliceAsBytes(vals[0..lanes]));
    } else {
        var vals: [16]f32 = @splat(0.0);
        for (0..lanes) |i| vals[i] = @floatFromInt(col + i);
        return b.constBytes(std.mem.sliceAsBytes(vals[0..lanes]));
    }
}

fn finishCompile(allocator: std.mem.Allocator, rt: *aj.Runtime, b: *aj.Builder, capture_disasm: bool) CompileError!CompileResult {
    const kernel = try b.finalize();
    errdefer rt.release(kernel.func);
    const disasm_copy: ?[]u8 = if (capture_disasm) try allocator.dupe(u8, b.disasm()) else null;
    return .{
        .kernel = kernel,
        .disasm = disasm_copy,
    };
}

const BodyEnv = struct {
    dst_ptr: aj.Reg,
    x_gp: aj.Reg,
    src_row_ptrs: []const ?aj.Reg,
    base_src_ptrs: []const ?aj.Reg,
    src_stride_vecs: []const ?aj.Reg,
    rel_row_keys: []const RelRowKey,
    rel_row_ptrs: []const aj.Reg,
    prop_regs: []const aj.Reg,
    v_x: ?aj.Reg,
    v_x_i32: ?aj.Reg,
    v_y: ?aj.Reg,
    v_frame_n_f32: ?aj.Reg,
    v_frame_n_i32: ?aj.Reg,
    node_inv: []const Invariance = &.{},
    hoisted_slots: []const ?RegSlot = &.{},
    hoisted_slots_hi: []const ?RegSlot = &.{},
    hoisted_masks: []const ?aj.Reg = &.{},
    hoisted_masks_hi: []const ?aj.Reg = &.{},
    const_f32_vals: []const ?f32 = &.{},
    abs_row_ptrs: []const ?aj.Reg = &.{},
    rpn_peak_live: u32 = 0,
    use_running_ptrs: bool = false,
    use_pair_mem: bool = false,

    pub fn rowPtrFor(self: BodyEnv, clip: ast.ClipId, dy: i32, mode: ast.BoundaryMode) aj.Reg {
        if (dy == 0) return self.src_row_ptrs[clip].?;
        for (self.rel_row_keys, 0..) |k, i| {
            if (k.clip == clip and k.dy == dy and k.mode == mode) {
                return self.rel_row_ptrs[i];
            }
        }
        unreachable;
    }

    pub inline fn rowMem(self: BodyEnv, row_ptr: aj.Reg, shift: u8, disp: i32) aj.Mem {
        return if (self.use_running_ptrs)
            aj.Mem.ptr(row_ptr, disp)
        else
            aj.Mem.indexed(row_ptr, self.x_gp, shift, disp);
    }
};

const RegSlot = struct {
    reg: aj.Reg,
    owned: bool,
};

fn emitRelLoad(
    b: *aj.Builder,
    dst: aj.Reg,
    out_ty: analysis_mod.ValueType,
    spec: CompileSpec,
    env: BodyEnv,
    x_mode: XMode,
    col_disp: usize,
    clip: ast.ClipId,
    dx: i32,
    dy: i32,
    mode: ast.BoundaryMode,
) void {
    const fmt = spec.src_fmts[clip];
    const bps: usize = fmt.bytesPerSample();
    const shift: u8 = switch (bps) {
        1 => 0,
        2 => 1,
        else => 2,
    };
    const row_ptr = env.rowPtrFor(clip, dy, mode);
    const w_i32: i32 = @intCast(spec.width);
    const eff_dx: i32 = interp.clampRelOffset(dx, w_i32);

    switch (x_mode) {
        .interior => {
            const disp: i32 = (eff_dx + @as(i32, @intCast(col_disp))) * @as(i32, @intCast(bps));
            const mem = env.rowMem(row_ptr, shift, disp);
            if (out_ty == .i32) {
                pixel.emitLoadI32(b, dst, mem, fmt);
            } else {
                pixel.emitLoadF32(b, dst, mem, fmt);
            }
        },
        .const_col => |col| {
            // B1 corrected single-window edge load + compile-time lane permutation!
            const lanes: usize = b.cpu.f32_lanes;
            var idxs: [16]usize = @splat(0);
            var min_idx: usize = spec.width - 1;
            for (0..lanes) |j| {
                const vx = col + j;
                if (vx < spec.width) {
                    const src_x: usize = switch (mode) {
                        .clamp => interp.clampCoord(@as(i32, @intCast(vx)) + eff_dx, w_i32),
                        .mirror => interp.mirrorCoord(@as(i32, @intCast(vx)) + eff_dx, w_i32),
                    };
                    idxs[j] = src_x;
                    if (src_x < min_idx) min_idx = src_x;
                }
            }
            const max_window_start: usize = if (spec.width >= lanes) spec.width - lanes else 0;
            const load_x: usize = @min(min_idx, max_window_start);

            var perm: [16]u32 = @splat(0);
            var is_identity = true;
            for (0..lanes) |j| {
                if (col + j < spec.width) {
                    perm[j] = @intCast(idxs[j] - load_x);
                    if (perm[j] != j) is_identity = false;
                } else {
                    perm[j] = 0;
                }
            }

            const disp: i32 = @intCast(load_x * bps);
            const mem = aj.Mem.ptr(row_ptr, disp);
            if (out_ty == .i32) {
                pixel.emitLoadI32(b, dst, mem, fmt);
            } else {
                pixel.emitLoadF32(b, dst, mem, fmt);
            }
            if (!is_identity) {
                b.vecPermuteF32(dst, dst, perm[0..lanes]);
            }
        },
        .dynamic_left, .dynamic_right => {
            // Direct vector load when `eff_dx` is provably in-bounds across this edge strip!
            const in_bounds = switch (x_mode) {
                .dynamic_left => |left_end| eff_dx >= 0 and left_end + @as(usize, @intCast(eff_dx)) <= spec.width,
                .dynamic_right => |right_start| eff_dx <= 0 and right_start >= @as(usize, @intCast(-eff_dx)),
                else => false,
            };
            if (eff_dx == 0 or in_bounds) {
                const mem = aj.Mem.indexed(row_ptr, env.x_gp, shift, eff_dx * @as(i32, @intCast(bps)));
                if (out_ty == .i32) {
                    pixel.emitLoadI32(b, dst, mem, fmt);
                } else {
                    pixel.emitLoadF32(b, dst, mem, fmt);
                }
                return;
            }
            const ix = b.newVec();
            if (spec.width <= 1) {
                b.vecOp2(.mov, ix, b.vecZero());
            } else {
                b.vecOp3(.add_i32, ix, env.v_x_i32.?, b.constI32Splat(eff_dx));
                emitBoundI32(b, ix, w_i32, mode);
            }
            if (shift != 0) {
                b.vecOp2i(.sll_u32, ix, ix, shift);
            }
            emitGatherFromBase(b, dst, out_ty, row_ptr, ix, fmt, spec.width * bps <= 0x7FFF_0000);
        },
    }
}

/// Branchless clamp or single-reflection mirror of an `i32` coordinate vector into `[0, dim - 1]`.
fn emitBoundI32(b: *aj.Builder, ix: aj.Reg, dim: i32, mode: ast.BoundaryMode) void {
    const zero = b.vecZero();
    const max_c = b.constI32Splat(dim - 1);
    switch (mode) {
        .clamp => {
            b.vecOp3(.max_i32, ix, ix, zero);
            b.vecOp3(.min_i32, ix, ix, max_c);
        },
        .mirror => {
            // Reflect < 0: ix = -1 - ix
            const neg_mask = b.newVec();
            b.vecOp3(.cmp_gt_i32, neg_mask, zero, ix);
            const neg_ref = b.newVec();
            b.vecOp3(.sub_i32, neg_ref, b.constI32Splat(-1), ix);
            pixel.emitSelectByMask(b, ix, ix, neg_ref, neg_mask);

            // Reflect >= dim: ix = (2*dim - 1) - ix
            const hi_mask = b.newVec();
            b.vecOp3(.cmp_gt_i32, hi_mask, ix, max_c);
            const hi_ref = b.newVec();
            b.vecOp3(.sub_i32, hi_ref, b.constI32Splat(2 * dim - 1), ix);
            pixel.emitSelectByMask(b, ix, ix, hi_ref, hi_mask);

            // Final safety clamp.
            b.vecOp3(.max_i32, ix, ix, zero);
            b.vecOp3(.min_i32, ix, ix, max_c);
        },
    }
}

/// D7/U6 bounds-safe float coordinate to `i32` index (`max_f32` first, `coord` first, 0 extra NaN cost).
fn emitAbsCoordToI32(b: *aj.Builder, dst_i32: aj.Reg, coord_f32: aj.Reg, dim: i32, mode: ast.BoundaryMode) void {
    if (dim <= 1) {
        b.vecOp2(.mov, dst_i32, b.vecZero());
        return;
    }
    const dim_f: f32 = @floatFromInt(dim);
    switch (mode) {
        .clamp => {
            // G1: use zero register (`vxorps`) rather than memory operand to avoid extra load-port traffic.
            b.vecOp3(.max_f32, dst_i32, coord_f32, b.vecZero());
            b.vecMinCF32(dst_i32, dst_i32, dim_f - 1.0);
            b.vecOp2(.cvt_round_f32_to_i32, dst_i32, dst_i32);
        },
        .mirror => {
            b.vecMaxCF32(dst_i32, coord_f32, -dim_f);
            b.vecMinCF32(dst_i32, dst_i32, 2.0 * dim_f - 1.0);
            b.vecOp2(.cvt_round_f32_to_i32, dst_i32, dst_i32);
            emitBoundI32(b, dst_i32, dim, .mirror);
        },
    }
}

fn nodeConstF32(prog: *const ast.Program, env: BodyEnv, idx: ast.NodeIdx) ?f32 {
    if (env.const_f32_vals.len == prog.nodes.items.len) {
        if (env.const_f32_vals[idx]) |v| return v;
    }
    return switch (prog.nodes.items[idx]) {
        .const_f32 => |v| v,
        .const_i32 => |v| if (!prog.opt_int) @floatFromInt(v) else null,
        else => null,
    };
}

fn isPosZeroF32(v: f32) bool {
    return @as(u32, @bitCast(v)) == 0;
}

fn resolveSlotReg(
    b: *aj.Builder,
    prog: *const ast.Program,
    info: *const analysis_mod.Analysis,
    env: BodyEnv,
    slots: []?RegSlot,
    idx: ast.NodeIdx,
) aj.Reg {
    if (slots[idx]) |s| return s.reg;
    if (env.const_f32_vals.len == prog.nodes.items.len) {
        if (env.const_f32_vals[idx]) |v| {
            const reg = b.constF32Splat(v);
            slots[idx] = .{ .reg = reg, .owned = false };
            return reg;
        }
    }
    const out_ty = info.value_types.items[idx];
    const reg: aj.Reg = switch (prog.nodes.items[idx]) {
        .const_f32 => |v| b.constF32Splat(v),
        .const_i32 => |v| if (out_ty == .i32) b.constI32Splat(v) else b.constF32Splat(@floatFromInt(v)),
        else => unreachable,
    };
    slots[idx] = .{ .reg = reg, .owned = false };
    return reg;
}

fn cmpBinOpToVecOp(op: ast.BinaryOp) ?aj.VecOp3 {
    return switch (op) {
        .lt => .cmp_lt_f32,
        .le => .cmp_le_f32,
        .gt => .cmp_gt_f32,
        .ge => .cmp_ge_f32,
        .eq => .cmp_eq_f32,
        else => null,
    };
}

/// Exact power-of-two modulo `a - trunc(a * (1 / c)) * c` (unfused D3, bit-exact for every f32).
fn emitModPow2F32(b: *aj.Builder, dst: aj.Reg, a: aj.Reg, c: f32) void {
    const q = b.newVec();
    b.vecOp3(.mul_f32, q, a, b.constF32Splat(1.0 / c));
    pixel.emitRoundF32(b, q, q, .trunc);
    b.vecOp3(.mul_f32, q, q, b.constF32Splat(c));
    b.vecOp3(.sub_f32, dst, a, q);
}

fn isFusableBoolF32(
    prog: *const ast.Program,
    info: *const analysis_mod.Analysis,
    env: BodyEnv,
    slots: []const ?RegSlot,
    idx: ast.NodeIdx,
) bool {
    if (prog.opt_int) return false;
    if (slots[idx] != null or nodeConstF32(prog, env, idx) != null) return false;
    if (info.use_counts.items[idx] != 1 or info.value_types.items[idx] != .f32) return false;
    return switch (prog.nodes.items[idx]) {
        .unary => |u| u.op == .not and info.value_types.items[u.a] == .f32,
        .binary => |bin| switch (bin.op) {
            .lt, .le, .gt, .ge, .eq, .and_op, .or_op, .xor_op => info.value_types.items[bin.a] == .f32 and info.value_types.items[bin.b] == .f32,
            else => false,
        },
        else => false,
    };
}

fn markFusableBoolTreeF32(
    prog: *const ast.Program,
    info: *const analysis_mod.Analysis,
    env: BodyEnv,
    slots: []const ?RegSlot,
    fused_cmp: []bool,
    idx: ast.NodeIdx,
) void {
    if (!isFusableBoolF32(prog, info, env, slots, idx)) return;
    fused_cmp[idx] = true;
    switch (prog.nodes.items[idx]) {
        .binary => |bin| switch (bin.op) {
            .and_op, .or_op, .xor_op => {
                markFusableBoolTreeF32(prog, info, env, slots, fused_cmp, bin.a);
                markFusableBoolTreeF32(prog, info, env, slots, fused_cmp, bin.b);
            },
            else => {},
        },
        else => {},
    }
}

fn emitBoolMaskF32(
    b: *aj.Builder,
    prog: *const ast.Program,
    info: *const analysis_mod.Analysis,
    env: BodyEnv,
    slots: []?RegSlot,
    rem_uses: []u32,
    fused_cmp: []const bool,
    use_const_mem: bool,
    idx: ast.NodeIdx,
) aj.Reg {
    if (!fused_cmp[idx]) {
        if (env.hoisted_masks.len == prog.nodes.items.len) {
            if (env.hoisted_masks[idx]) |hm| {
                if (rem_uses[idx] > 0) rem_uses[idx] -= 1;
                return hm;
            }
        }
        const mask = b.newVec();
        const reg = resolveSlotReg(b, prog, info, env, slots, idx);
        b.vecOp3(.cmp_gt_f32, mask, reg, b.vecZero());
        if (rem_uses[idx] > 0) {
            rem_uses[idx] -= 1;
            if (rem_uses[idx] == 0) {
                if (slots[idx]) |s| {
                    if (s.owned) slots[idx] = null;
                }
            }
        }
        return mask;
    }
    const mask = b.newVec();
    switch (prog.nodes.items[idx]) {
        .unary => |u| {
            // `.not`: `a <= 0.0` (matches `emitUnaryF32(.not)` bit-for-bit incl. NaN).
            const a_reg = resolveSlotReg(b, prog, info, env, slots, u.a);
            b.vecOp3(.cmp_le_f32, mask, a_reg, b.vecZero());
            if (rem_uses[u.a] > 0) {
                rem_uses[u.a] -= 1;
                if (rem_uses[u.a] == 0) {
                    if (slots[u.a]) |s| {
                        if (s.owned) slots[u.a] = null;
                    }
                }
            }
            return mask;
        },
        .binary => |bin| {
            if (cmpBinOpToVecOp(bin.op)) |cmp_op| {
                const lhs_reg = resolveSlotReg(b, prog, info, env, slots, bin.a);
                if (use_const_mem) {
                    if (nodeConstF32(prog, env, bin.b)) |rhs_c| {
                        b.vecCmpCF32(cmp_op, mask, lhs_reg, rhs_c);
                    } else {
                        b.vecOp3(cmp_op, mask, lhs_reg, resolveSlotReg(b, prog, info, env, slots, bin.b));
                    }
                } else {
                    b.vecOp3(cmp_op, mask, lhs_reg, resolveSlotReg(b, prog, info, env, slots, bin.b));
                }
                for ([_]ast.NodeIdx{ bin.a, bin.b }) |child| {
                    if (rem_uses[child] > 0) {
                        rem_uses[child] -= 1;
                        if (rem_uses[child] == 0) {
                            if (slots[child]) |s| {
                                if (s.owned) slots[child] = null;
                            }
                        }
                    }
                }
                return mask;
            }
            const ma = emitBoolMaskF32(b, prog, info, env, slots, rem_uses, fused_cmp, use_const_mem, bin.a);
            const mb = emitBoolMaskF32(b, prog, info, env, slots, rem_uses, fused_cmp, use_const_mem, bin.b);
            switch (bin.op) {
                .and_op => b.vecOp3(.and_u32, mask, ma, mb),
                .or_op => b.vecOp3(.or_u32, mask, ma, mb),
                .xor_op => b.vecOp3(.xor_u32, mask, ma, mb),
                else => unreachable,
            }
            return mask;
        },
        else => unreachable,
    }
}

fn emitGatherFromBase(
    b: *aj.Builder,
    dst: aj.Reg,
    out_ty: analysis_mod.ValueType,
    base_ptr: aj.Reg,
    byte_off_i32: aj.Reg,
    fmt: ast.SampleFormat,
    can_use_hw_gather: bool,
) void {
    if (fmt.is_float) {
        if (fmt.bits_per_sample == 16) {
            b.vecGatherF16(dst, base_ptr, byte_off_i32);
        } else {
            b.vecGatherF32(dst, base_ptr, byte_off_i32, can_use_hw_gather);
        }
        return;
    }
    if (fmt.bits_per_sample <= 8) {
        b.vecGatherU8(dst, base_ptr, byte_off_i32);
    } else if (fmt.bits_per_sample <= 16) {
        b.vecGatherU16(dst, base_ptr, byte_off_i32);
    } else {
        b.vecGatherF32(dst, base_ptr, byte_off_i32, can_use_hw_gather);
    }
    if (out_ty == .f32) {
        b.vecOp2(.cvt_i32_to_f32, dst, dst);
    }
}

/// Evaluate live DAG nodes at `target_inv` (`.frame` before y_loop or `.row` before x_loop),
/// recording their registers in `hoisted_slots` as non-owned (`owned = false`) for deeper loops.
/// G2: hoist all invariant nodes without a frontier cap; spilling a row/frame value and folding
/// its stack reload into the consumer is faster than recomputing invariant arithmetic per pixel.
fn evalHoistedStage(
    allocator: std.mem.Allocator,
    b: *aj.Builder,
    prog: *const ast.Program,
    info: *const analysis_mod.Analysis,
    spec: CompileSpec,
    env: BodyEnv,
    node_inv: []const Invariance,
    hoisted_slots: []?RegSlot,
    hoisted_masks: []?aj.Reg,
    need_hoisted_mask: []const bool,
    target_inv: Invariance,
    periodic_lanes: usize,
    phase_offset: usize,
) std.mem.Allocator.Error!void {
    const n_nodes = prog.nodes.items.len;

    const trig_cache = try allocator.alloc(?math_jit.TrigState, n_nodes);
    defer allocator.free(trig_cache);
    @memset(trig_cache, null);

    const max_plane_bytes_f32: u64 = @as(u64, spec.height) * std.mem.alignForward(u64, @as(u64, spec.width) * 4 + 64, 64);
    const can_hw_gather_f32 = max_plane_bytes_f32 <= 0x7FFF_0000;

    const ensureF32 = struct {
        fn run(builder: *aj.Builder, reg: aj.Reg, ty: analysis_mod.ValueType) aj.Reg {
            if (ty == .f32) return reg;
            const dst = builder.newVec();
            builder.vecOp2(.cvt_i32_to_f32, dst, reg);
            return dst;
        }
    }.run;

    for (prog.nodes.items, 0..) |node, idx| {
        if (idx != prog.root and info.use_counts.items[idx] == 0) continue;
        if (node_inv[idx] != target_inv) continue;
        // Compile-time folded constants (including `width 2 /`, `height 2 /`) need no runtime prologue work.
        if (nodeConstF32(prog, env, @intCast(idx)) != null) continue;

        const out_ty = info.value_types.items[idx];
        switch (node) {
            .const_f32 => {},
            .const_i32 => |v| {
                if (out_ty == .i32) {
                    hoisted_slots[idx] = .{ .reg = b.constI32Splat(v), .owned = false };
                }
            },
            .load_abs => |la| {
                const fmt = spec.src_fmts[la.clip];
                const shift: u8 = switch (fmt.bytesPerSample()) {
                    1 => 0,
                    2 => 1,
                    else => 2,
                };
                const vx_raw = resolveSlotReg(b, prog, info, env, hoisted_slots, la.x);
                const fx = ensureF32(b, vx_raw, info.value_types.items[la.x]);
                const ix = b.newVec();
                emitAbsCoordToI32(b, ix, fx, @intCast(spec.width), la.mode);
                if (shift != 0) b.vecOp2i(.sll_u32, ix, ix, shift);

                const vy_raw = resolveSlotReg(b, prog, info, env, hoisted_slots, la.y);
                const fy = ensureF32(b, vy_raw, info.value_types.items[la.y]);
                const iy = b.newVec();
                emitAbsCoordToI32(b, iy, fy, @intCast(spec.height), la.mode);
                b.vecOp3(.mul_i32, iy, iy, env.src_stride_vecs[la.clip].?);

                const byte_off = b.newVec();
                b.vecOp3(.add_i32, byte_off, iy, ix);
                const dst = b.newVec();
                emitGatherFromBase(b, dst, out_ty, env.base_src_ptrs[la.clip].?, byte_off, fmt, can_hw_gather_f32);
                hoisted_slots[idx] = .{ .reg = dst, .owned = false };
            },
            .load_prop => |pid| {
                hoisted_slots[idx] = .{ .reg = env.prop_regs[pid], .owned = false };
            },
            .coord_y => {
                hoisted_slots[idx] = .{ .reg = env.v_y.?, .owned = false };
            },
            .frame_n => {
                hoisted_slots[idx] = .{ .reg = if (out_ty == .i32) env.v_frame_n_i32.? else env.v_frame_n_f32.?, .owned = false };
            },
            .plane_width => {
                hoisted_slots[idx] = .{
                    .reg = if (out_ty == .i32) b.constI32Splat(@intCast(spec.width)) else b.constF32Splat(@floatFromInt(spec.width)),
                    .owned = false,
                };
            },
            .plane_height => {
                hoisted_slots[idx] = .{
                    .reg = if (out_ty == .i32) b.constI32Splat(@intCast(spec.height)) else b.constF32Splat(@floatFromInt(spec.height)),
                    .owned = false,
                };
            },
            .unary => |u| {
                const src_reg = resolveSlotReg(b, prog, info, env, hoisted_slots, u.a);
                const src_ty = info.value_types.items[u.a];
                const dst = b.newVec();
                if (u.op == .to_f32) {
                    if (src_ty == .i32) b.vecOp2(.cvt_i32_to_f32, dst, src_reg) else b.vecOp2(.mov, dst, src_reg);
                } else if (out_ty == .i32 and src_ty == .i32) {
                    emitUnaryI32(b, u.op, dst, src_reg);
                } else if (out_ty == .i32) {
                    emitUnaryF32ToI32(b, u.op, dst, src_reg);
                } else {
                    const a_f = ensureF32(b, src_reg, src_ty);
                    switch (u.op) {
                        .sin, .cos => {
                            const st = trig_cache[u.a] orelse blk: {
                                const prepared = math_jit.emitTrigPrepare(b, a_f);
                                trig_cache[u.a] = prepared;
                                break :blk prepared;
                            };
                            switch (u.op) {
                                .sin => math_jit.emitSinFromState(b, dst, st),
                                .cos => math_jit.emitCosFromState(b, dst, st),
                                else => unreachable,
                            }
                        },
                        else => emitUnaryF32(b, u.op, dst, a_f, true),
                    }
                }
                hoisted_slots[idx] = .{ .reg = dst, .owned = false };
            },
            .binary => |bin| {
                const ta = info.value_types.items[bin.a];
                const tb = info.value_types.items[bin.b];
                if (periodicCoordXMod(prog, env.const_f32_vals, node, spec.width, periodic_lanes)) |period| {
                    var lane_f: [16]f32 = @splat(0.0);
                    for (0..b.cpu.f32_lanes) |j| {
                        lane_f[j] = @floatFromInt((j + phase_offset) % period);
                    }
                    hoisted_slots[idx] = .{
                        .reg = b.constBytes(std.mem.sliceAsBytes(lane_f[0..b.cpu.f32_lanes])),
                        .owned = false,
                    };
                } else {
                    const dst = b.newVec();
                    if (out_ty == .i32 and ta == .i32 and tb == .i32) {
                        emitBinaryI32(b, bin.op, dst, resolveSlotReg(b, prog, info, env, hoisted_slots, bin.a), resolveSlotReg(b, prog, info, env, hoisted_slots, bin.b));
                    } else if (out_ty == .i32) {
                        emitBinaryF32ToI32(b, bin.op, dst, ensureF32(b, resolveSlotReg(b, prog, info, env, hoisted_slots, bin.a), ta), ensureF32(b, resolveSlotReg(b, prog, info, env, hoisted_slots, bin.b), tb));
                    } else if (bin.op == .mod and nodeConstF32(prog, env, bin.b) != null and isExactPow2Divisor(nodeConstF32(prog, env, bin.b).?)) {
                        emitModPow2F32(b, dst, ensureF32(b, resolveSlotReg(b, prog, info, env, hoisted_slots, bin.a), ta), nodeConstF32(prog, env, bin.b).?);
                    } else if (bin.op == .mul and nodeConstF32(prog, env, bin.b) != null) {
                        b.vecMulCF32(dst, ensureF32(b, resolveSlotReg(b, prog, info, env, hoisted_slots, bin.a), ta), nodeConstF32(prog, env, bin.b).?);
                    } else if (bin.op == .mul and nodeConstF32(prog, env, bin.a) != null) {
                        b.vecMulCF32(dst, ensureF32(b, resolveSlotReg(b, prog, info, env, hoisted_slots, bin.b), tb), nodeConstF32(prog, env, bin.a).?);
                    } else {
                        emitBinaryF32(b, bin.op, dst, ensureF32(b, resolveSlotReg(b, prog, info, env, hoisted_slots, bin.a), ta), ensureF32(b, resolveSlotReg(b, prog, info, env, hoisted_slots, bin.b), tb));
                    }
                    hoisted_slots[idx] = .{ .reg = dst, .owned = false };
                }
            },
            .ternary => |t| {
                const a_reg = resolveSlotReg(b, prog, info, env, hoisted_slots, t.a);
                const b_reg = resolveSlotReg(b, prog, info, env, hoisted_slots, t.b);
                const c_reg = resolveSlotReg(b, prog, info, env, hoisted_slots, t.c);
                const ta = info.value_types.items[t.a];
                const tb = info.value_types.items[t.b];
                const tc = info.value_types.items[t.c];
                const dst = b.newVec();
                if (out_ty == .i32 and t.op == .select and tb == .i32 and tc == .i32) {
                    const mask = b.newVec();
                    if (ta == .i32) b.vecOp3(.cmp_gt_i32, mask, a_reg, b.vecZero()) else b.vecOp3(.cmp_gt_f32, mask, a_reg, b.vecZero());
                    pixel.emitSelectByMask(b, dst, c_reg, b_reg, mask);
                } else {
                    emitTernaryF32(b, t.op, dst, ensureF32(b, a_reg, ta), ensureF32(b, b_reg, tb), ensureF32(b, c_reg, tc));
                }
                hoisted_slots[idx] = .{ .reg = dst, .owned = false };
            },
            else => unreachable,
        }

        if (need_hoisted_mask[idx] and out_ty == .f32) {
            if (hoisted_slots[idx]) |hs| {
                const m = b.newVec();
                b.vecOp3(.cmp_gt_f32, m, hs.reg, b.vecZero());
                hoisted_masks[idx] = m;
            }
        }
    }
}

fn evalDagVector(
    allocator: std.mem.Allocator,
    b: *aj.Builder,
    prog: *const ast.Program,
    info: *const analysis_mod.Analysis,
    spec: CompileSpec,
    env: BodyEnv,
    x_mode: XMode,
    col_disp: usize,
    use_const_mem: bool,
    preloaded: ?[]const ?aj.Reg,
) std.mem.Allocator.Error!aj.Reg {
    const res = try evalDagVectors(1, allocator, b, prog, info, spec, .{env}, x_mode, .{col_disp}, use_const_mem, .{preloaded});
    return res[0];
}

fn evalDagVectors(
    comptime count: usize,
    allocator: std.mem.Allocator,
    b: *aj.Builder,
    prog: *const ast.Program,
    info: *const analysis_mod.Analysis,
    spec: CompileSpec,
    envs: [count]BodyEnv,
    x_mode: XMode,
    col_disps: [count]usize,
    use_const_mem: bool,
    preloaded_arr: [count]?[]const ?aj.Reg,
) std.mem.Allocator.Error![count]aj.Reg {
    const n_nodes = prog.nodes.items.len;
    const env0 = envs[0];

    var slots_arr: [count][]?RegSlot = undefined;
    var slots_init: usize = 0;
    defer for (0..slots_init) |s| allocator.free(slots_arr[s]);

    inline for (0..count) |s| {
        const sl = try allocator.alloc(?RegSlot, n_nodes);
        slots_arr[s] = sl;
        slots_init = s + 1;
        if (envs[s].hoisted_slots.len == n_nodes) {
            @memcpy(sl, envs[s].hoisted_slots);
        } else {
            @memset(sl, null);
        }
        if (preloaded_arr[s]) |pre| {
            for (pre, 0..) |maybe_r, i| {
                if (maybe_r) |r| {
                    sl[i] = .{ .reg = r, .owned = true };
                }
            }
        }
    }

    if (slots_arr[0][prog.root] != null) {
        var fast_out: [count]aj.Reg = undefined;
        inline for (0..count) |s| {
            fast_out[s] = slots_arr[s][prog.root].?.reg;
        }
        return fast_out;
    }

    // Cache shared Cody-Waite trig range reduction per operand node (speeds up `x sin x cos +`).
    var trig_arr: [count][]?math_jit.TrigState = undefined;
    var trig_init: usize = 0;
    defer for (0..trig_init) |s| allocator.free(trig_arr[s]);
    inline for (0..count) |s| {
        const tc = try allocator.alloc(?math_jit.TrigState, n_nodes);
        @memset(tc, null);
        trig_arr[s] = tc;
        trig_init = s + 1;
    }

    // Cache deduplicated load_abs X byte offsets (`clamp(x) << shift`) and Y row offsets (`clamp(y) * stride`).
    const AbsXEntry = struct { node: ast.NodeIdx, mode: ast.BoundaryMode, shift: u8, reg: aj.Reg };
    const AbsYEntry = struct { node: ast.NodeIdx, mode: ast.BoundaryMode, clip: ast.ClipId, reg: aj.Reg };
    var abs_x_arr: [count]std.ArrayList(AbsXEntry) = @splat(.empty);
    defer inline for (0..count) |s| abs_x_arr[s].deinit(allocator);
    var abs_y_arr: [count]std.ArrayList(AbsYEntry) = @splat(.empty);
    defer inline for (0..count) |s| abs_y_arr[s].deinit(allocator);

    // Compute per-body remaining uses stopping at already-hoisted slots and compile-time constants
    // so children of hoisted nodes (e.g. `coord_x` under periodic `X % C`) are never emitted in the x loop.
    var rem_arr: [count][]u32 = undefined;
    var rem_init: usize = 0;
    defer for (0..rem_init) |s| allocator.free(rem_arr[s]);

    const rem_uses0 = try allocator.alloc(u32, n_nodes);
    rem_arr[0] = rem_uses0;
    rem_init = 1;
    @memset(rem_uses0, 0);
    rem_uses0[prog.root] = 1;
    var ri: usize = prog.root + 1;
    while (ri > 0) {
        ri -= 1;
        if (rem_uses0[ri] == 0) continue;
        if (slots_arr[0][ri] != null or nodeConstF32(prog, env0, @intCast(ri)) != null) continue;
        switch (prog.nodes.items[ri]) {
            .load_abs => |la| {
                rem_uses0[la.x] += 1;
                rem_uses0[la.y] += 1;
            },
            .unary => |u| rem_uses0[u.a] += 1,
            .binary => |bin| {
                rem_uses0[bin.a] += 1;
                rem_uses0[bin.b] += 1;
            },
            .ternary => |t| {
                rem_uses0[t.a] += 1;
                rem_uses0[t.b] += 1;
                rem_uses0[t.c] += 1;
            },
            else => {},
        }
    }
    inline for (1..count) |s| {
        const r = try allocator.alloc(u32, n_nodes);
        @memcpy(r, rem_uses0);
        rem_arr[s] = r;
        rem_init = s + 1;
    }

    const slots0 = slots_arr[0];

    // Identify single-use comparison / boolean logic trees directly feeding a `.select` so we fuse
    // `vcmpps/vpand/vpor/vpxor + vblendvps/vandps/vandnps` (f32) or `vpcmpgtd/vpcmpeqd + vpblendvb/vpand` (opt=1 i32).
    const fused_cmp = try allocator.alloc(bool, n_nodes);
    defer allocator.free(fused_cmp);
    @memset(fused_cmp, false);
    for (prog.nodes.items, 0..) |node, idx| {
        if (rem_uses0[idx] == 0 or slots0[idx] != null or nodeConstF32(prog, env0, @intCast(idx)) != null) continue;
        if (node == .ternary and node.ternary.op == .select) {
            const cond_idx = node.ternary.a;
            const out_t = info.value_types.items[idx];
            if (!prog.opt_int and out_t == .f32) {
                markFusableBoolTreeF32(prog, info, env0, slots0, fused_cmp, cond_idx);
            } else if (prog.opt_int and out_t == .i32 and
                slots0[cond_idx] == null and nodeConstF32(prog, env0, cond_idx) == null and
                info.use_counts.items[cond_idx] == 1 and prog.nodes.items[cond_idx] == .binary)
            {
                const cb = prog.nodes.items[cond_idx].binary;
                if (cmpBinOpToVecOp(cb.op) != null and
                    info.value_types.items[node.ternary.b] == .i32 and
                    info.value_types.items[node.ternary.c] == .i32 and
                    info.value_types.items[cb.a] == .i32 and
                    info.value_types.items[cb.b] == .i32)
                {
                    fused_cmp[cond_idx] = true;
                }
            }
        }
    }

    // On AVX2/AVX-512 interior loops, fold single-use native f32 `.load_clip`/`.load_rel`
    // directly into VEX/EVEX reg-reg-mem instructions (`vaddps`/`vsubps`/`vmulps`/`vminps`/`vmaxps`).
    // For non-commutative / NaN-order-sensitive ops (`sub`, `min`, `max`), only fold RHS (`bin.b`).
    const folded_load = try allocator.alloc(bool, n_nodes);
    defer allocator.free(folded_load);
    @memset(folded_load, false);
    const isNativeF32Load = struct {
        fn run(nodes: []const ast.Node, fmts: []const ast.SampleFormat, types: []const analysis_mod.ValueType, slot_arr: []const ?RegSlot, rem: []const u32, fused: []const bool, idx: ast.NodeIdx) bool {
            if (rem[idx] != 1 or slot_arr[idx] != null or fused[idx] or types[idx] != .f32) return false;
            return switch (nodes[idx]) {
                .load_clip => |clip| fmts[clip].is_float and fmts[clip].bits_per_sample == 32,
                .load_rel => |rel| fmts[rel.clip].is_float and fmts[rel.clip].bits_per_sample == 32,
                else => false,
            };
        }
    }.run;

    // Pin high-frequency constants (`use_counts >= 6`, e.g. `2.0` in Mandelbrot or `5.0`/`-3.0` in
    // Kirsch) in a vector register on non-gather loops instead of reloading `[rip+off]` repeatedly.
    const pin_hot_const = !info.uses_abs_access;

    if (b.cpu.vec_bytes >= 32 and x_mode == .interior) {
        // 1. Single-use native f32 loads folded into binary ops (e.g. box3x3, wide-reach).
        for (prog.nodes.items[0 .. prog.root + 1], 0..) |node, idx| {
            if (rem_uses0[idx] == 0 or slots0[idx] != null or fused_cmp[idx]) continue;
            if (nodeConstF32(prog, env0, @intCast(idx)) != null) continue;
            if (node != .binary or info.value_types.items[idx] != .f32) continue;
            const bin = node.binary;
            if (bin.a == bin.b) continue;
            const ca = if (use_const_mem and (!pin_hot_const or info.use_counts.items[bin.a] < 6)) nodeConstF32(prog, env0, bin.a) else null;
            const cb = if (use_const_mem and (!pin_hot_const or info.use_counts.items[bin.b] < 6)) nodeConstF32(prog, env0, bin.b) else null;
            if (ca != null or cb != null) continue;
            const is_comm = switch (bin.op) {
                .add, .mul => true,
                .sub, .min, .max => false,
                else => continue,
            };
            if (!folded_load[bin.b] and isNativeF32Load(prog.nodes.items, spec.src_fmts, info.value_types.items, slots0, rem_uses0, fused_cmp, bin.b)) {
                folded_load[bin.b] = true;
            } else if (is_comm and !folded_load[bin.a] and isNativeF32Load(prog.nodes.items, spec.src_fmts, info.value_types.items, slots0, rem_uses0, fused_cmp, bin.a)) {
                folded_load[bin.a] = true;
            }
        }
    }

    // 2. High-register-pressure multi-use native f32 loads (e.g. 5x5 Kirsch with 24 shared pixels):
    //    on AVX2/AVX-512 folds each L1D load into VEX/EVEX mem-operand ALU; on AArch64 NEON
    //    rematerializes fresh `ldr q` from L1D at each use instead of spilling 24 loads to `[sp]`.
    if ((b.cpu.vec_bytes >= 32 or b.cpu.isa == .neon) and x_mode == .interior and b.cpu.has_fma and info.live_node_count > 48) {
        const isAnyNativeF32Load = struct {
            fn run(nodes: []const ast.Node, fmts: []const ast.SampleFormat, types: []const analysis_mod.ValueType, slot_arr: []const ?RegSlot, rem: []const u32, fused: []const bool, idx: ast.NodeIdx) bool {
                if (rem[idx] == 0 or slot_arr[idx] != null or fused[idx] or types[idx] != .f32) return false;
                return switch (nodes[idx]) {
                    .load_clip => |clip| fmts[clip].is_float and fmts[clip].bits_per_sample == 32,
                    .load_rel => |rel| fmts[rel.clip].is_float and fmts[rel.clip].bits_per_sample == 32,
                    else => false,
                };
            }
        }.run;

        var live_f32_loads: usize = 0;
        for (0..prog.root + 1) |idx| {
            if (!folded_load[idx] and isAnyNativeF32Load(prog.nodes.items, spec.src_fmts, info.value_types.items, slots0, rem_uses0, fused_cmp, @intCast(idx))) {
                live_f32_loads += 1;
            }
        }
        if (live_f32_loads > 12) {
            const can_remat = try allocator.alloc(bool, n_nodes);
            defer allocator.free(can_remat);
            @memset(can_remat, false);
            for (0..prog.root + 1) |idx| {
                if (!folded_load[idx] and isAnyNativeF32Load(prog.nodes.items, spec.src_fmts, info.value_types.items, slots0, rem_uses0, fused_cmp, @intCast(idx))) {
                    can_remat[idx] = true;
                }
            }
            can_remat[prog.root] = false;

            for (prog.nodes.items[0 .. prog.root + 1], 0..) |node, idx| {
                if (rem_uses0[idx] == 0 or slots0[idx] != null or folded_load[idx]) continue;
                if (nodeConstF32(prog, env0, @intCast(idx)) != null) continue;
                switch (node) {
                    .load_abs => |la| {
                        can_remat[la.x] = false;
                        can_remat[la.y] = false;
                    },
                    .unary => |u| {
                        can_remat[u.a] = false;
                    },
                    .binary => |bin| {
                        const ok_op = info.value_types.items[idx] == .f32 and !fused_cmp[idx] and
                            (bin.op == .add or bin.op == .mul or bin.op == .sub or bin.op == .min or bin.op == .max);
                        if (!ok_op or bin.a == bin.b) {
                            can_remat[bin.a] = false;
                            can_remat[bin.b] = false;
                        } else {
                            if (bin.op == .sub or bin.op == .min or bin.op == .max) {
                                can_remat[bin.a] = false;
                            }
                            // Never fold both operands of a single binary op simultaneously.
                            if (can_remat[bin.a] and can_remat[bin.b]) {
                                can_remat[bin.a] = false;
                            }
                        }
                    },
                    .ternary => |t| {
                        can_remat[t.c] = false;
                        const ok_fma = info.value_types.items[idx] == .f32 and (t.op == .fmadd or t.op == .fnmadd);
                        if (!ok_fma or t.a == t.b or t.a == t.c or t.b == t.c) {
                            can_remat[t.a] = false;
                            can_remat[t.b] = false;
                        } else if (can_remat[t.a] and can_remat[t.b]) {
                            can_remat[t.a] = false;
                        }
                    },
                    else => {},
                }
            }
            for (0..prog.root + 1) |idx| {
                if (can_remat[idx]) folded_load[idx] = true;
            }
        }
    }

    // Iterative Sethi-Ullman / minimum-register-pressure topological schedule for high-pressure
    // non-gather interior stencils (e.g. 5x5 Kirsch where RPN peak live > 10). Narrow serial chains
    // (like Mandelbrot, peak_live <= 6) and gather-heavy loops keep RPN order.
    var schedule: std.ArrayList(ast.NodeIdx) = .empty;
    defer schedule.deinit(allocator);
    try schedule.ensureTotalCapacity(allocator, info.live_node_count + 1);

    const use_su_schedule = count == 1 and x_mode == .interior and !info.uses_abs_access and
        info.live_node_count > 40 and env0.rpn_peak_live > 10;
    if (!use_su_schedule) {
        for (0..prog.root + 1) |i| {
            const idx: ast.NodeIdx = @intCast(i);
            if (rem_uses0[idx] == 0 or slots0[idx] != null or fused_cmp[idx] or folded_load[idx]) continue;
            if (nodeConstF32(prog, env0, idx) != null) continue;
            try schedule.append(allocator, idx);
        }
    } else {
        const su = try allocator.alloc(u32, n_nodes);
        defer allocator.free(su);
        @memset(su, 0);
        for (prog.nodes.items[0 .. prog.root + 1], 0..) |node, idx| {
            if (rem_uses0[idx] == 0) continue;
            if (slots0[idx] != null or folded_load[idx] or nodeConstF32(prog, env0, @intCast(idx)) != null) {
                su[idx] = 0;
                continue;
            }
            su[idx] = switch (node) {
                .load_abs => |la| blk: {
                    const s0 = su[la.x];
                    const s1 = su[la.y];
                    break :blk @max(@as(u32, 2), if (s0 == s1) s0 + 1 else @max(s0, s1));
                },
                .unary => |u| @max(@as(u32, 1), su[u.a]),
                .binary => |bin| blk: {
                    const s0 = su[bin.a];
                    const s1 = su[bin.b];
                    break :blk @max(@as(u32, 1), if (s0 == s1) s0 + 1 else @max(s0, s1));
                },
                .ternary => |t| blk: {
                    var arr = [3]u32{ su[t.a], su[t.b], su[t.c] };
                    if (arr[0] < arr[1]) std.mem.swap(u32, &arr[0], &arr[1]);
                    if (arr[1] < arr[2]) std.mem.swap(u32, &arr[1], &arr[2]);
                    if (arr[0] < arr[1]) std.mem.swap(u32, &arr[0], &arr[1]);
                    break :blk @max(@as(u32, 1), @max(arr[0], @max(arr[1] + 1, arr[2] + 2)));
                },
                else => 1,
            };
        }

        const VisState = enum(u8) { unvisited, expanded, emitted };
        const vis = try allocator.alloc(VisState, n_nodes);
        defer allocator.free(vis);
        @memset(vis, .unvisited);

        const Frame = struct { idx: ast.NodeIdx, post: bool };
        var dfs_stack: std.ArrayList(Frame) = .empty;
        defer dfs_stack.deinit(allocator);
        var dep_buf: std.ArrayList(ast.NodeIdx) = .empty;
        defer dep_buf.deinit(allocator);
        var fused_walk: std.ArrayList(ast.NodeIdx) = .empty;
        defer fused_walk.deinit(allocator);

        const needsEmit = struct {
            fn run(p: *const ast.Program, e: BodyEnv, rem: []const u32, sl: []const ?RegSlot, fc: []const bool, fl: []const bool, idx: ast.NodeIdx) bool {
                if (rem[idx] == 0 or sl[idx] != null or fc[idx] or fl[idx]) return false;
                return nodeConstF32(p, e, idx) == null;
            }
        }.run;

        if (needsEmit(prog, env0, rem_uses0, slots0, fused_cmp, folded_load, prog.root)) {
            try dfs_stack.append(allocator, .{ .idx = prog.root, .post = false });
        }
        while (dfs_stack.items.len > 0) {
            const top = dfs_stack.pop().?;
            const idx = top.idx;
            if (top.post) {
                if (vis[idx] != .emitted) {
                    vis[idx] = .emitted;
                    try schedule.append(allocator, idx);
                }
                continue;
            }
            if (vis[idx] != .unvisited) continue;
            vis[idx] = .expanded;
            try dfs_stack.append(allocator, .{ .idx = idx, .post = true });

            dep_buf.clearRetainingCapacity();
            const node = prog.nodes.items[idx];
            switch (node) {
                .load_abs => |la| {
                    try dep_buf.append(allocator, la.x);
                    try dep_buf.append(allocator, la.y);
                },
                .unary => |u| try dep_buf.append(allocator, u.a),
                .binary => |bin| {
                    if (!folded_load[bin.a]) try dep_buf.append(allocator, bin.a);
                    if (!folded_load[bin.b]) try dep_buf.append(allocator, bin.b);
                },
                .ternary => |t| {
                    if (t.op == .select and fused_cmp[t.a]) {
                        fused_walk.clearRetainingCapacity();
                        try fused_walk.append(allocator, t.a);
                        while (fused_walk.items.len > 0) {
                            const fidx = fused_walk.pop().?;
                            if (!fused_cmp[fidx]) {
                                try dep_buf.append(allocator, fidx);
                                continue;
                            }
                            switch (prog.nodes.items[fidx]) {
                                .unary => |fu| try fused_walk.append(allocator, fu.a),
                                .binary => |fbin| {
                                    try fused_walk.append(allocator, fbin.a);
                                    try fused_walk.append(allocator, fbin.b);
                                },
                                else => {},
                            }
                        }
                    } else {
                        if (!folded_load[t.a]) try dep_buf.append(allocator, t.a);
                    }
                    if (!folded_load[t.b]) try dep_buf.append(allocator, t.b);
                    if (!folded_load[t.c]) try dep_buf.append(allocator, t.c);
                },
                else => {},
            }

            var write_i: usize = 0;
            for (dep_buf.items) |d| {
                if (vis[d] == .unvisited and needsEmit(prog, env0, rem_uses0, slots0, fused_cmp, folded_load, d)) {
                    dep_buf.items[write_i] = d;
                    write_i += 1;
                }
            }
            const deps_slice = dep_buf.items[0..write_i];
            var si: usize = 1;
            while (si < deps_slice.len) : (si += 1) {
                const key = deps_slice[si];
                const key_su = su[key];
                var sj: usize = si;
                while (sj > 0 and su[deps_slice[sj - 1]] > key_su) : (sj -= 1) {
                    deps_slice[sj] = deps_slice[sj - 1];
                }
                deps_slice[sj] = key;
            }
            for (deps_slice) |d| {
                try dfs_stack.append(allocator, .{ .idx = d, .post = false });
            }
        }
    }

    const releaseUse = struct {
        fn run(rem: []u32, slot_arr: []?RegSlot, idx: ast.NodeIdx) void {
            if (rem[idx] == 0) return;
            rem[idx] -= 1;
            if (rem[idx] == 0) {
                if (slot_arr[idx]) |s| {
                    if (s.owned) {
                        slot_arr[idx] = null;
                    }
                }
            }
        }
    }.run;

    const ensureF32 = struct {
        fn run(builder: *aj.Builder, reg: aj.Reg, ty: analysis_mod.ValueType) aj.Reg {
            if (ty == .f32) return reg;
            const dst = builder.newVec();
            builder.vecOp2(.cvt_i32_to_f32, dst, reg);
            return dst;
        }
    }.run;

    const loadMemF32Interior = struct {
        fn run(node: ast.Node, w: i32, disp_px: usize, e: BodyEnv) aj.Mem {
            return switch (node) {
                .load_clip => |clip| e.rowMem(e.src_row_ptrs[clip].?, 2, @intCast(disp_px * 4)),
                .load_rel => |rel| blk: {
                    const row_ptr = e.rowPtrFor(rel.clip, rel.dy, rel.mode);
                    const eff_dx: i32 = interp.clampRelOffset(rel.dx, w);
                    const disp: i32 = (eff_dx + @as(i32, @intCast(disp_px))) * 4;
                    break :blk e.rowMem(row_ptr, 2, disp);
                },
                else => unreachable,
            };
        }
    }.run;

    const max_plane_bytes_f32: u64 = @as(u64, spec.height) * std.mem.alignForward(u64, @as(u64, spec.width) * 4 + 64, 64);
    const can_hw_gather_f32 = max_plane_bytes_f32 <= 0x7FFF_0000;
    const w_i32_spec: i32 = @intCast(spec.width);

    for (schedule.items) |idx| {
        const node = prog.nodes.items[idx];
        const out_ty = info.value_types.items[idx];

        // On AArch64 NEON 2x-interleaved interior loops, pair contiguous `.load_clip`/`.load_rel`
        // loads into a single 8B/16B/32B transfer (`ldr d/q + ushll/ushll2`, `fcvtl/fcvtl2`, or `ldp`).
        if (count == 2 and b.cpu.isa == .neon and x_mode == .interior and
            col_disps[1] == col_disps[0] + b.cpu.f32_lanes and
            rem_arr[0][idx] > 0 and slots_arr[0][idx] == null and slots_arr[1][idx] == null and
            !fused_cmp[idx] and !folded_load[idx] and
            (node == .load_clip or node == .load_rel))
        {
            const clip: ast.ClipId = if (node == .load_clip) node.load_clip else node.load_rel.clip;
            const fmt = spec.src_fmts[clip];
            const bps: usize = fmt.bytesPerSample();
            const shift: u8 = switch (bps) {
                1 => 0,
                2 => 1,
                else => 2,
            };
            const mem0 = switch (node) {
                .load_clip => envs[0].rowMem(envs[0].src_row_ptrs[clip].?, shift, @intCast(col_disps[0] * bps)),
                .load_rel => |rel| blk: {
                    const row_ptr = envs[0].rowPtrFor(rel.clip, rel.dy, rel.mode);
                    const eff_dx: i32 = interp.clampRelOffset(rel.dx, w_i32_spec);
                    break :blk envs[0].rowMem(row_ptr, shift, (eff_dx + @as(i32, @intCast(col_disps[0]))) * @as(i32, @intCast(bps)));
                },
                else => unreachable,
            };
            const r0 = b.newVec();
            const r1 = b.newVec();
            if (fmt.is_float) {
                if (fmt.bits_per_sample == 16) {
                    b.vecCvtF16ToF32x2(r0, r1, mem0);
                } else {
                    b.vecLoadUVecx2(r0, r1, mem0);
                }
            } else if (fmt.bits_per_sample <= 16 and out_ty == .f32) {
                b.vecLoadUintToF32x2(r0, r1, mem0, fmt.bytesPerSample());
            } else {
                if (fmt.bits_per_sample <= 8) {
                    b.vecLoadU8ToU32x2(r0, r1, mem0);
                } else if (fmt.bits_per_sample <= 16) {
                    b.vecLoadU16ToU32x2(r0, r1, mem0);
                } else {
                    b.vecLoadUVecx2(r0, r1, mem0);
                }
                if (out_ty == .f32) {
                    b.vecOp2(.cvt_i32_to_f32, r0, r0);
                    b.vecOp2(.cvt_i32_to_f32, r1, r1);
                }
            }
            slots_arr[0][idx] = .{ .reg = r0, .owned = true };
            slots_arr[1][idx] = .{ .reg = r1, .owned = true };
            continue;
        }

        for (0..count) |s| {
            const env = envs[s];
            const slots = slots_arr[s];
            const rem_uses = rem_arr[s];
            const col_disp = col_disps[s];
            const trig_cache = trig_arr[s];
            const abs_x_cache = &abs_x_arr[s];
            const abs_y_cache = &abs_y_arr[s];
            if (rem_uses[idx] == 0) continue;
            if (slots[idx] != null) continue;
            if (fused_cmp[idx] or folded_load[idx]) continue;
            if (nodeConstF32(prog, env, idx) != null) continue;

            switch (node) {
                .const_f32 => {},
                .const_i32 => |v| {
                    if (out_ty == .i32) {
                        slots[idx] = .{ .reg = b.constI32Splat(v), .owned = false };
                    }
                },
                .load_clip => |clip| {
                    const dst = b.newVec();
                    const fmt = spec.src_fmts[clip];
                    const bps: usize = fmt.bytesPerSample();
                    const shift: u8 = switch (bps) {
                        1 => 0,
                        2 => 1,
                        else => 2,
                    };
                    const mem = switch (x_mode) {
                        .const_col => |col| aj.Mem.ptr(env.src_row_ptrs[clip].?, @intCast((col + col_disp) * bps)),
                        .interior => env.rowMem(env.src_row_ptrs[clip].?, shift, @intCast(col_disp * bps)),
                        .dynamic_left, .dynamic_right => aj.Mem.indexed(env.src_row_ptrs[clip].?, env.x_gp, shift, @intCast(col_disp * bps)),
                    };
                    if (out_ty == .i32) {
                        pixel.emitLoadI32(b, dst, mem, fmt);
                    } else {
                        pixel.emitLoadF32(b, dst, mem, fmt);
                    }
                    slots[idx] = .{ .reg = dst, .owned = true };
                },
                .load_rel => |rel| {
                    const dst = b.newVec();
                    emitRelLoad(b, dst, out_ty, spec, env, x_mode, col_disp, rel.clip, rel.dx, rel.dy, rel.mode);
                    slots[idx] = .{ .reg = dst, .owned = true };
                },
                .load_abs => |la| {
                    const fmt = spec.src_fmts[la.clip];
                    const bps: usize = fmt.bytesPerSample();
                    const shift: u8 = switch (bps) {
                        1 => 0,
                        2 => 1,
                        else => 2,
                    };

                    // Fast contiguous vector load for `X const_y x[]` and `X Y x[]` (zero gather).
                    const direct_row: ?aj.Reg = if (x_mode == .interior or x_mode == .const_col) blk: {
                        if (env.abs_row_ptrs.len == n_nodes) {
                            if (env.abs_row_ptrs[idx]) |rp| break :blk rp;
                        }
                        if (prog.nodes.items[la.x] == .coord_x and prog.nodes.items[la.y] == .coord_y and
                            spec.width <= (1 << 24) and spec.height <= (1 << 24))
                        {
                            break :blk env.src_row_ptrs[la.clip].?;
                        }
                        break :blk null;
                    } else null;

                    if (direct_row) |rptr| {
                        releaseUse(rem_uses, slots, la.x);
                        releaseUse(rem_uses, slots, la.y);
                        const mem = switch (x_mode) {
                            .const_col => |col| aj.Mem.ptr(rptr, @intCast((col + col_disp) * bps)),
                            .interior => env.rowMem(rptr, shift, @intCast(col_disp * bps)),
                            .dynamic_left, .dynamic_right => unreachable,
                        };
                        const dst = b.newVec();
                        if (out_ty == .i32) {
                            pixel.emitLoadI32(b, dst, mem, fmt);
                        } else {
                            pixel.emitLoadF32(b, dst, mem, fmt);
                        }
                        slots[idx] = .{ .reg = dst, .owned = true };
                        continue;
                    }

                    var ix_reg: ?aj.Reg = null;
                    for (abs_x_cache.items) |e| {
                        if (e.node == la.x and e.mode == la.mode and e.shift == shift) {
                            ix_reg = e.reg;
                            break;
                        }
                    }
                    if (ix_reg == null) {
                        const vx_raw = resolveSlotReg(b, prog, info, env, slots, la.x);
                        const tx = info.value_types.items[la.x];
                        const fx = ensureF32(b, vx_raw, tx);
                        const ix = b.newVec();
                        emitAbsCoordToI32(b, ix, fx, @intCast(spec.width), la.mode);
                        if (shift != 0) b.vecOp2i(.sll_u32, ix, ix, shift);
                        try abs_x_cache.append(allocator, .{ .node = la.x, .mode = la.mode, .shift = shift, .reg = ix });
                        ix_reg = ix;
                    }

                    var iy_reg: ?aj.Reg = null;
                    for (abs_y_cache.items) |e| {
                        if (e.node == la.y and e.mode == la.mode and e.clip == la.clip) {
                            iy_reg = e.reg;
                            break;
                        }
                    }
                    if (iy_reg == null) {
                        const vy_raw = resolveSlotReg(b, prog, info, env, slots, la.y);
                        const ty = info.value_types.items[la.y];
                        const fy = ensureF32(b, vy_raw, ty);
                        const iy = b.newVec();
                        emitAbsCoordToI32(b, iy, fy, @intCast(spec.height), la.mode);
                        b.vecOp3(.mul_i32, iy, iy, env.src_stride_vecs[la.clip].?);
                        try abs_y_cache.append(allocator, .{ .node = la.y, .mode = la.mode, .clip = la.clip, .reg = iy });
                        iy_reg = iy;
                    }

                    releaseUse(rem_uses, slots, la.x);
                    releaseUse(rem_uses, slots, la.y);

                    const byte_off = b.newVec();
                    b.vecOp3(.add_i32, byte_off, iy_reg.?, ix_reg.?);

                    const dst = b.newVec();
                    emitGatherFromBase(b, dst, out_ty, env.base_src_ptrs[la.clip].?, byte_off, fmt, can_hw_gather_f32);
                    slots[idx] = .{ .reg = dst, .owned = true };
                },
                .load_prop => |pid| {
                    slots[idx] = .{ .reg = env.prop_regs[pid], .owned = false };
                },
                .coord_x => {
                    slots[idx] = .{ .reg = env.v_x.?, .owned = false };
                },
                .coord_y => {
                    slots[idx] = .{ .reg = env.v_y.?, .owned = false };
                },
                .frame_n => {
                    slots[idx] = .{ .reg = if (out_ty == .i32) env.v_frame_n_i32.? else env.v_frame_n_f32.?, .owned = false };
                },
                .plane_width => {
                    slots[idx] = .{
                        .reg = if (out_ty == .i32) b.constI32Splat(@intCast(spec.width)) else b.constF32Splat(@floatFromInt(spec.width)),
                        .owned = false,
                    };
                },
                .plane_height => {
                    slots[idx] = .{
                        .reg = if (out_ty == .i32) b.constI32Splat(@intCast(spec.height)) else b.constF32Splat(@floatFromInt(spec.height)),
                        .owned = false,
                    };
                },
                .unary => |u| {
                    const src_reg = resolveSlotReg(b, prog, info, env, slots, u.a);
                    const src_ty = info.value_types.items[u.a];
                    const dst = b.newVec();

                    if (u.op == .to_f32) {
                        if (src_ty == .i32) {
                            b.vecOp2(.cvt_i32_to_f32, dst, src_reg);
                        } else {
                            b.vecOp2(.mov, dst, src_reg);
                        }
                    } else if (out_ty == .i32 and src_ty == .i32) {
                        emitUnaryI32(b, u.op, dst, src_reg);
                    } else if (out_ty == .i32) {
                        // V7: `not` / `bitnot` on `.f32` operand producing `.i32` under `opt=1`.
                        emitUnaryF32ToI32(b, u.op, dst, src_reg);
                    } else {
                        const a_f = ensureF32(b, src_reg, src_ty);
                        switch (u.op) {
                            .sin, .cos => {
                                const st = trig_cache[u.a] orelse blk: {
                                    const prepared = math_jit.emitTrigPrepare(b, a_f);
                                    trig_cache[u.a] = prepared;
                                    break :blk prepared;
                                };
                                switch (u.op) {
                                    .sin => math_jit.emitSinFromState(b, dst, st),
                                    .cos => math_jit.emitCosFromState(b, dst, st),
                                    else => unreachable,
                                }
                            },
                            else => emitUnaryF32(b, u.op, dst, a_f, use_const_mem),
                        }
                    }
                    releaseUse(rem_uses, slots, u.a);
                    slots[idx] = .{ .reg = dst, .owned = true };
                },
                .binary => |bin| {
                    const ta = info.value_types.items[bin.a];
                    const tb = info.value_types.items[bin.b];
                    const dst = b.newVec();

                    if (out_ty == .i32 and ta == .i32 and tb == .i32) {
                        const a_reg = resolveSlotReg(b, prog, info, env, slots, bin.a);
                        const b_reg = resolveSlotReg(b, prog, info, env, slots, bin.b);
                        if (bin.op == .mul) {
                            if (prog.nodes.items[bin.b] == .const_i32 and prog.nodes.items[bin.b].const_i32 > 0) {
                                const iv: u32 = @intCast(prog.nodes.items[bin.b].const_i32);
                                if (std.math.isPowerOfTwo(iv)) {
                                    const sh: u5 = @intCast(@ctz(iv));
                                    if (sh == 0) b.vecOp2(.mov, dst, a_reg) else if (sh == 1) b.vecOp3(.add_i32, dst, a_reg, a_reg) else b.vecOp2i(.sll_u32, dst, a_reg, sh);
                                } else {
                                    b.vecOp3(.mul_i32, dst, a_reg, b_reg);
                                }
                            } else if (prog.nodes.items[bin.a] == .const_i32 and prog.nodes.items[bin.a].const_i32 > 0) {
                                const iv: u32 = @intCast(prog.nodes.items[bin.a].const_i32);
                                if (std.math.isPowerOfTwo(iv)) {
                                    const sh: u5 = @intCast(@ctz(iv));
                                    if (sh == 0) b.vecOp2(.mov, dst, b_reg) else if (sh == 1) b.vecOp3(.add_i32, dst, b_reg, b_reg) else b.vecOp2i(.sll_u32, dst, b_reg, sh);
                                } else {
                                    b.vecOp3(.mul_i32, dst, a_reg, b_reg);
                                }
                            } else {
                                b.vecOp3(.mul_i32, dst, a_reg, b_reg);
                            }
                        } else {
                            emitBinaryI32(b, bin.op, dst, a_reg, b_reg);
                        }
                    } else if (out_ty == .i32) {
                        // V7: comparisons, logic, and bitwise ops on float/mixed operands producing `.i32` under `opt=1`.
                        const af = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, bin.a), ta);
                        const bf = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, bin.b), tb);
                        emitBinaryF32ToI32(b, bin.op, dst, af, bf);
                    } else {
                        var handled = false;
                        // Bit-exact IEEE-754 strength reduction: `x * 2.0 -> x + x` (`vaddps` runs on Zn3FP2/3
                        // instead of saturated Zn3FP0/1 and needs no constant load/register).
                        if (bin.op == .mul) {
                            const c_rhs = if (!folded_load[bin.a]) nodeConstF32(prog, env, bin.b) else null;
                            const c_lhs = if (!folded_load[bin.b]) nodeConstF32(prog, env, bin.a) else null;
                            if (c_rhs != null and c_rhs.? == 2.0) {
                                const af = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, bin.a), ta);
                                b.vecOp3(.add_f32, dst, af, af);
                                handled = true;
                            } else if (c_lhs != null and c_lhs.? == 2.0) {
                                const bf = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, bin.b), tb);
                                b.vecOp3(.add_f32, dst, bf, bf);
                                handled = true;
                            }
                        }
                        // Pin high-frequency constants (`use_counts >= 6`, e.g. `2.0` in Mandelbrot) in a
                        // vector register on low-pressure chains instead of reloading `[rip+off]` repeatedly.
                        const ca = if (!handled and use_const_mem and !folded_load[bin.b] and (!pin_hot_const or info.use_counts.items[bin.a] < 6)) nodeConstF32(prog, env, bin.a) else null;
                        const cb = if (!handled and use_const_mem and !folded_load[bin.a] and (!pin_hot_const or info.use_counts.items[bin.b] < 6)) nodeConstF32(prog, env, bin.b) else null;
                        if (cb) |cv| {
                            switch (bin.op) {
                                .add => {
                                    b.vecAddCF32(dst, ensureF32(b, resolveSlotReg(b, prog, info, env, slots, bin.a), ta), cv);
                                    handled = true;
                                },
                                .sub => {
                                    b.vecSubCF32(dst, ensureF32(b, resolveSlotReg(b, prog, info, env, slots, bin.a), ta), cv);
                                    handled = true;
                                },
                                .mul => {
                                    b.vecMulCF32(dst, ensureF32(b, resolveSlotReg(b, prog, info, env, slots, bin.a), ta), cv);
                                    handled = true;
                                },
                                .min => {
                                    b.vecMinCF32(dst, ensureF32(b, resolveSlotReg(b, prog, info, env, slots, bin.a), ta), cv);
                                    handled = true;
                                },
                                .max => {
                                    b.vecMaxCF32(dst, ensureF32(b, resolveSlotReg(b, prog, info, env, slots, bin.a), ta), cv);
                                    handled = true;
                                },
                                else => {},
                            }
                        }
                        if (!handled and ca != null) {
                            const cv = ca.?;
                            switch (bin.op) {
                                .add => {
                                    b.vecAddCF32(dst, ensureF32(b, resolveSlotReg(b, prog, info, env, slots, bin.b), tb), cv);
                                    handled = true;
                                },
                                .mul => {
                                    b.vecMulCF32(dst, ensureF32(b, resolveSlotReg(b, prog, info, env, slots, bin.b), tb), cv);
                                    handled = true;
                                },
                                else => {},
                            }
                        }
                        if (!handled and bin.op == .mod) {
                            if (nodeConstF32(prog, env, bin.b)) |mod_c| {
                                if (isExactPow2Divisor(mod_c)) {
                                    emitModPow2F32(b, dst, ensureF32(b, resolveSlotReg(b, prog, info, env, slots, bin.a), ta), mod_c);
                                    handled = true;
                                }
                            }
                        }
                        if (!handled and (folded_load[bin.b] or folded_load[bin.a])) {
                            const vop: aj.VecOp3 = switch (bin.op) {
                                .add => .add_f32,
                                .sub => .sub_f32,
                                .mul => .mul_f32,
                                .min => .min_f32,
                                .max => .max_f32,
                                else => unreachable,
                            };
                            if (folded_load[bin.b]) {
                                const af = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, bin.a), ta);
                                const mem = loadMemF32Interior(prog.nodes.items[bin.b], w_i32_spec, col_disp, env);
                                b.vecOp3Mem(vop, dst, af, mem);
                            } else {
                                const bf = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, bin.b), tb);
                                const mem = loadMemF32Interior(prog.nodes.items[bin.a], w_i32_spec, col_disp, env);
                                b.vecOp3Mem(vop, dst, bf, mem);
                            }
                            handled = true;
                        }
                        if (!handled) {
                            const af = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, bin.a), ta);
                            const bf = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, bin.b), tb);
                            emitBinaryF32(b, bin.op, dst, af, bf);
                        }
                    }
                    releaseUse(rem_uses, slots, bin.a);
                    releaseUse(rem_uses, slots, bin.b);
                    slots[idx] = .{ .reg = dst, .owned = true };
                },
                .ternary => |t| {
                    const ta = info.value_types.items[t.a];
                    const tb = info.value_types.items[t.b];
                    const tc = info.value_types.items[t.c];
                    const soleHere = struct {
                        fn run(rem: []const u32, a_idx: ast.NodeIdx, b_idx: ast.NodeIdx, c_idx: ast.NodeIdx, target: ast.NodeIdx) bool {
                            var n: u32 = 0;
                            if (a_idx == target) n += 1;
                            if (b_idx == target) n += 1;
                            if (c_idx == target) n += 1;
                            return rem[target] == n;
                        }
                    }.run;
                    // Reuse single-use owned FMA operand in-place (`vfmadd231ps`/`vfmsub231ps`/`vfnmadd231ps`
                    // on `t.c`, and `vfmadd213ps`/`vfmsub213ps` on `t.a`/`t.b` on x86 FMA) to eliminate
                    // `vmovdqa`/`vmovaps` copies.
                    const is_fma = t.op == .fmadd or t.op == .fmsub or t.op == .fnmadd or t.op == .fnmsub;
                    const c_owned = if (slots[t.c]) |sc| sc.owned else false;
                    const reuse_c = is_fma and tc == .f32 and c_owned and soleHere(rem_uses, t.a, t.b, t.c, t.c);
                    var dst: aj.Reg = undefined;

                    if (out_ty == .i32 and t.op == .select and tb == .i32 and tc == .i32) {
                        dst = b.newVec();
                        if (fused_cmp[t.a]) {
                            const cb = prog.nodes.items[t.a].binary;
                            const lhs_reg = resolveSlotReg(b, prog, info, env, slots, cb.a);
                            const rhs_reg = resolveSlotReg(b, prog, info, env, slots, cb.b);
                            const mask = b.newVec();
                            var invert = false;
                            switch (cb.op) {
                                .gt => b.vecOp3(.cmp_gt_i32, mask, lhs_reg, rhs_reg),
                                .lt => b.vecOp3(.cmp_gt_i32, mask, rhs_reg, lhs_reg),
                                .eq => b.vecOp3(.cmp_eq_i32, mask, lhs_reg, rhs_reg),
                                .le => {
                                    b.vecOp3(.cmp_gt_i32, mask, lhs_reg, rhs_reg);
                                    invert = true;
                                },
                                .ge => {
                                    b.vecOp3(.cmp_gt_i32, mask, rhs_reg, lhs_reg);
                                    invert = true;
                                },
                                else => unreachable,
                            }
                            releaseUse(rem_uses, slots, cb.a);
                            releaseUse(rem_uses, slots, cb.b);

                            const t_idx = if (invert) t.c else t.b;
                            const f_idx = if (invert) t.b else t.c;
                            const f_is_zero = prog.nodes.items[f_idx] == .const_i32 and prog.nodes.items[f_idx].const_i32 == 0;
                            const t_is_zero = prog.nodes.items[t_idx] == .const_i32 and prog.nodes.items[t_idx].const_i32 == 0;
                            if (f_is_zero) {
                                const t_reg = resolveSlotReg(b, prog, info, env, slots, t_idx);
                                b.vecOp3(.and_u32, dst, t_reg, mask);
                            } else if (t_is_zero) {
                                const f_reg = resolveSlotReg(b, prog, info, env, slots, f_idx);
                                b.vecOp3(.andn_u32, dst, mask, f_reg);
                            } else {
                                const t_reg = resolveSlotReg(b, prog, info, env, slots, t_idx);
                                const f_reg = resolveSlotReg(b, prog, info, env, slots, f_idx);
                                if (b.cpu.prefersMaskDstReuse()) dst = mask;
                                pixel.emitSelectByMask(b, dst, f_reg, t_reg, mask);
                            }
                        } else {
                            const a_reg = resolveSlotReg(b, prog, info, env, slots, t.a);
                            const b_reg = resolveSlotReg(b, prog, info, env, slots, t.b);
                            const c_reg = resolveSlotReg(b, prog, info, env, slots, t.c);
                            const mask = if (b.cpu.prefersMaskDstReuse()) dst else b.newVec();
                            if (ta == .i32) {
                                b.vecOp3(.cmp_gt_i32, mask, a_reg, b.vecZero());
                            } else {
                                b.vecOp3(.cmp_gt_f32, mask, a_reg, b.vecZero());
                            }
                            pixel.emitSelectByMask(b, dst, c_reg, b_reg, mask);
                        }
                    } else if (t.op == .select and fused_cmp[t.a]) {
                        // Fused `(cmp / and / or / xor / not tree) ? true_val : false_val` in FP domain!
                        const mask = emitBoolMaskF32(b, prog, info, env, slots, rem_uses, fused_cmp, use_const_mem, t.a);

                        const c_zero = if (nodeConstF32(prog, env, t.c)) |cv| isPosZeroF32(cv) else false;
                        const b_zero = if (nodeConstF32(prog, env, t.b)) |bv| isPosZeroF32(bv) else false;
                        if (c_zero) {
                            dst = b.newVec();
                            const bf = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, t.b), tb);
                            b.vecAndF32(dst, bf, mask);
                        } else if (b_zero) {
                            dst = b.newVec();
                            const cf = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, t.c), tc);
                            b.vecAndnF32(dst, mask, cf);
                        } else {
                            dst = if (b.cpu.prefersMaskDstReuse()) mask else b.newVec();
                            const bf = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, t.b), tb);
                            const cf = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, t.c), tc);
                            pixel.emitSelectF32ByMask(b, dst, cf, bf, mask);
                        }
                    } else {
                        var handled = false;
                        if ((t.op == .fmadd or t.op == .fnmadd) and (folded_load[t.a] or folded_load[t.b])) {
                            dst = if (reuse_c) slots[t.c].?.reg else b.newVec();
                            const cf = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, t.c), tc);
                            if (folded_load[t.a]) {
                                const bf = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, t.b), tb);
                                const mem = loadMemF32Interior(prog.nodes.items[t.a], w_i32_spec, col_disp, env);
                                if (t.op == .fmadd) b.vecFmaMemF32(dst, bf, mem, cf) else b.vecFnmaMemF32(dst, bf, mem, cf);
                            } else {
                                const af = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, t.a), ta);
                                const mem = loadMemF32Interior(prog.nodes.items[t.b], w_i32_spec, col_disp, env);
                                if (t.op == .fmadd) b.vecFmaMemF32(dst, af, mem, cf) else b.vecFnmaMemF32(dst, af, mem, cf);
                            }
                            handled = true;
                        }
                        const ca = if (!handled and use_const_mem and (!pin_hot_const or info.use_counts.items[t.a] < 6)) nodeConstF32(prog, env, t.a) else null;
                        const cb = if (!handled and use_const_mem and (!pin_hot_const or info.use_counts.items[t.b] < 6)) nodeConstF32(prog, env, t.b) else null;
                        const cc = if (!handled and use_const_mem and (!pin_hot_const or info.use_counts.items[t.c] < 6)) nodeConstF32(prog, env, t.c) else null;
                        if (!handled and t.op == .fmadd) {
                            if (cb != null and ca == null and cc == null) {
                                dst = if (reuse_c) slots[t.c].?.reg else b.newVec();
                                const af = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, t.a), ta);
                                const cf = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, t.c), tc);
                                b.vecFmaACF32(dst, af, cb.?, cf);
                                handled = true;
                            } else if (ca != null and cb == null and cc == null) {
                                dst = if (reuse_c) slots[t.c].?.reg else b.newVec();
                                const bf = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, t.b), tb);
                                const cf = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, t.c), tc);
                                b.vecFmaACF32(dst, bf, ca.?, cf);
                                handled = true;
                            } else if (cc != null and ca == null and cb == null) {
                                const a_owned = if (slots[t.a]) |sa| sa.owned else false;
                                const b_owned = if (slots[t.b]) |sb| sb.owned else false;
                                if (b.cpu.has_fma and b.cpu.isa != .neon and ta == .f32 and a_owned and soleHere(rem_uses, t.a, t.b, t.c, t.a)) {
                                    dst = slots[t.a].?.reg;
                                } else if (b.cpu.has_fma and b.cpu.isa != .neon and tb == .f32 and b_owned and soleHere(rem_uses, t.a, t.b, t.c, t.b)) {
                                    dst = slots[t.b].?.reg;
                                } else {
                                    dst = b.newVec();
                                }
                                const af = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, t.a), ta);
                                const bf = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, t.b), tb);
                                b.vecFmaBCF32(dst, af, bf, cc.?);
                                handled = true;
                            } else if (ca != null and cc != null and cb == null) {
                                dst = b.newVec();
                                const bf = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, t.b), tb);
                                b.vecFmaCCF32(dst, bf, ca.?, cc.?);
                                handled = true;
                            } else if (cb != null and cc != null and ca == null) {
                                dst = b.newVec();
                                const af = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, t.a), ta);
                                b.vecFmaCCF32(dst, af, cb.?, cc.?);
                                handled = true;
                            }
                        } else if (!handled and t.op == .fnmadd) {
                            if (cb != null and ca == null and cc == null) {
                                dst = if (reuse_c) slots[t.c].?.reg else b.newVec();
                                const af = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, t.a), ta);
                                const cf = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, t.c), tc);
                                b.vecFnmaBCF32(dst, af, cb.?, cf);
                                handled = true;
                            } else if (ca != null and cb == null and cc == null) {
                                dst = if (reuse_c) slots[t.c].?.reg else b.newVec();
                                const bf = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, t.b), tb);
                                const cf = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, t.c), tc);
                                b.vecFnmaBCF32(dst, bf, ca.?, cf);
                                handled = true;
                            }
                        }
                        if (!handled) {
                            // On AArch64 NEON, `fmsub(a, b, const_c)` and `fnmsub(a, b, const_c)` lower to
                            // `fneg + fmla/fmls` (2 insns). Negating a compile-time f32 constant is IEEE-754
                            // bit-exact and turns them into a single `fmla`/`fmls` with `-const_c`.
                            if (b.cpu.isa == .neon and (t.op == .fmsub or t.op == .fnmsub)) {
                                if (nodeConstF32(prog, env, t.c)) |cv| {
                                    dst = b.newVec();
                                    const af = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, t.a), ta);
                                    const bf = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, t.b), tb);
                                    const neg_c_reg = b.constF32Splat(-cv);
                                    const alt_op: ast.TernaryOp = if (t.op == .fmsub) .fmadd else .fnmadd;
                                    emitTernaryF32(b, alt_op, dst, af, bf, neg_c_reg);
                                    handled = true;
                                }
                            }
                        }
                        if (!handled) {
                            if (reuse_c) {
                                dst = slots[t.c].?.reg;
                            } else if (is_fma and b.cpu.has_fma and b.cpu.isa != .neon) {
                                const a_owned = if (slots[t.a]) |sa| sa.owned else false;
                                const b_owned = if (slots[t.b]) |sb| sb.owned else false;
                                if (ta == .f32 and a_owned and soleHere(rem_uses, t.a, t.b, t.c, t.a)) {
                                    dst = slots[t.a].?.reg;
                                } else if (tb == .f32 and b_owned and soleHere(rem_uses, t.a, t.b, t.c, t.b)) {
                                    dst = slots[t.b].?.reg;
                                } else {
                                    dst = b.newVec();
                                }
                            } else {
                                dst = b.newVec();
                            }
                            const af = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, t.a), ta);
                            const bf = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, t.b), tb);
                            const cf = ensureF32(b, resolveSlotReg(b, prog, info, env, slots, t.c), tc);
                            emitTernaryF32(b, t.op, dst, af, bf, cf);
                        }
                    }
                    releaseUse(rem_uses, slots, t.a);
                    releaseUse(rem_uses, slots, t.b);
                    releaseUse(rem_uses, slots, t.c);
                    slots[idx] = .{ .reg = dst, .owned = true };
                },
                .arg_extrema, .arg_sort_elem => unreachable,
            }
        }
    }

    var results: [count]aj.Reg = undefined;
    inline for (0..count) |s| {
        results[s] = resolveSlotReg(b, prog, info, envs[s], slots_arr[s], prog.root);
    }
    return results;
}

fn emitDagBody(
    allocator: std.mem.Allocator,
    b: *aj.Builder,
    prog: *const ast.Program,
    info: *const analysis_mod.Analysis,
    spec: CompileSpec,
    env: BodyEnv,
    x_mode: XMode,
) std.mem.Allocator.Error!void {
    // P2: use RIP-relative const-memory operands only on high-pressure / gather bodies;
    // small low-pressure kernels keep constants hoisted in vector registers (32-reg AVX-512
    // keeps constants in ZMMs up to 48 live nodes using 4B prologue vbroadcastss).
    const const_mem_thresh: usize = if (b.cpu.has_avx512) 48 else 24;
    const use_const_mem = info.live_node_count > const_mem_thresh or info.uses_abs_access;
    const root_reg = try evalDagVector(allocator, b, prog, info, spec, env, x_mode, 0, use_const_mem, null);
    const dst_bps: usize = spec.dst_fmt.bytesPerSample();
    const dst_shift: u8 = switch (dst_bps) {
        1 => 0,
        2 => 1,
        else => 2,
    };
    const dst_mem = switch (x_mode) {
        .const_col => |col| aj.Mem.ptr(env.dst_ptr, @intCast(col * dst_bps)),
        .interior => env.rowMem(env.dst_ptr, dst_shift, 0),
        .dynamic_left, .dynamic_right => aj.Mem.indexed(env.dst_ptr, env.x_gp, dst_shift, 0),
    };
    if (info.value_types.items[prog.root] == .i32) {
        pixel.emitStoreI32(b, dst_mem, root_reg, spec.dst_fmt);
    } else {
        pixel.emitStoreF32(b, dst_mem, root_reg, spec.dst_fmt);
    }
}

/// 2x unrolled interior body evaluating `x` and `x + f32_lanes`,
/// pairing `f16`/`f32`/`u32` NEON loads and `u8`/`u16`/`f16`/`f32`/`u32` stores.
/// Largest body (`live_node_count`) that converts NEON u8/u16 loads with the 2^23 bias
/// sequence; past it, the two extra pinned constants cost spills (kirsch p8: 14 -> 21 stores).
const neon_bias_load_max_nodes = 48;

fn emitDagBodyX2(
    allocator: std.mem.Allocator,
    b: *aj.Builder,
    prog: *const ast.Program,
    info: *const analysis_mod.Analysis,
    spec: CompileSpec,
    env: BodyEnv,
    v_lane_step_f32: ?aj.Reg,
    v_lane_step_i32: ?aj.Reg,
    interleave: bool,
) std.mem.Allocator.Error!void {
    const lanes: usize = b.cpu.f32_lanes;
    // P2: small 2x-unrolled kernels keep constants hoisted in vector registers (32-reg AVX-512
    // keeps constants in ZMMs up to 48 live nodes using 4B prologue vbroadcastss).
    const const_mem_thresh: usize = if (b.cpu.has_avx512) 48 else 24;
    const use_const_mem = info.live_node_count > const_mem_thresh;

    // On AArch64 NEON, preload paired `.load_clip` nodes via single 16B `ldr q + fcvtl/fcvtl2` (f16)
    // or `ldp q0, q1` (f32/u32 with running pointers) before evaluating both halves.
    const n_nodes = prog.nodes.items.len;
    var pre0: ?[]?aj.Reg = null;
    var pre1: ?[]?aj.Reg = null;
    defer if (pre0) |p| allocator.free(p);
    defer if (pre1) |p| allocator.free(p);

    if (b.cpu.isa == .neon and !interleave) {
        var any_pair = false;
        for (prog.nodes.items, 0..) |node, idx| {
            if (idx != prog.root and info.use_counts.items[idx] == 0) continue;
            if (node != .load_clip) continue;
            const fmt = spec.src_fmts[node.load_clip];
            const is_u8_pair = !fmt.is_float and fmt.bits_per_sample <= 8;
            const is_u16_pair = !fmt.is_float and fmt.bits_per_sample > 8 and fmt.bits_per_sample <= 16;
            const is_f16_pair = fmt.is_float and fmt.bits_per_sample == 16;
            const is_32_pair = env.use_pair_mem and fmt.bits_per_sample == 32;
            if (!is_u8_pair and !is_u16_pair and !is_f16_pair and !is_32_pair) continue;

            if (!any_pair) {
                pre0 = try allocator.alloc(?aj.Reg, n_nodes);
                pre1 = try allocator.alloc(?aj.Reg, n_nodes);
                @memset(pre0.?, null);
                @memset(pre1.?, null);
                any_pair = true;
            }
            const clip = node.load_clip;
            const bps = fmt.bytesPerSample();
            const shift: u8 = switch (bps) {
                1 => 0,
                2 => 1,
                else => 2,
            };
            const mem0 = env.rowMem(env.src_row_ptrs[clip].?, shift, 0);
            const r0 = b.newVec();
            const r1 = b.newVec();
            if ((is_u8_pair or is_u16_pair) and info.value_types.items[idx] == .f32) {
                b.vecLoadUintToF32x2(r0, r1, mem0, bps);
            } else if (is_u8_pair) {
                b.vecLoadU8ToU32x2(r0, r1, mem0);
            } else if (is_u16_pair) {
                b.vecLoadU16ToU32x2(r0, r1, mem0);
            } else if (is_f16_pair) {
                b.vecCvtF16ToF32x2(r0, r1, mem0);
            } else {
                b.vecLoadUVecx2(r0, r1, mem0);
                if (!fmt.is_float and info.value_types.items[idx] == .f32) {
                    b.vecOp2(.cvt_i32_to_f32, r0, r0);
                    b.vecOp2(.cvt_i32_to_f32, r1, r1);
                }
            }
            pre0.?[idx] = r0;
            pre1.?[idx] = r1;
        }
    }

    var env1 = env;
    if (env.hoisted_slots_hi.len == n_nodes) {
        env1.hoisted_slots = env.hoisted_slots_hi;
    }
    if (env.hoisted_masks_hi.len == n_nodes) {
        env1.hoisted_masks = env.hoisted_masks_hi;
    }
    if (env.v_x) |vx0| {
        const vx1 = b.newVec();
        if (prog.opt_int) {
            b.vecOp3(.add_i32, vx1, vx0, v_lane_step_i32.?);
        } else {
            b.vecOp3(.add_f32, vx1, vx0, v_lane_step_f32.?);
        }
        env1.v_x = vx1;
    }

    var res0: aj.Reg = undefined;
    var res1: aj.Reg = undefined;
    if (interleave) {
        const pair = try evalDagVectors(2, allocator, b, prog, info, spec, .{ env, env1 }, .interior, .{ 0, lanes }, use_const_mem, .{ pre0, pre1 });
        res0 = pair[0];
        res1 = pair[1];
    } else {
        res0 = try evalDagVector(allocator, b, prog, info, spec, env, .interior, 0, use_const_mem, pre0);
        res1 = try evalDagVector(allocator, b, prog, info, spec, env1, .interior, lanes, use_const_mem, pre1);
    }

    const dst_bps: usize = spec.dst_fmt.bytesPerSample();
    const dst_shift: u8 = switch (dst_bps) {
        1 => 0,
        2 => 1,
        else => 2,
    };
    const dst_mem0 = env.rowMem(env.dst_ptr, dst_shift, 0);
    const dst_mem1 = env.rowMem(env.dst_ptr, dst_shift, @intCast(lanes * dst_bps));
    if (info.value_types.items[prog.root] == .i32) {
        pixel.emitStoreI32x2(b, dst_mem0, dst_mem1, res0, res1, spec.dst_fmt, env.use_pair_mem);
    } else {
        pixel.emitStoreF32x2(b, dst_mem0, dst_mem1, res0, res1, spec.dst_fmt, env.use_pair_mem);
    }
}

/// 4x interleaved interior body for narrow serial FP chains on 32-register AArch64 NEON
/// (16 pixels/iteration lockstep, zero spills).
fn emitDagBodyX4(
    allocator: std.mem.Allocator,
    b: *aj.Builder,
    prog: *const ast.Program,
    info: *const analysis_mod.Analysis,
    spec: CompileSpec,
    env: BodyEnv,
    v_lane_step_f32: ?aj.Reg,
    v_lane_step_i32: ?aj.Reg,
) std.mem.Allocator.Error!void {
    _ = v_lane_step_i32;
    const lanes: usize = b.cpu.f32_lanes;
    const use_const_mem = info.live_node_count > 24;

    var envs: [4]BodyEnv = .{ env, env, env, env };
    if (env.v_x) |vx0| {
        const vx1 = b.newVec();
        const vx2 = b.newVec();
        const vx3 = b.newVec();
        b.vecOp3(.add_f32, vx1, vx0, v_lane_step_f32.?);
        b.vecOp3(.add_f32, vx2, vx0, b.constF32Splat(@floatFromInt(lanes * 2)));
        b.vecOp3(.add_f32, vx3, vx0, b.constF32Splat(@floatFromInt(lanes * 3)));
        envs[1].v_x = vx1;
        envs[2].v_x = vx2;
        envs[3].v_x = vx3;
    }

    const res = try evalDagVectors(
        4,
        allocator,
        b,
        prog,
        info,
        spec,
        envs,
        .interior,
        .{ 0, lanes, lanes * 2, lanes * 3 },
        use_const_mem,
        .{ null, null, null, null },
    );

    const dst_bps: usize = spec.dst_fmt.bytesPerSample();
    const dst_shift: u8 = switch (dst_bps) {
        1 => 0,
        2 => 1,
        else => 2,
    };
    const m0 = env.rowMem(env.dst_ptr, dst_shift, 0);
    const m1 = env.rowMem(env.dst_ptr, dst_shift, @intCast(lanes * dst_bps));
    const m2 = env.rowMem(env.dst_ptr, dst_shift, @intCast(lanes * 2 * dst_bps));
    const m3 = env.rowMem(env.dst_ptr, dst_shift, @intCast(lanes * 3 * dst_bps));
    pixel.emitStoreF32x2(b, m0, m1, res[0], res[1], spec.dst_fmt, env.use_pair_mem);
    pixel.emitStoreF32x2(b, m2, m3, res[2], res[3], spec.dst_fmt, env.use_pair_mem);
}

fn emitUnaryF32(b: *aj.Builder, op: ast.UnaryOp, dst: aj.Reg, a: aj.Reg, use_const_mem: bool) void {
    switch (op) {
        .neg => b.vecOp2(.neg_f32, dst, a),
        .abs => b.vecOp2(.abs_f32, dst, a),
        .sqrt => {
            b.vecOp3(.max_f32, dst, a, b.vecZero());
            b.vecOp2(.sqrt_f32, dst, dst);
        },
        .rsqrt => {
            const clamped = b.newVec();
            if (use_const_mem) {
                b.vecMaxCF32(clamped, a, interp.flt_min_f32);
                b.vecMinCF32(clamped, clamped, interp.flt_max_f32);
            } else {
                b.vecOp3(.max_f32, clamped, a, b.constF32Splat(interp.flt_min_f32));
                b.vecOp3(.min_f32, clamped, clamped, b.constF32Splat(interp.flt_max_f32));
            }
            b.vecOp2(.rsqrt_f32, dst, clamped);
        },
        .sgn => {
            const zero = b.vecZero();
            const one = b.constF32Splat(1.0);
            const gt = b.newVec();
            const lt = b.newVec();
            b.vecOp3(.cmp_gt_f32, gt, a, zero);
            b.vecOp3(.and_u32, gt, gt, one);
            b.vecOp3(.cmp_lt_f32, lt, a, zero);
            b.vecOp3(.and_u32, lt, lt, one);
            b.vecOp3(.sub_f32, dst, gt, lt);
        },
        .trunc => pixel.emitRoundF32(b, dst, a, .trunc),
        .floor => pixel.emitRoundF32(b, dst, a, .floor),
        .ceil => pixel.emitRoundF32(b, dst, a, .ceil),
        .round => pixel.emitRoundF32(b, dst, a, .round_even),
        .not => {
            b.vecOp3(.cmp_le_f32, dst, a, b.vecZero());
            b.vecOp3(.and_u32, dst, dst, b.constF32Splat(1.0));
        },
        .bitnot => {
            b.vecOp2(.cvt_round_f32_to_i32, dst, a);
            b.vecOp2(.not_u32, dst, dst);
            b.vecOp2(.cvt_i32_to_f32, dst, dst);
        },
        .exp => math_jit.emitExp(b, dst, a, use_const_mem),
        .exp2 => math_jit.emitExp2(b, dst, a, use_const_mem),
        .log => math_jit.emitLog(b, dst, a, use_const_mem),
        .log2 => math_jit.emitLog2(b, dst, a, use_const_mem),
        .sin => math_jit.emitSin(b, dst, a),
        .cos => math_jit.emitCos(b, dst, a),
        .tan => math_jit.emitTan(b, dst, a),
        .atan => math_jit.emitAtan(b, dst, a),
        .asin => math_jit.emitAsin(b, dst, a),
        .acos => math_jit.emitAcos(b, dst, a),
        .to_f32 => b.vecOp2(.mov, dst, a),
    }
}

fn emitUnaryF32ToI32(b: *aj.Builder, op: ast.UnaryOp, dst: aj.Reg, a: aj.Reg) void {
    switch (op) {
        .not => {
            b.vecOp3(.cmp_le_f32, dst, a, b.vecZero());
            b.vecOp3(.and_u32, dst, dst, b.constI32Splat(1));
        },
        .bitnot => {
            b.vecOp2(.cvt_round_f32_to_i32, dst, a);
            b.vecOp2(.not_u32, dst, dst);
        },
        else => unreachable,
    }
}

fn emitUnaryI32(b: *aj.Builder, op: ast.UnaryOp, dst: aj.Reg, a: aj.Reg) void {
    switch (op) {
        .neg => b.vecOp2(.neg_i32, dst, a),
        .abs => b.vecOp2(.abs_i32, dst, a),
        .sgn => {
            const zero = b.vecZero();
            const one = b.constI32Splat(1);
            const gt = b.newVec();
            const lt = b.newVec();
            b.vecOp3(.cmp_gt_i32, gt, a, zero);
            b.vecOp3(.and_u32, gt, gt, one);
            b.vecOp3(.cmp_gt_i32, lt, zero, a);
            b.vecOp3(.and_u32, lt, lt, one);
            b.vecOp3(.sub_i32, dst, gt, lt);
        },
        .not => {
            b.vecOp3(.cmp_gt_i32, dst, a, b.vecZero());
            b.vecOp2(.not_u32, dst, dst);
            b.vecOp3(.and_u32, dst, dst, b.constI32Splat(1));
        },
        .bitnot => b.vecOp2(.not_u32, dst, a),
        else => unreachable,
    }
}

fn emitBinaryF32ToI32(b: *aj.Builder, op: ast.BinaryOp, dst: aj.Reg, a: aj.Reg, rhs: aj.Reg) void {
    switch (op) {
        .lt => {
            b.vecOp3(.cmp_lt_f32, dst, a, rhs);
            b.vecOp3(.and_u32, dst, dst, b.constI32Splat(1));
        },
        .le => {
            b.vecOp3(.cmp_le_f32, dst, a, rhs);
            b.vecOp3(.and_u32, dst, dst, b.constI32Splat(1));
        },
        .gt => {
            b.vecOp3(.cmp_gt_f32, dst, a, rhs);
            b.vecOp3(.and_u32, dst, dst, b.constI32Splat(1));
        },
        .ge => {
            b.vecOp3(.cmp_ge_f32, dst, a, rhs);
            b.vecOp3(.and_u32, dst, dst, b.constI32Splat(1));
        },
        .eq => {
            b.vecOp3(.cmp_eq_f32, dst, a, rhs);
            b.vecOp3(.and_u32, dst, dst, b.constI32Splat(1));
        },
        .and_op => {
            const zero = b.vecZero();
            const ma = b.newVec();
            const mb = b.newVec();
            b.vecOp3(.cmp_gt_f32, ma, a, zero);
            b.vecOp3(.cmp_gt_f32, mb, rhs, zero);
            b.vecOp3(.and_u32, dst, ma, mb);
            b.vecOp3(.and_u32, dst, dst, b.constI32Splat(1));
        },
        .or_op => {
            const zero = b.vecZero();
            const ma = b.newVec();
            const mb = b.newVec();
            b.vecOp3(.cmp_gt_f32, ma, a, zero);
            b.vecOp3(.cmp_gt_f32, mb, rhs, zero);
            b.vecOp3(.or_u32, dst, ma, mb);
            b.vecOp3(.and_u32, dst, dst, b.constI32Splat(1));
        },
        .xor_op => {
            const zero = b.vecZero();
            const ma = b.newVec();
            const mb = b.newVec();
            b.vecOp3(.cmp_gt_f32, ma, a, zero);
            b.vecOp3(.cmp_gt_f32, mb, rhs, zero);
            b.vecOp3(.xor_u32, dst, ma, mb);
            b.vecOp3(.and_u32, dst, dst, b.constI32Splat(1));
        },
        .bitand => {
            const ia = b.newVec();
            const ib = b.newVec();
            b.vecOp2(.cvt_round_f32_to_i32, ia, a);
            b.vecOp2(.cvt_round_f32_to_i32, ib, rhs);
            b.vecOp3(.and_u32, dst, ia, ib);
        },
        .bitor => {
            const ia = b.newVec();
            const ib = b.newVec();
            b.vecOp2(.cvt_round_f32_to_i32, ia, a);
            b.vecOp2(.cvt_round_f32_to_i32, ib, rhs);
            b.vecOp3(.or_u32, dst, ia, ib);
        },
        .bitxor => {
            const ia = b.newVec();
            const ib = b.newVec();
            b.vecOp2(.cvt_round_f32_to_i32, ia, a);
            b.vecOp2(.cvt_round_f32_to_i32, ib, rhs);
            b.vecOp3(.xor_u32, dst, ia, ib);
        },
        else => unreachable,
    }
}

fn emitBinaryF32(b: *aj.Builder, op: ast.BinaryOp, dst: aj.Reg, a: aj.Reg, rhs: aj.Reg) void {
    switch (op) {
        .add => b.vecOp3(.add_f32, dst, a, rhs),
        .sub => b.vecOp3(.sub_f32, dst, a, rhs),
        .mul => b.vecOp3(.mul_f32, dst, a, rhs),
        .div => b.vecOp3(.div_f32, dst, a, rhs),
        .mod => {
            const q = b.newVec();
            b.vecOp3(.div_f32, q, a, rhs);
            pixel.emitRoundF32(b, q, q, .trunc);
            b.vecOp3(.mul_f32, q, q, rhs);
            b.vecOp3(.sub_f32, dst, a, q);
        },
        .min => b.vecOp3(.min_f32, dst, a, rhs),
        .max => b.vecOp3(.max_f32, dst, a, rhs),
        .pow => {
            const l = b.newVec();
            math_jit.emitLog2(b, l, a, true);
            b.vecOp3(.mul_f32, l, l, rhs);
            math_jit.emitExp2(b, dst, l, true);
        },
        .atan2 => math_jit.emitAtan2(b, dst, a, rhs),
        .copysign => {
            const mag = b.newVec();
            const sgn = b.newVec();
            b.vecOp3(.and_u32, mag, a, b.constI32Splat(0x7FFF_FFFF));
            b.vecOp3(.and_u32, sgn, rhs, b.constI32Splat(@bitCast(@as(u32, 0x8000_0000))));
            b.vecOp3(.or_u32, dst, mag, sgn);
        },
        .lt => {
            b.vecOp3(.cmp_lt_f32, dst, a, rhs);
            b.vecOp3(.and_u32, dst, dst, b.constF32Splat(1.0));
        },
        .le => {
            b.vecOp3(.cmp_le_f32, dst, a, rhs);
            b.vecOp3(.and_u32, dst, dst, b.constF32Splat(1.0));
        },
        .gt => {
            b.vecOp3(.cmp_gt_f32, dst, a, rhs);
            b.vecOp3(.and_u32, dst, dst, b.constF32Splat(1.0));
        },
        .ge => {
            b.vecOp3(.cmp_ge_f32, dst, a, rhs);
            b.vecOp3(.and_u32, dst, dst, b.constF32Splat(1.0));
        },
        .eq => {
            b.vecOp3(.cmp_eq_f32, dst, a, rhs);
            b.vecOp3(.and_u32, dst, dst, b.constF32Splat(1.0));
        },
        .and_op => {
            const zero = b.vecZero();
            const ma = b.newVec();
            const mb = b.newVec();
            b.vecOp3(.cmp_gt_f32, ma, a, zero);
            b.vecOp3(.cmp_gt_f32, mb, rhs, zero);
            b.vecOp3(.and_u32, dst, ma, mb);
            b.vecOp3(.and_u32, dst, dst, b.constF32Splat(1.0));
        },
        .or_op => {
            const zero = b.vecZero();
            const ma = b.newVec();
            const mb = b.newVec();
            b.vecOp3(.cmp_gt_f32, ma, a, zero);
            b.vecOp3(.cmp_gt_f32, mb, rhs, zero);
            b.vecOp3(.or_u32, dst, ma, mb);
            b.vecOp3(.and_u32, dst, dst, b.constF32Splat(1.0));
        },
        .xor_op => {
            const zero = b.vecZero();
            const ma = b.newVec();
            const mb = b.newVec();
            b.vecOp3(.cmp_gt_f32, ma, a, zero);
            b.vecOp3(.cmp_gt_f32, mb, rhs, zero);
            b.vecOp3(.xor_u32, dst, ma, mb);
            b.vecOp3(.and_u32, dst, dst, b.constF32Splat(1.0));
        },
        .bitand => {
            const ia = b.newVec();
            const ib = b.newVec();
            b.vecOp2(.cvt_round_f32_to_i32, ia, a);
            b.vecOp2(.cvt_round_f32_to_i32, ib, rhs);
            b.vecOp3(.and_u32, dst, ia, ib);
            b.vecOp2(.cvt_i32_to_f32, dst, dst);
        },
        .bitor => {
            const ia = b.newVec();
            const ib = b.newVec();
            b.vecOp2(.cvt_round_f32_to_i32, ia, a);
            b.vecOp2(.cvt_round_f32_to_i32, ib, rhs);
            b.vecOp3(.or_u32, dst, ia, ib);
            b.vecOp2(.cvt_i32_to_f32, dst, dst);
        },
        .bitxor => {
            const ia = b.newVec();
            const ib = b.newVec();
            b.vecOp2(.cvt_round_f32_to_i32, ia, a);
            b.vecOp2(.cvt_round_f32_to_i32, ib, rhs);
            b.vecOp3(.xor_u32, dst, ia, ib);
            b.vecOp2(.cvt_i32_to_f32, dst, dst);
        },
    }
}

fn emitBinaryI32(b: *aj.Builder, op: ast.BinaryOp, dst: aj.Reg, a: aj.Reg, rhs: aj.Reg) void {
    switch (op) {
        .add => b.vecOp3(.add_i32, dst, a, rhs),
        .sub => b.vecOp3(.sub_i32, dst, a, rhs),
        .mul => b.vecOp3(.mul_i32, dst, a, rhs),
        .min => b.vecOp3(.min_i32, dst, a, rhs),
        .max => b.vecOp3(.max_i32, dst, a, rhs),
        .lt => {
            b.vecOp3(.cmp_gt_i32, dst, rhs, a);
            b.vecOp3(.and_u32, dst, dst, b.constI32Splat(1));
        },
        .le => {
            b.vecOp3(.cmp_gt_i32, dst, a, rhs);
            b.vecOp2(.not_u32, dst, dst);
            b.vecOp3(.and_u32, dst, dst, b.constI32Splat(1));
        },
        .gt => {
            b.vecOp3(.cmp_gt_i32, dst, a, rhs);
            b.vecOp3(.and_u32, dst, dst, b.constI32Splat(1));
        },
        .ge => {
            b.vecOp3(.cmp_gt_i32, dst, rhs, a);
            b.vecOp2(.not_u32, dst, dst);
            b.vecOp3(.and_u32, dst, dst, b.constI32Splat(1));
        },
        .eq => {
            b.vecOp3(.cmp_eq_i32, dst, a, rhs);
            b.vecOp3(.and_u32, dst, dst, b.constI32Splat(1));
        },
        .and_op => {
            const zero = b.vecZero();
            const ma = b.newVec();
            const mb = b.newVec();
            b.vecOp3(.cmp_gt_i32, ma, a, zero);
            b.vecOp3(.cmp_gt_i32, mb, rhs, zero);
            b.vecOp3(.and_u32, dst, ma, mb);
            b.vecOp3(.and_u32, dst, dst, b.constI32Splat(1));
        },
        .or_op => {
            const zero = b.vecZero();
            const ma = b.newVec();
            const mb = b.newVec();
            b.vecOp3(.cmp_gt_i32, ma, a, zero);
            b.vecOp3(.cmp_gt_i32, mb, rhs, zero);
            b.vecOp3(.or_u32, dst, ma, mb);
            b.vecOp3(.and_u32, dst, dst, b.constI32Splat(1));
        },
        .xor_op => {
            const zero = b.vecZero();
            const ma = b.newVec();
            const mb = b.newVec();
            b.vecOp3(.cmp_gt_i32, ma, a, zero);
            b.vecOp3(.cmp_gt_i32, mb, rhs, zero);
            b.vecOp3(.xor_u32, dst, ma, mb);
            b.vecOp3(.and_u32, dst, dst, b.constI32Splat(1));
        },
        .bitand => b.vecOp3(.and_u32, dst, a, rhs),
        .bitor => b.vecOp3(.or_u32, dst, a, rhs),
        .bitxor => b.vecOp3(.xor_u32, dst, a, rhs),
        else => unreachable,
    }
}

fn emitTernaryF32(b: *aj.Builder, op: ast.TernaryOp, dst: aj.Reg, a: aj.Reg, b_reg: aj.Reg, c_reg: aj.Reg) void {
    switch (op) {
        .select => {
            const mask = if (b.cpu.prefersMaskDstReuse() and dst.raw.id != b_reg.raw.id and dst.raw.id != c_reg.raw.id) dst else b.newVec();
            b.vecOp3(.cmp_gt_f32, mask, a, b.vecZero());
            pixel.emitSelectF32ByMask(b, dst, c_reg, b_reg, mask);
        },
        .fmadd => b.vecOp4(.madd_f32, dst, a, b_reg, c_reg),
        .fmsub => b.vecOp4(.msub_f32, dst, a, b_reg, c_reg),
        .fnmadd => b.vecOp4(.nmadd_f32, dst, a, b_reg, c_reg),
        .fnmsub => b.vecOp4(.nmsub_f32, dst, a, b_reg, c_reg),
        .linstep => unreachable,
    }
}

// ============================================================================
// Host-free JIT-vs-oracle bit-exact tests (Phase 2a + 2b + Phase 3 spatial/f16)
// ============================================================================

fn verifyBitExactF32(allocator: std.mem.Allocator, cpu: cpu_mod.CpuInfo, expr_str: []const u8, width: usize, height: usize) !void {
    const parser = @import("../expr/parser.zig");
    const lower_mod = @import("../expr/lower.zig");
    const optimize_mod = @import("../expr/optimize.zig");

    const profile = interp.TargetProfile.fromCpu(cpu);
    const fmt = ast.SampleFormat{ .is_float = true, .bits_per_sample = 32 };
    const fmts = [_]ast.SampleFormat{ fmt, fmt };

    var prog = try parser.parse(allocator, expr_str, .{ .num_clips = 2 });
    defer prog.deinit(allocator);
    try lower_mod.lower(allocator, &prog, profile);
    try optimize_mod.optimize(allocator, &prog, profile, .{
        .enable_value_changing = true,
        .enable_fma = profile.use_fma,
    });

    var info = try analysis_mod.analyze(allocator, &prog, &fmts, fmt, @intCast(width), @intCast(height), .{ .has_fast_u8_lut = aj.Builder.cpuSupportsFastLutU8(cpu), .f32_lanes = cpu.f32_lanes });
    defer info.deinit(allocator);

    var rt = try aj.Runtime.init();
    defer rt.deinit();

    var compiled = try compilePlane(allocator, &rt, cpu, &prog, &info, .{
        .width = width,
        .height = height,
        .src_fmts = &fmts,
        .dst_fmt = fmt,
    });
    defer compiled.deinit(allocator, &rt);

    const stride_elems = std.mem.alignForward(usize, width, 16);
    const stride_bytes: isize = @intCast(stride_elems * 4);
    const total = stride_elems * height;

    const src0 = try allocator.alignedAlloc(f32, .@"64", total);
    defer allocator.free(src0);
    const src1 = try allocator.alignedAlloc(f32, .@"64", total);
    defer allocator.free(src1);
    const dst_jit = try allocator.alignedAlloc(f32, .@"64", total);
    defer allocator.free(dst_jit);
    const dst_ref = try allocator.alignedAlloc(f32, .@"64", total);
    defer allocator.free(dst_ref);

    // Include quiet NaN and signaling NaN (V10) alongside edge values.
    const snan: f32 = @bitCast(@as(u32, 0x7FA0_0001));
    const edge_vals = [_]f32{
        -10.5,             -2.5, -1.5,              -0.5,
        -0.3,              -0.0, 0.0,               0.1,
        0.5,               1.5,  2.5,               10.7,
        std.math.nan(f32), snan, std.math.inf(f32), -std.math.inf(f32),
        1e10,              0.25,
    };
    // M3: cover every single edge value on x and y (and all n*n pairs when width*height >= n*n).
    const n_edge = edge_vals.len;
    for (0..height) |y| {
        for (0..stride_elems) |x| {
            const dst_idx = y * stride_elems + x;
            const pair_idx = y * width + @min(x, width - 1);
            src0[dst_idx] = edge_vals[pair_idx % n_edge];
            src1[dst_idx] = edge_vals[(pair_idx + pair_idx / n_edge) % n_edge];
            dst_jit[dst_idx] = -777.0;
            dst_ref[dst_idx] = -777.0;
        }
    }

    const srcs = [_][*]const u8{ @ptrCast(src0.ptr), @ptrCast(src1.ptr) };
    const strides = [_]isize{ stride_bytes, stride_bytes };
    const props = [_]f32{ 3.25, -1.5 };

    compiled.kernel.func(@ptrCast(dst_jit.ptr), stride_bytes, &srcs, &strides, 7, &props);
    try interp.evalPlane(allocator, &prog, profile, .{
        .dst = @ptrCast(dst_ref.ptr),
        .dst_stride = stride_bytes,
        .dst_fmt = fmt,
        .srcs = &srcs,
        .src_strides = &strides,
        .src_fmts = &fmts,
        .width = width,
        .height = height,
        .frame_n = 7,
        .props = &props,
    });

    for (0..height) |y| {
        for (0..width) |x| {
            const idx = y * stride_elems + x;
            const g = dst_jit[idx];
            const e = dst_ref[idx];
            if (std.math.isNan(e)) {
                try std.testing.expect(std.math.isNan(g));
            } else {
                const gb: u32 = @bitCast(g);
                const eb: u32 = @bitCast(e);
                if (eb != gb) {
                    std.debug.print("MISMATCH isa={s} expr='{s}' w={d} h={d} x={d} y={d}: expected={d} (0x{x}) got={d} (0x{x})\n", .{
                        cpu.isaName(), expr_str, width, height, x, y, e, eb, g, gb,
                    });
                }
                try std.testing.expectEqual(eb, gb);
            }
        }
    }
}

fn verifyBitExactU8(allocator: std.mem.Allocator, cpu: cpu_mod.CpuInfo, expr_str: []const u8, width: usize, height: usize, opt_int: bool) !void {
    const parser = @import("../expr/parser.zig");
    const lower_mod = @import("../expr/lower.zig");
    const optimize_mod = @import("../expr/optimize.zig");

    const profile = interp.TargetProfile.fromCpu(cpu);
    const fmt8 = ast.SampleFormat{ .is_float = false, .bits_per_sample = 8 };
    const fmts = [_]ast.SampleFormat{ fmt8, fmt8 };

    var prog = try parser.parse(allocator, expr_str, .{ .num_clips = 2, .opt_int = opt_int });
    defer prog.deinit(allocator);
    try lower_mod.lower(allocator, &prog, profile);
    try optimize_mod.optimize(allocator, &prog, profile, .{
        .enable_value_changing = true,
        .enable_fma = profile.use_fma,
    });

    var info = try analysis_mod.analyze(allocator, &prog, &fmts, fmt8, @intCast(width), @intCast(height), .{ .has_fast_u8_lut = aj.Builder.cpuSupportsFastLutU8(cpu), .f32_lanes = cpu.f32_lanes });
    defer info.deinit(allocator);

    var rt = try aj.Runtime.init();
    defer rt.deinit();

    var compiled = try compilePlane(allocator, &rt, cpu, &prog, &info, .{
        .width = width,
        .height = height,
        .src_fmts = &fmts,
        .dst_fmt = fmt8,
    });
    defer compiled.deinit(allocator, &rt);

    const stride_u8 = std.mem.alignForward(usize, width, 64);

    const src0 = try allocator.alignedAlloc(u8, .@"64", stride_u8 * height);
    defer allocator.free(src0);
    const src1 = try allocator.alignedAlloc(u8, .@"64", stride_u8 * height);
    defer allocator.free(src1);
    const dst_jit = try allocator.alignedAlloc(u8, .@"64", stride_u8 * height);
    defer allocator.free(dst_jit);
    const dst_ref = try allocator.alignedAlloc(u8, .@"64", stride_u8 * height);
    defer allocator.free(dst_ref);

    for (0..height) |y| {
        for (0..width) |x| {
            src0[y * stride_u8 + x] = if (x == 0) 254 else @intCast((y * 37 + x * 19 + 3) & 255);
            src1[y * stride_u8 + x] = @intCast((y * 17 + x * 13 + 7) & 255);
        }
    }

    const srcs = [_][*]const u8{ src0.ptr, src1.ptr };
    const strides = [_]isize{ @intCast(stride_u8), @intCast(stride_u8) };

    compiled.kernel.func(dst_jit.ptr, @intCast(stride_u8), &srcs, &strides, 0, null);
    try interp.evalPlane(allocator, &prog, profile, .{
        .dst = dst_ref.ptr,
        .dst_stride = @intCast(stride_u8),
        .dst_fmt = fmt8,
        .srcs = &srcs,
        .src_strides = &strides,
        .src_fmts = &fmts,
        .width = width,
        .height = height,
        .frame_n = 0,
        .props = &.{},
    });

    for (0..height) |y| {
        for (0..width) |x| {
            try std.testing.expectEqual(dst_ref[y * stride_u8 + x], dst_jit[y * stride_u8 + x]);
        }
    }
}

/// K4: Float -> u8 D4 store parity against the scalar oracle across qNaN, +-sNaN, +-inf, and OOB values.
fn verifyBitExactF32ToU8(allocator: std.mem.Allocator, cpu: cpu_mod.CpuInfo, expr_str: []const u8, width: usize, height: usize) !void {
    const parser = @import("../expr/parser.zig");
    const lower_mod = @import("../expr/lower.zig");
    const optimize_mod = @import("../expr/optimize.zig");

    const profile = interp.TargetProfile.fromCpu(cpu);
    const fmt_f32 = ast.SampleFormat{ .is_float = true, .bits_per_sample = 32 };
    const fmt8 = ast.SampleFormat{ .is_float = false, .bits_per_sample = 8 };
    const fmts = [_]ast.SampleFormat{ fmt_f32, fmt_f32 };

    var prog = try parser.parse(allocator, expr_str, .{ .num_clips = 2 });
    defer prog.deinit(allocator);
    try lower_mod.lower(allocator, &prog, profile);
    try optimize_mod.optimize(allocator, &prog, profile, .{
        .enable_value_changing = true,
        .enable_fma = profile.use_fma,
    });

    var info = try analysis_mod.analyze(allocator, &prog, &fmts, fmt8, @intCast(width), @intCast(height), .{ .has_fast_u8_lut = aj.Builder.cpuSupportsFastLutU8(cpu), .f32_lanes = cpu.f32_lanes });
    defer info.deinit(allocator);

    var rt = try aj.Runtime.init();
    defer rt.deinit();

    var compiled = try compilePlane(allocator, &rt, cpu, &prog, &info, .{
        .width = width,
        .height = height,
        .src_fmts = &fmts,
        .dst_fmt = fmt8,
    });
    defer compiled.deinit(allocator, &rt);

    const stride_f32 = std.mem.alignForward(usize, width, 16);
    const stride_f32_bytes: isize = @intCast(stride_f32 * 4);
    const stride_u8 = std.mem.alignForward(usize, width, 64);

    const src0 = try allocator.alignedAlloc(f32, .@"64", stride_f32 * height);
    defer allocator.free(src0);
    const src1 = try allocator.alignedAlloc(f32, .@"64", stride_f32 * height);
    defer allocator.free(src1);
    const dst_jit = try allocator.alignedAlloc(u8, .@"64", stride_u8 * height);
    defer allocator.free(dst_jit);
    const dst_ref = try allocator.alignedAlloc(u8, .@"64", stride_u8 * height);
    defer allocator.free(dst_ref);

    const pos_snan: f32 = @bitCast(@as(u32, 0x7FA0_0001));
    const neg_snan: f32 = @bitCast(@as(u32, 0xFFA0_0001));
    const edge_vals = [_]f32{
        -100.0,            -0.5,     -0.0,     0.0,               0.5,                1.5, 127.5, 254.5, 255.0, 300.0,
        std.math.nan(f32), pos_snan, neg_snan, std.math.inf(f32), -std.math.inf(f32),
    };
    // M3: cover every single edge value on x and y (and all 15*15=225 pairs at width >= 225).
    const n_edge = edge_vals.len;
    for (0..height) |y| {
        for (0..width) |x| {
            const idx = y * width + x;
            src0[y * stride_f32 + x] = edge_vals[idx % n_edge];
            src1[y * stride_f32 + x] = edge_vals[(idx + idx / n_edge) % n_edge];
        }
    }

    const srcs = [_][*]const u8{ @ptrCast(src0.ptr), @ptrCast(src1.ptr) };
    const strides = [_]isize{ stride_f32_bytes, stride_f32_bytes };

    compiled.kernel.func(dst_jit.ptr, @intCast(stride_u8), &srcs, &strides, 0, null);
    try interp.evalPlane(allocator, &prog, profile, .{
        .dst = dst_ref.ptr,
        .dst_stride = @intCast(stride_u8),
        .dst_fmt = fmt8,
        .srcs = &srcs,
        .src_strides = &strides,
        .src_fmts = &fmts,
        .width = width,
        .height = height,
        .frame_n = 0,
        .props = &.{},
    });

    for (0..height) |y| {
        for (0..width) |x| {
            try std.testing.expectEqual(dst_ref[y * stride_u8 + x], dst_jit[y * stride_u8 + x]);
        }
    }
}

/// 32-bit integer (signed i32 semantics matching akarin) bit-exact JIT vs scalar oracle verification.
fn verifyBitExactI32(allocator: std.mem.Allocator, cpu: cpu_mod.CpuInfo, expr_str: []const u8, width: usize, height: usize, opt_int: bool) !void {
    const parser = @import("../expr/parser.zig");
    const lower_mod = @import("../expr/lower.zig");
    const optimize_mod = @import("../expr/optimize.zig");

    const profile = interp.TargetProfile.fromCpu(cpu);
    const fmt32 = ast.SampleFormat{ .is_float = false, .bits_per_sample = 32 };
    const fmts = [_]ast.SampleFormat{ fmt32, fmt32 };

    var prog = try parser.parse(allocator, expr_str, .{ .num_clips = 2, .opt_int = opt_int });
    defer prog.deinit(allocator);
    try lower_mod.lower(allocator, &prog, profile);
    try optimize_mod.optimize(allocator, &prog, profile, .{
        .enable_value_changing = true,
        .enable_fma = profile.use_fma,
    });

    var info = try analysis_mod.analyze(allocator, &prog, &fmts, fmt32, @intCast(width), @intCast(height), .{ .has_fast_u8_lut = aj.Builder.cpuSupportsFastLutU8(cpu), .f32_lanes = cpu.f32_lanes });
    defer info.deinit(allocator);
    // 32-bit integer inputs must never be routed into the 65536-entry u16 LUT.
    try std.testing.expectEqual(analysis_mod.LutKind.none, info.lut_kind);
    try std.testing.expectEqual(analysis_mod.LutKind.none, info.fallback_lut_kind);

    var rt = try aj.Runtime.init();
    defer rt.deinit();

    var compiled = try compilePlane(allocator, &rt, cpu, &prog, &info, .{
        .width = width,
        .height = height,
        .src_fmts = &fmts,
        .dst_fmt = fmt32,
    });
    defer compiled.deinit(allocator, &rt);

    const stride_i32 = std.mem.alignForward(usize, width, 16);
    const stride_bytes: isize = @intCast(stride_i32 * 4);

    const src0 = try allocator.alignedAlloc(i32, .@"64", stride_i32 * height);
    defer allocator.free(src0);
    const src1 = try allocator.alignedAlloc(i32, .@"64", stride_i32 * height);
    defer allocator.free(src1);
    const dst_jit = try allocator.alignedAlloc(i32, .@"64", stride_i32 * height);
    defer allocator.free(dst_jit);
    const dst_ref = try allocator.alignedAlloc(i32, .@"64", stride_i32 * height);
    defer allocator.free(dst_ref);

    const samples = [_]i32{
        0,                    1,                        -1,                       42,
        16_777_217,           100_000_000,              -100_000_000,             std.math.maxInt(i32),
        std.math.minInt(i32), std.math.maxInt(i32) - 5, std.math.minInt(i32) + 5, 0x1234_5678,
    };
    for (0..height) |y| {
        for (0..width) |x| {
            const idx = y * width + x;
            src0[y * stride_i32 + x] = samples[idx % samples.len];
            src1[y * stride_i32 + x] = samples[(idx * 5 + 3) % samples.len];
        }
    }

    const srcs = [_][*]const u8{ @ptrCast(src0.ptr), @ptrCast(src1.ptr) };
    const strides = [_]isize{ stride_bytes, stride_bytes };

    compiled.kernel.func(@ptrCast(dst_jit.ptr), stride_bytes, &srcs, &strides, 0, null);
    try interp.evalPlane(allocator, &prog, profile, .{
        .dst = @ptrCast(dst_ref.ptr),
        .dst_stride = stride_bytes,
        .dst_fmt = fmt32,
        .srcs = &srcs,
        .src_strides = &strides,
        .src_fmts = &fmts,
        .width = width,
        .height = height,
        .frame_n = 0,
        .props = &.{},
    });

    for (0..height) |y| {
        for (0..width) |x| {
            try std.testing.expectEqual(dst_ref[y * stride_i32 + x], dst_jit[y * stride_i32 + x]);
        }
    }
}

/// Float -> 32-bit integer D4 store parity across qNaN, +-sNaN, +-inf, and 2^31 boundary values.
fn verifyBitExactF32ToI32(allocator: std.mem.Allocator, cpu: cpu_mod.CpuInfo, expr_str: []const u8, width: usize, height: usize) !void {
    const parser = @import("../expr/parser.zig");
    const lower_mod = @import("../expr/lower.zig");
    const optimize_mod = @import("../expr/optimize.zig");

    const profile = interp.TargetProfile.fromCpu(cpu);
    const fmt_f32 = ast.SampleFormat{ .is_float = true, .bits_per_sample = 32 };
    const fmt32 = ast.SampleFormat{ .is_float = false, .bits_per_sample = 32 };
    const fmts = [_]ast.SampleFormat{ fmt_f32, fmt_f32 };

    var prog = try parser.parse(allocator, expr_str, .{ .num_clips = 2 });
    defer prog.deinit(allocator);
    try lower_mod.lower(allocator, &prog, profile);
    try optimize_mod.optimize(allocator, &prog, profile, .{
        .enable_value_changing = true,
        .enable_fma = profile.use_fma,
    });

    var info = try analysis_mod.analyze(allocator, &prog, &fmts, fmt32, @intCast(width), @intCast(height), .{ .has_fast_u8_lut = aj.Builder.cpuSupportsFastLutU8(cpu), .f32_lanes = cpu.f32_lanes });
    defer info.deinit(allocator);

    var rt = try aj.Runtime.init();
    defer rt.deinit();

    var compiled = try compilePlane(allocator, &rt, cpu, &prog, &info, .{
        .width = width,
        .height = height,
        .src_fmts = &fmts,
        .dst_fmt = fmt32,
    });
    defer compiled.deinit(allocator, &rt);

    const stride_w = std.mem.alignForward(usize, width, 16);
    const stride_bytes: isize = @intCast(stride_w * 4);

    const src0 = try allocator.alignedAlloc(f32, .@"64", stride_w * height);
    defer allocator.free(src0);
    const src1 = try allocator.alignedAlloc(f32, .@"64", stride_w * height);
    defer allocator.free(src1);
    const dst_jit = try allocator.alignedAlloc(i32, .@"64", stride_w * height);
    defer allocator.free(dst_jit);
    const dst_ref = try allocator.alignedAlloc(i32, .@"64", stride_w * height);
    defer allocator.free(dst_ref);

    const pos_snan: f32 = @bitCast(@as(u32, 0x7FA0_0001));
    const neg_snan: f32 = @bitCast(@as(u32, 0xFFA0_0001));
    const edge_vals = [_]f32{
        -1e12,              -100.0,            -0.5,     -0.0,         0.0,
        0.5,                1.5,               12345.5,  2147483520.0, 2147483648.0,
        3e9,                std.math.nan(f32), pos_snan, neg_snan,     std.math.inf(f32),
        -std.math.inf(f32),
    };
    const n_edge = edge_vals.len;
    for (0..height) |y| {
        for (0..width) |x| {
            const idx = y * width + x;
            src0[y * stride_w + x] = edge_vals[idx % n_edge];
            src1[y * stride_w + x] = edge_vals[(idx + idx / n_edge) % n_edge];
        }
    }

    const srcs = [_][*]const u8{ @ptrCast(src0.ptr), @ptrCast(src1.ptr) };
    const strides = [_]isize{ stride_bytes, stride_bytes };

    compiled.kernel.func(@ptrCast(dst_jit.ptr), stride_bytes, &srcs, &strides, 0, null);
    try interp.evalPlane(allocator, &prog, profile, .{
        .dst = @ptrCast(dst_ref.ptr),
        .dst_stride = stride_bytes,
        .dst_fmt = fmt32,
        .srcs = &srcs,
        .src_strides = &strides,
        .src_fmts = &fmts,
        .width = width,
        .height = height,
        .frame_n = 0,
        .props = &.{},
    });

    for (0..height) |y| {
        for (0..width) |x| {
            try std.testing.expectEqual(dst_ref[y * stride_w + x], dst_jit[y * stride_w + x]);
        }
    }
}

test "JIT-vs-oracle bit-exact parity across ISA tiers, widths, spatial, FMA, opt=1, and transcendentals" {
    const allocator = std.testing.allocator;
    const exprs = [_][]const u8{
        "x y + 2 /",
        "x 3 %",
        "x round",
        "x trunc",
        "x floor",
        "x sqrt",
        "x abs y sgn +",
        "x y min",
        "x y max",
        "x 0 1 clip",
        "x y > x y < + x y = +",
        "x not y and x y or xor",
        "x y bitand x bitnot bitxor",
        "x y > 10 20 ?",
        "x 10 20 ?",
        "X Y + N + width + height + pi + x.A +",
        "x y 1 2 3 4 5 6 7 sort9 drop4 drop4",
        "x exp",
        "x log",
        "x sin",
        "x cos",
        "x tan",
        "x 6.24 * tan",
        "x y atan2",
        "x abs 0.01 + y 0.5 * pow",
        // Fused FMA expressions
        "x 0.75 * y 0.25 * + 0.1 +",
        "x y * 1.5 -",
        "1.5 x y * -",
        "x 2.5 * y 0.5 * - 1.25 +",
        // Spatial relative, extreme offsets (V1), wide reach (V12), and dynamic gather
        "x[-1,0]:c x[1,0]:m + x[0,-1]:c + x[0,1]:m +",
        "x[-9,2]:m x[9,-2]:c +",
        "x[-2147483648,0] x max",
        "y x x[]:c",
        "y x x[]:m",
    };

    var cpus: [5]cpu_mod.CpuInfo = undefined;
    var n_cpus: usize = 0;

    switch (cpu_mod.IsaFamily.host()) {
        .x86 => {
            const host = cpu_mod.detectWithOverride(null);
            cpus[n_cpus] = cpu_mod.detectWithOverride(1);
            n_cpus += 1;
            if (@backingInt(host.isa) >= @backingInt(cpu_mod.Isa.v2)) {
                cpus[n_cpus] = cpu_mod.detectWithOverride(2);
                n_cpus += 1;
            }
            if (@backingInt(host.isa) >= @backingInt(cpu_mod.Isa.v3)) {
                cpus[n_cpus] = cpu_mod.detectWithOverride(3);
                n_cpus += 1;
            }
            if (@backingInt(host.isa) >= @backingInt(cpu_mod.Isa.v4)) {
                cpus[n_cpus] = cpu_mod.detectWithOverride(4);
                n_cpus += 1;
            }
        },
        .aarch64 => {
            cpus[n_cpus] = cpu_mod.detectWithOverride(null);
            n_cpus += 1;
        },
    }

    const widths = [_]usize{ 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 33 };
    for (cpus[0..n_cpus]) |cpu| {
        for (exprs) |e| {
            try verifyBitExactF32(allocator, cpu, e, 17, 3);
        }
        for (widths) |w| {
            try verifyBitExactF32(allocator, cpu, "x 3 % y + round", w, 2);
            try verifyBitExactF32(allocator, cpu, "x[-3,0]:m x[5,1]:c +", w, 2);
            try verifyBitExactU8(allocator, cpu, "x y 0.1 * +", w, 2, false);
        }
        // V1/V2/V12 regressions: wide 1920 u8 spatial load (V2 AArch64 4-byte load + V12 wide reach)
        try verifyBitExactU8(allocator, cpu, "x[1,0] 0.5 *", 1920, 2, false);
        try verifyBitExactU8(allocator, cpu, "x[-65,0] x[65,0] max x min", 64, 2, false);
        try verifyBitExactU8(allocator, cpu, "x[-65,0] x[65,0] max x min", 192, 2, false);
        try verifyBitExactU8(allocator, cpu, "x[-65,0] x[65,0] max x min", 1920, 2, false);
        try verifyBitExactU8(allocator, cpu, "x[-2147483648,0] x max", 65, 2, false);
        // Bare-load select condition in a > 48-node 5x5 kernel: 32-reg i16/i32 load remat must keep the condition load.
        try verifyBitExactU8(allocator, cpu, "x[-2,-2] 7 x[-1,-2] x[0,-2] + x[1,-2] + x[2,-2] + x[-2,-1] + x[-1,-1] + x[0,-1] + x[1,-1] + x[2,-1] + x[-2,0] + x[-1,0] + x[0,0] + x[1,0] + x[2,0] + x[-2,1] + x[-1,1] + x[0,1] + x[1,1] + x[2,1] + x[-2,2] + x[-1,2] + x[0,2] + x[1,2] + x[2,2] + ?", 300, 8, false);
        // K4 & M2/M3: f32 -> u8 D4 store parity with qNaN, +-sNaN, +-inf, rounding, and out-of-range inputs
        // (width 225 covers all 15x15 = 225 (x, y) edge-value pairs on x86 and AArch64).
        const d4_exprs = [_][]const u8{ "x", "x trunc", "x floor", "x ceil", "x round", "x[1,0] x max", "x y min", "x y max", "x sqrt", "x exp", "x y +" };
        for ([_]usize{ 1, 4, 8, 16, 17, 225 }) |w| {
            for (d4_exprs) |e| {
                try verifyBitExactF32ToU8(allocator, cpu, e, w, 1);
            }
        }
        // M3 & rotate-plan P3: full 18x18 = 324 (x, y) pair sweep on f32 output for logic, min/max,
        // select (incl. compare-select fusion on +-0.0/NaN/multi-use), const-mem NaN order, rounding,
        // and new operators (neg, ceil, atan, asin, acos, lerp, polyvalN/polyevalN).
        for ([_][]const u8{
            "x y min",
            "x y max",
            "x not y and x y or xor",
            "x y > 10 20 ?",
            "x 10 20 ?",
            "x 0 < y 0 ?",
            "x 0 < 0 y ?",
            "x 0 < y -0.0 ?",
            "x 0 < dup 10 20 ? +",
            "x 0 > y 0 > and 10 20 ?",
            "x 0 > y 0 > or 0 255 ?",
            "x y xor 255 0 ?",
            "x not 10 20 ?",
            "x y < x 0 > or 10 0 ?",
            "x 8 %",
            "x 0.25 %",
            "x -4 %",
            "x 0 min 0 x min + x 0 max + 0 x max + x 1.5 - + 1.5 x - +",
            "x round",
            "x trunc",
            "x floor",
            "x ceil",
            "x neg",
            "x atan",
            "x asin",
            "x acos",
            "x y 0.3 lerp",
            "x y x y - lerp",
            "5 x polyval0",
            "2 3 x polyval1",
            "3 2 1 x polyval2",
            "0.5 -0.25 0.75 0.1 x polyval3",
            "0.1 -0.2 0.3 -0.4 0.5 0.6 x polyeval5",
            "0.1 -0.2 0.3 -0.4 0.5 -0.6 0.7 0.8 x polyval7",
            "0.05 -0.1 0.15 -0.2 0.25 -0.3 0.35 -0.4 0.45 -0.5 0.55 0.6 x polyval11",
            "x y hypot",
            "16 75 x linstep",
            "y x 10 + x linstep",
            "16 75 x smoothstep",
            "y x 10 + x smoothstep",
            "x y difflimit",
            "x y absdiff",
            "x y copysign",
            "x y minabs",
            "x y maxabs",
            "x exp2",
            "x log2",
            "x y pow",
        }) |e| {
            try verifyBitExactF32(allocator, cpu, e, 324, 1);
        }
        // Hardware-refined rsqrt (rsqrtps/vrsqrtps/vrsqrt14ps + 1 NR on x86, frsqrte + 2x frsqrts on NEON):
        // verify <= 5e-7 relative error across all ISA tiers and finite output on x <= 0.
        {
            const parser = @import("../expr/parser.zig");
            const lower_mod = @import("../expr/lower.zig");
            const optimize_mod = @import("../expr/optimize.zig");
            var rt_rs = try aj.Runtime.init();
            defer rt_rs.deinit();
            const profile = interp.TargetProfile.fromCpu(cpu);
            var prog_rs = try parser.parse(allocator, "x rsqrt", .{ .num_clips = 1 });
            defer prog_rs.deinit(allocator);
            try lower_mod.lower(allocator, &prog_rs, profile);
            try optimize_mod.optimize(allocator, &prog_rs, profile, .{});
            const f32_fmts = [_]ast.SampleFormat{.{ .is_float = true, .bits_per_sample = 32 }};
            var info_rs = try analysis_mod.analyze(allocator, &prog_rs, &f32_fmts, f32_fmts[0], 17, 1, .{ .has_fast_u8_lut = aj.Builder.cpuSupportsFastLutU8(cpu), .f32_lanes = cpu.f32_lanes });
            defer info_rs.deinit(allocator);
            var comp_rs = try compilePlane(allocator, &rt_rs, cpu, &prog_rs, &info_rs, .{
                .width = 17,
                .height = 1,
                .src_fmts = &f32_fmts,
                .dst_fmt = f32_fmts[0],
            });
            defer comp_rs.deinit(allocator, &rt_rs);
            var src_rs: [32]f32 align(64) = @splat(1.0);
            const init_rs = [_]f32{ -1.0, 0.0, 1e-10, 0.0625, 0.25, 0.5, 1.0, 2.0, 4.0, 9.0, 16.0, 100.0, 1e4, 1e10, 0.333, 7.5, std.math.inf(f32) };
            @memcpy(src_rs[0..init_rs.len], &init_rs);
            var dst_rs: [32]f32 align(64) = @splat(0);
            const srcs_rs = [_][*]const u8{@ptrCast(&src_rs)};
            const strides_rs = [_]isize{128};
            const props_rs = [_]f32{0.0};
            comp_rs.kernel.func(@ptrCast(&dst_rs), 128, &srcs_rs, &strides_rs, 0, &props_rs);
            for (init_rs, dst_rs[0..init_rs.len]) |xv, got| {
                try std.testing.expect(std.math.isFinite(got) and got > 0.0);
                const clamped: f64 = @min(@max(@as(f64, xv), @as(f64, interp.flt_min_f32)), @as(f64, interp.flt_max_f32));
                const ref: f64 = 1.0 / @sqrt(clamped);
                try std.testing.expect(@abs(@as(f64, got) - ref) <= 5e-7 * ref);
            }
        }
        // Rotate-plan P3: last-pixel bounds safety on 32B/64B-aligned widths, shared X/Y load_abs cache
        // (opt=0/1, clamp/mirror, multi-clip), and root-invariant frame/row LICM + spatial strips.
        try verifyBitExactU8(allocator, cpu, "width 1 - height 1 - x[]:c", 64, 3, false);
        try verifyBitExactU8(allocator, cpu, "width 1 - height 1 - x[]:c", 128, 3, false);
        try verifyBitExactF32(allocator, cpu, "width 1 - height 1 - x[]:c", 64, 3);
        try verifyBitExactU8(allocator, cpu, "X 1 - Y x[]:m X 1 - Y y[]:m + X 1 + Y x[]:m +", 65, 3, true);
        try verifyBitExactF32(allocator, cpu, "X width 2 / - Y height 2 / - x[]:m X width 2 / - Y 1 + y[]:m +", 33, 3);
        try verifyBitExactF32(allocator, cpu, "N 2 * 1 +", 33, 3);
        try verifyBitExactF32(allocator, cpu, "Y 0.5 * N +", 33, 3);
        try verifyBitExactF32(allocator, cpu, "x[-1,0] Y + N sin +", 33, 3);
        // Periodic `X % C` + hoisted-root pre-packing (`deblock_qed` mask), i16 cmp/logic/select,
        // and native u8 saturating add/sub parity across all widths and ISA tiers (incl. AR1 two-phase tails).
        for ([_]usize{ 1, 3, 4, 5, 6, 7, 8, 9, 12, 13, 15, 16, 17, 25, 31, 32, 33, 63, 64, 65 }) |w| {
            try verifyBitExactU8(allocator, cpu, "X 8 % 7 % Y 8 % 7 % and 0 255 ?", w, 9, false);
            try verifyBitExactU8(allocator, cpu, "X 4 % 3 % Y 4 % 3 % and 0 255 ?", w, 5, false);
            try verifyBitExactU8(allocator, cpu, "X 16 % 16 *", w, 3, false);
            try verifyBitExactU8(allocator, cpu, "X 32 % 4 *", w, 3, false);
            try verifyBitExactU8(allocator, cpu, "X 8 % x +", w, 3, false);
            try verifyBitExactU8(allocator, cpu, "X 16 % x +", w, 3, false);
            try verifyBitExactU8(allocator, cpu, "X 32 % x +", w, 3, false);
            try verifyBitExactF32(allocator, cpu, "X 8 %", w, 3);
            try verifyBitExactF32(allocator, cpu, "X 16 % 4 *", w, 3);
            try verifyBitExactF32(allocator, cpu, "X 32 % 2 *", w, 3);
            try verifyBitExactF32(allocator, cpu, "X 8 % 7 % Y 8 % 7 % and 0 1 ?", w, 9);
            try verifyBitExactF32(allocator, cpu, "X 8 % x +", w, 3);
            try verifyBitExactF32(allocator, cpu, "X 16 % x +", w, 3);
            try verifyBitExactF32(allocator, cpu, "X 32 % x +", w, 3);
            try verifyBitExactF32(allocator, cpu, "X 16 % x * y +", w, 3);
            try verifyBitExactI32(allocator, cpu, "X 8 % x +", w, 3, false);
            try verifyBitExactI32(allocator, cpu, "X 16 % 4 *", w, 3, false);
            try verifyBitExactI32(allocator, cpu, "X 32 % x +", w, 3, false);
        }
        // AR1: also verify u16 & f16 two-phase `X % C` roots and pixel tails with tight 32B/64B strides.
        {
            const parser = @import("../expr/parser.zig");
            const lower_mod = @import("../expr/lower.zig");
            const optimize_mod = @import("../expr/optimize.zig");
            const prof = interp.TargetProfile.fromCpu(cpu);
            var rt_ar1 = try aj.Runtime.init();
            defer rt_ar1.deinit();
            const ar1_fmts = [_]ast.SampleFormat{
                .{ .is_float = false, .bits_per_sample = 16 },
                .{ .is_float = true, .bits_per_sample = 16 },
            };
            const ar1_exprs = [_][]const u8{
                "X 8 % 7 % Y 8 % 7 % and 0 1 ?",
                "X 16 % 4096 *",
                "X 32 % 2 *",
                "X 8 % x +",
                "X 16 % x +",
                "X 32 % x +",
            };
            const stride_align: usize = if (cpu.has_avx512) 64 else 32;
            for (ar1_fmts) |fmt16| {
                if (fmt16.is_float and !cpu.has_f16c and cpu.isa != .neon) continue;
                const fmts16 = [_]ast.SampleFormat{fmt16};
                for (ar1_exprs) |e| {
                    var p16 = try parser.parse(allocator, e, .{ .num_clips = 1 });
                    defer p16.deinit(allocator);
                    try lower_mod.lower(allocator, &p16, prof);
                    try optimize_mod.optimize(allocator, &p16, prof, .{ .enable_value_changing = true, .enable_fma = prof.use_fma });
                    for ([_]usize{ 1, 5, 7, 8, 9, 13, 15, 16, 17, 25, 31, 32, 33, 63, 65 }) |w| {
                        var inf16 = try analysis_mod.analyze(allocator, &p16, &fmts16, fmt16, @intCast(w), 3, .{ .has_fast_u8_lut = aj.Builder.cpuSupportsFastLutU8(cpu), .f32_lanes = cpu.f32_lanes });
                        defer inf16.deinit(allocator);
                        var comp16 = try compilePlane(allocator, &rt_ar1, cpu, &p16, &inf16, .{
                            .width = w,
                            .height = 3,
                            .src_fmts = &fmts16,
                            .dst_fmt = fmt16,
                        });
                        defer comp16.deinit(allocator, &rt_ar1);
                        const row_bytes = std.mem.alignForward(usize, w * 2, stride_align);
                        const row_elems = row_bytes / 2;
                        const src16 = try allocator.alignedAlloc(u16, .@"64", row_elems * 3);
                        defer allocator.free(src16);
                        const dj16 = try allocator.alignedAlloc(u16, .@"64", row_elems * 3);
                        defer allocator.free(dj16);
                        const dr16 = try allocator.alignedAlloc(u16, .@"64", row_elems * 3);
                        defer allocator.free(dr16);
                        for (0..row_elems * 3) |i| {
                            src16[i] = if (fmt16.is_float) 0x3800 else @intCast((i * 137 + 19) & 0x3fff);
                            dj16[i] = 0x5A5A;
                            dr16[i] = 0x5A5A;
                        }
                        const s16 = [_][*]const u8{@ptrCast(src16.ptr)};
                        const st16 = [_]isize{@intCast(row_bytes)};
                        comp16.kernel.func(@ptrCast(dj16.ptr), @intCast(row_bytes), &s16, &st16, 0, null);
                        try interp.evalPlane(allocator, &p16, prof, .{
                            .dst = @ptrCast(dr16.ptr),
                            .dst_stride = @intCast(row_bytes),
                            .dst_fmt = fmt16,
                            .srcs = &s16,
                            .src_strides = &st16,
                            .src_fmts = &fmts16,
                            .width = w,
                            .height = 3,
                            .frame_n = 0,
                            .props = &.{},
                        });
                        for (0..3) |y| {
                            const off = y * row_elems;
                            try std.testing.expectEqualSlices(u16, dr16[off .. off + w], dj16[off .. off + w]);
                        }
                    }
                }
            }
        }
        for ([_][]const u8{
            "x y -",
            "x y +",
            "x y - 0 max",
            "x y + 255 min",
            "x[-1,0] x max x[1,0] max x[-1,0] x min x[1,0] min -",
            "x y - 10 < 0 255 ?",
            "x y < x y ?",
            "x y <= x y ?",
            "x y >= x y ?",
            "x y = 200 10 ?",
            "x y and 255 0 ?",
            "x y or 0 255 ?",
            "x y xor 10 20 ?",
            "x not 255 0 ?",
            "x y - abs y 128 - abs < x y ?",
            "y x - D1! y 128 - D2! D1@ D2@ xor x x D1@ abs D2@ abs < D1@ D2@ ? - ?",
        }) |e| {
            try verifyBitExactU8(allocator, cpu, e, 65, 3, false);
        }
        if (cpu.isa == .v3) {
            // P1: verify AVX2 hardware vpgatherdd path (`ZEXPR_HW_GATHER=1`) alongside default register gather.
            var hw_cpu = cpu;
            hw_cpu.prefer_hw_gather_f32 = true;
            try verifyBitExactF32(allocator, hw_cpu, "y x x[]:c", 33, 3);
            try verifyBitExactF32(allocator, hw_cpu, "y x x[]:m", 33, 3);
        }
        // V3/V4/V7 `opt=1` typing regressions & i32 compare-select fusion (all 5 ops + 0-branches + multi-use)
        try verifyBitExactU8(allocator, cpu, "x 4 pow", 33, 2, true);
        try verifyBitExactU8(allocator, cpu, "x neg 255 +", 33, 2, true);
        try verifyBitExactU8(allocator, cpu, "x y 2 lerp", 33, 2, true);
        try verifyBitExactU8(allocator, cpu, "3 2 1 x polyval2", 33, 2, true);
        try verifyBitExactU8(allocator, cpu, "1 x 0.5 ? 65536 * 65536 *", 33, 2, true);
        try verifyBitExactU8(allocator, cpu, "x 0.5 + 0 > 2147483647 * 1 +", 33, 2, true);
        for ([_][]const u8{
            "x y > x y ?",
            "x y < x y ?",
            "x y >= x y ?",
            "x y <= x y ?",
            "x y = 200 10 ?",
            "x 128 > y 0 ?",
            "x 128 <= y 0 ?",
            "x 128 > 0 y ?",
            "x 128 >= 0 y ?",
            "x 128 > dup 100 20 ? +",
        }) |e| {
            try verifyBitExactU8(allocator, cpu, e, 65, 2, true);
        }
        // 32-bit integer format parity across opt=0, opt=1, wraparound, spatial rel/abs, and D4 edge floats
        for ([_]usize{ 1, 4, 7, 8, 9, 16, 17, 33 }) |w| {
            try verifyBitExactI32(allocator, cpu, "x 1 +", w, 2, true);
            try verifyBitExactI32(allocator, cpu, "x -10 + y bitxor", w, 2, true);
            try verifyBitExactI32(allocator, cpu, "x y > x y ?", w, 2, true);
            try verifyBitExactI32(allocator, cpu, "x[-1,0]:c x[1,0]:m +", w, 2, true);
            try verifyBitExactI32(allocator, cpu, "width 1 - X - Y x[]:c", w, 2, true);
            try verifyBitExactI32(allocator, cpu, "x 0.5 +", w, 2, true);
            try verifyBitExactI32(allocator, cpu, "x y + 0.5 *", w, 2, false);
        }
        for ([_]usize{ 1, 4, 8, 16, 17, 256 }) |w| {
            for ([_][]const u8{ "x", "x y +", "x y min", "x y max", "x round" }) |e| {
                try verifyBitExactF32ToI32(allocator, cpu, e, w, 1);
            }
        }
    }
}
