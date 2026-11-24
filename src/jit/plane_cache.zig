//! Bounded thread-safe refcounted LRU cache for compiled `zexpr.Expr` planes.
//!
//! Caches dimension-specialized JIT kernels (with frame property bindings) and
//! precomputed 1D/2D LUTs across `Expr` instances (e.g. `FrameEval` callbacks
//! and identical U/V planes), sharing a single lazily-initialized `aj.Runtime`
//! so executable pages are reused without per-frame `mmap`/`munmap` syscalls.

const std = @import("std");
const ast = @import("../expr/ast.zig");
const builtins_mod = @import("../expr/builtins.zig");
const interp = @import("../expr/interp.zig");
const aj = @import("asmjit.zig");
const cpu_mod = @import("cpu.zig");
const lut_mod = @import("lut.zig");

pub const default_capacity: usize = 512;
/// 16 MiB LUT byte budget (matches the old 64 x 256 KiB worst case while allowing up to 512 small JIT entries).
pub const default_max_lut_bytes: usize = 16 * 1024 * 1024;

const cache_io: std.Io = std.Io.Threaded.global_single_threaded.io();

pub const PropBinding = struct {
    clip: ast.ClipId,
    name_z: [:0]u8,
};

pub fn freePropBindings(allocator: std.mem.Allocator, props: []PropBinding) void {
    for (props) |p| allocator.free(p.name_z);
    allocator.free(props);
}

pub fn buildPropBindings(allocator: std.mem.Allocator, prog: *const ast.Program) std.mem.Allocator.Error![]PropBinding {
    const out = try allocator.alloc(PropBinding, prog.props.items.len);
    var built: usize = 0;
    errdefer {
        for (out[0..built]) |p| allocator.free(p.name_z);
        allocator.free(out);
    }
    for (prog.props.items, 0..) |p, i| {
        const fp = switch (p) {
            .frame_prop => |fp| fp,
            // D2: Expr always resolves built-ins at parse time; `.builtin` slots exist only in Select/PropExpr.
            .builtin => unreachable,
        };
        out[i] = .{
            .clip = fp.clip,
            .name_z = try allocator.dupeSentinel(u8, fp.name, 0),
        };
        built += 1;
    }
    return out;
}

pub const Key = struct {
    expr: []const u8,
    src_fmts: []const ast.SampleFormat,
    dst_fmt: ast.SampleFormat,
    width: usize,
    height: usize,
    opt_int: bool,
    default_boundary: ast.BoundaryMode,
    cpu: cpu_mod.CpuInfo,
    profile: interp.TargetProfile,
    resolved_builtins: []const builtins_mod.BuiltinVal = &.{},
};

pub const CompiledPlane = union(enum) {
    lut: lut_mod.Lut,
    jit: struct {
        kernel: aj.CompiledKernel,
        props: []PropBinding,
    },
    /// F3: Dynamic-mode unsupported-JIT fallback (e.g. non-LUT `f16` on CPUs without F16C,
    /// or `ExecMemDenied` without fallback LUT). Level 0 (`force_scalar`) stays outside the cache.
    scalar: struct {
        prog: ast.Program,
        props: []PropBinding,
    },

    pub fn deinit(self: *CompiledPlane, allocator: std.mem.Allocator, rt: ?*aj.Runtime) void {
        switch (self.*) {
            .lut => |*l| l.deinit(allocator, rt),
            .jit => |*j| {
                if (rt) |runtime| runtime.release(j.kernel.func);
                freePropBindings(allocator, j.props);
            },
            .scalar => |*s| {
                s.prog.deinit(allocator);
                freePropBindings(allocator, s.props);
            },
        }
        self.* = undefined;
    }
};

pub const Entry = struct {
    hash: u64,
    key: Key,
    exec: CompiledPlane,
    ref_count: usize,
    last_used: u64,
};

/// Heap-allocated fallback scalar program so `SinglePlaneExec` stays pointer-sized (16 bytes).
pub const ScalarExec = struct {
    prog: ast.Program,
    props: []PropBinding,

    pub fn deinit(self: *ScalarExec, allocator: std.mem.Allocator) void {
        self.prog.deinit(allocator);
        freePropBindings(allocator, self.props);
        allocator.destroy(self);
    }
};

/// Single compiled plane handle (`Expr` static mode); kept pointer-sized so `ExprData` stays compact.
pub fn SinglePlaneExec(comptime CacheType: type) type {
    return union(enum) {
        copy,
        undefined_plane,
        cached: *Entry,
        scalar: *ScalarExec,

        pub fn deinit(self: *@This(), allocator: std.mem.Allocator, cache: *CacheType) void {
            switch (self.*) {
                .copy, .undefined_plane => {},
                .cached => |entry| cache.release(allocator, entry),
                .scalar => |s| s.deinit(allocator),
            }
            self.* = .copy;
        }
    };
}

/// F2 + E2: Precompiled `2^k` (`1 <= k <= 2`) limited/full range variants for constant-format planes (C1).
/// `clip_fmts[i]` stores an integer format descriptor (using `dst`'s format when clip `0` is float and
/// `dst` is integer so clip `0`'s shared bit never forces full range). Heap-allocated only for planes that need it.
pub fn RangeVariants(comptime CacheType: type, comptime FormatMeta: type) type {
    const Single = SinglePlaneExec(CacheType);
    return struct {
        clip_ids: [2]ast.ClipId = .{ 0, 0 },
        clip_fmts: [2]FormatMeta = undefined,
        clip_count: u8 = 0,
        init_count: u8 = 0,
        variants: [4]Single = .{ .copy, .copy, .copy, .copy },

        pub fn deinit(self: *@This(), allocator: std.mem.Allocator, cache: *CacheType) void {
            for (self.variants[0..self.init_count]) |*v| {
                v.deinit(allocator, cache);
            }
            self.init_count = 0;
            self.clip_count = 0;
        }
    };
}

pub fn PlaneCache(comptime capacity: usize) type {
    return PlaneCacheWithLutCap(capacity, default_max_lut_bytes);
}

fn entryLutBytes(exec: *const CompiledPlane) usize {
    return switch (exec.*) {
        .lut => |*l| l.table.len,
        .jit, .scalar => 0,
    };
}

pub fn PlaneCacheWithLutCap(comptime capacity: usize, comptime max_lut_bytes: usize) type {
    comptime std.debug.assert(capacity > 0);
    return struct {
        const Self = @This();

        mutex: std.Io.Mutex = .init,
        rt: ?aj.Runtime = null,
        /// Packed active entries in `slots[0..count]` with parallel `hashes[0..count]` so lookups
        /// scan only live entries without chasing cold pointers on hash misses.
        slots: [capacity]?*Entry = @splat(null),
        hashes: [capacity]u64 = @splat(0),
        count: usize = 0,
        lut_bytes: usize = 0,
        stamp: u64 = 0,
        hits: u64 = 0,
        misses: u64 = 0,

        pub fn ensureRuntime(self: *Self) ?*aj.Runtime {
            self.mutex.lockUncancelable(cache_io);
            defer self.mutex.unlock(cache_io);
            if (self.rt == null) {
                self.rt = aj.Runtime.init() catch null;
            }
            return if (self.rt != null) &self.rt.? else null;
        }

        fn runtimePtrLocked(self: *Self) ?*aj.Runtime {
            return if (self.rt != null) &self.rt.? else null;
        }

        /// Look up a compiled plane by key. On hit, increments `ref_count` and
        /// returns the entry (caller must release it with `release`).
        pub fn get(self: *Self, key: Key) ?*Entry {
            const h = hashKey(key);
            self.mutex.lockUncancelable(cache_io);
            defer self.mutex.unlock(cache_io);

            for (0..self.count) |i| {
                if (self.hashes[i] != h) continue;
                const e = self.slots[i].?;
                if (keyEql(e.key, key)) {
                    self.stamp +%= 1;
                    e.last_used = self.stamp;
                    e.ref_count += 1;
                    self.hits += 1;
                    return e;
                }
            }
            self.misses += 1;
            return null;
        }

        /// Single-lock lookup preferring an exact `(width, height)` match and falling back
        /// to a dimension-free (`0x0`) LUT entry in the same pass over live entries.
        pub fn getOrDim(self: *Self, key: Key, dim_key: Key) ?*Entry {
            const h_exact = hashKey(key);
            const h_dim = hashKey(dim_key);
            self.mutex.lockUncancelable(cache_io);
            defer self.mutex.unlock(cache_io);

            var dim_hit: ?*Entry = null;
            for (0..self.count) |i| {
                const hi = self.hashes[i];
                if (hi == h_exact) {
                    const e = self.slots[i].?;
                    if (keyEql(e.key, key)) {
                        self.stamp +%= 1;
                        e.last_used = self.stamp;
                        e.ref_count += 1;
                        self.hits += 1;
                        return e;
                    }
                }
                if (dim_hit == null and hi == h_dim) {
                    const e = self.slots[i].?;
                    if (keyEql(e.key, dim_key)) {
                        dim_hit = e;
                    }
                }
            }
            if (dim_hit) |e| {
                self.stamp +%= 1;
                e.last_used = self.stamp;
                e.ref_count += 1;
                self.hits += 1;
                return e;
            }
            self.misses += 1;
            return null;
        }

        /// Insert a freshly compiled plane into the LRU cache (taking ownership of
        /// `exec` on success) and return a refcounted handle for the caller.
        /// If another thread inserted the same key concurrently, destroys the
        /// duplicate `exec` and returns a handle to the existing entry.
        /// On OOM while allocating the cache entry, leaves `exec` untouched for the caller.
        pub fn putOrGet(self: *Self, allocator: std.mem.Allocator, key: Key, exec: CompiledPlane) std.mem.Allocator.Error!*Entry {
            const h = hashKey(key);
            const new_lut_bytes = entryLutBytes(&exec);

            const owned_expr = try allocator.dupe(u8, key.expr);
            errdefer allocator.free(owned_expr);

            const owned_src_fmts = try allocator.dupe(ast.SampleFormat, key.src_fmts);
            errdefer allocator.free(owned_src_fmts);

            const owned_builtins: []const builtins_mod.BuiltinVal = if (key.resolved_builtins.len > 0)
                try allocator.dupe(builtins_mod.BuiltinVal, key.resolved_builtins)
            else
                &.{};
            errdefer if (owned_builtins.len > 0) allocator.free(owned_builtins);

            const new_entry = try allocator.create(Entry);
            errdefer allocator.destroy(new_entry);

            var owned_key = key;
            owned_key.expr = owned_expr;
            owned_key.src_fmts = owned_src_fmts;
            owned_key.resolved_builtins = owned_builtins;

            var duplicate_exec: ?CompiledPlane = null;
            var dead_buf: [capacity]*Entry = undefined;
            var dead_count: usize = 0;
            var rt_ptr: ?*aj.Runtime = null;

            const result: *Entry = blk: {
                self.mutex.lockUncancelable(cache_io);
                defer self.mutex.unlock(cache_io);
                rt_ptr = self.runtimePtrLocked();

                for (0..self.count) |i| {
                    if (self.hashes[i] != h) continue;
                    const e = self.slots[i].?;
                    if (keyEql(e.key, key)) {
                        self.stamp +%= 1;
                        e.last_used = self.stamp;
                        e.ref_count += 1;
                        duplicate_exec = exec;
                        break :blk e;
                    }
                }

                // Enforce the LUT byte cap by evicting LRU LUT entries before touching small JIT entries.
                if (new_lut_bytes > 0) {
                    while (self.lut_bytes + new_lut_bytes > max_lut_bytes) {
                        var lru_lut_idx: ?usize = null;
                        var min_used: u64 = std.math.maxInt(u64);
                        for (0..self.count) |i| {
                            const e = self.slots[i].?;
                            if (entryLutBytes(&e.exec) == 0) continue;
                            if (lru_lut_idx == null or e.last_used < min_used) {
                                min_used = e.last_used;
                                lru_lut_idx = i;
                            }
                        }
                        const idx = lru_lut_idx orelse break;
                        const evicted = self.slots[idx].?;
                        self.lut_bytes -= entryLutBytes(&evicted.exec);
                        evicted.ref_count -= 1;
                        if (evicted.ref_count == 0) {
                            dead_buf[dead_count] = evicted;
                            dead_count += 1;
                        }
                        const last = self.count - 1;
                        self.slots[idx] = self.slots[last];
                        self.hashes[idx] = self.hashes[last];
                        self.slots[last] = null;
                        self.hashes[last] = 0;
                        self.count = last;
                    }
                }

                var target_slot: usize = self.count;
                if (self.count == capacity) {
                    var min_used = self.slots[0].?.last_used;
                    target_slot = 0;
                    for (1..self.count) |i| {
                        const e = self.slots[i].?;
                        if (e.last_used < min_used) {
                            min_used = e.last_used;
                            target_slot = i;
                        }
                    }
                    const evicted = self.slots[target_slot].?;
                    self.lut_bytes -= entryLutBytes(&evicted.exec);
                    evicted.ref_count -= 1;
                    if (evicted.ref_count == 0) {
                        dead_buf[dead_count] = evicted;
                        dead_count += 1;
                    }
                } else {
                    self.count += 1;
                }

                self.stamp +%= 1;
                new_entry.* = .{
                    .hash = h,
                    .key = owned_key,
                    .exec = exec,
                    .ref_count = 2, // 1 for cache slot + 1 for caller
                    .last_used = self.stamp,
                };
                self.slots[target_slot] = new_entry;
                self.hashes[target_slot] = h;
                self.lut_bytes += new_lut_bytes;
                break :blk new_entry;
            };

            if (duplicate_exec) |*dup| {
                dup.deinit(allocator, rt_ptr);
                allocator.free(owned_expr);
                allocator.free(owned_src_fmts);
                if (owned_builtins.len > 0) allocator.free(owned_builtins);
                allocator.destroy(new_entry);
            }
            for (dead_buf[0..dead_count]) |dead| {
                destroyEntry(allocator, dead, rt_ptr);
            }
            return result;
        }

        /// Drop one caller reference to `entry`. Destroys the entry once evicted
        /// from the cache and no active filters hold a reference.
        pub fn release(self: *Self, allocator: std.mem.Allocator, entry: *Entry) void {
            var destroy = false;
            var rt_ptr: ?*aj.Runtime = null;
            {
                self.mutex.lockUncancelable(cache_io);
                defer self.mutex.unlock(cache_io);
                std.debug.assert(entry.ref_count > 0);
                entry.ref_count -= 1;
                if (entry.ref_count == 0) {
                    destroy = true;
                    rt_ptr = self.runtimePtrLocked();
                }
            }
            if (destroy) {
                destroyEntry(allocator, entry, rt_ptr);
            }
        }

        /// Release all cached entries and the shared `JitRuntime` (for tests;
        /// callers must have released all active entries first).
        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            self.mutex.lockUncancelable(cache_io);
            defer self.mutex.unlock(cache_io);
            const rt_ptr = self.runtimePtrLocked();
            for (0..self.count) |i| {
                const e = self.slots[i].?;
                e.ref_count -= 1;
                std.debug.assert(e.ref_count == 0);
                destroyEntry(allocator, e, rt_ptr);
                self.slots[i] = null;
                self.hashes[i] = 0;
            }
            self.count = 0;
            self.lut_bytes = 0;
            if (self.rt) |*r| {
                r.deinit();
                self.rt = null;
            }
        }
    };
}

fn destroyEntry(allocator: std.mem.Allocator, entry: *Entry, rt: ?*aj.Runtime) void {
    entry.exec.deinit(allocator, rt);
    allocator.free(entry.key.expr);
    allocator.free(entry.key.src_fmts);
    if (entry.key.resolved_builtins.len > 0) allocator.free(entry.key.resolved_builtins);
    allocator.destroy(entry);
}

fn hashKey(k: Key) u64 {
    var h = std.hash.Wyhash.init(0);
    h.update(std.mem.asBytes(&k.expr.len));
    h.update(k.expr);
    h.update(std.mem.asBytes(&k.src_fmts.len));
    for (k.src_fmts) |f| {
        const packed_fmt: u16 = (@as(u16, @intFromBool(f.is_float)) << 8) | f.bits_per_sample;
        h.update(std.mem.asBytes(&packed_fmt));
    }
    const packed_dst: u16 = (@as(u16, @intFromBool(k.dst_fmt.is_float)) << 8) | k.dst_fmt.bits_per_sample;
    h.update(std.mem.asBytes(&packed_dst));
    h.update(std.mem.asBytes(&k.width));
    h.update(std.mem.asBytes(&k.height));
    const flags: u8 =
        (@as(u8, @intFromBool(k.opt_int)) << 0) |
        (@as(u8, @backingInt(k.default_boundary)) << 1) |
        (@as(u8, @intFromBool(k.profile.use_fma)) << 2) |
        (@as(u8, @backingInt(k.profile.isa_family)) << 3);
    h.update(std.mem.asBytes(&flags));
    const cpu_isa: u32 = @backingInt(k.cpu.isa);
    h.update(std.mem.asBytes(&cpu_isa));
    h.update(std.mem.asBytes(&k.cpu.vec_bytes));
    h.update(std.mem.asBytes(&k.cpu.f32_lanes));
    const cpu_bits: u8 =
        (@as(u8, @intFromBool(k.cpu.has_ssse3)) << 0) |
        (@as(u8, @intFromBool(k.cpu.has_sse4_1)) << 1) |
        (@as(u8, @intFromBool(k.cpu.has_fma)) << 2) |
        (@as(u8, @intFromBool(k.cpu.has_f16c)) << 3) |
        (@as(u8, @intFromBool(k.cpu.has_avx2)) << 4) |
        (@as(u8, @intFromBool(k.cpu.has_avx512)) << 5) |
        (@as(u8, @intFromBool(k.cpu.prefer_hw_gather_f32)) << 6);
    h.update(std.mem.asBytes(&cpu_bits));
    // D5: hash resolved_builtins field by field (never raw struct bytes over `?i32` padding).
    h.update(std.mem.asBytes(&k.resolved_builtins.len));
    for (k.resolved_builtins) |bv| {
        bv.hashInto(&h);
    }
    return h.final();
}

fn fmtEql(a: ast.SampleFormat, b: ast.SampleFormat) bool {
    return a.is_float == b.is_float and a.bits_per_sample == b.bits_per_sample;
}

fn cpuEql(a: cpu_mod.CpuInfo, b: cpu_mod.CpuInfo) bool {
    return a.isa == b.isa and
        a.vec_bytes == b.vec_bytes and
        a.f32_lanes == b.f32_lanes and
        a.has_ssse3 == b.has_ssse3 and
        a.has_sse4_1 == b.has_sse4_1 and
        a.has_fma == b.has_fma and
        a.has_f16c == b.has_f16c and
        a.has_avx2 == b.has_avx2 and
        a.has_avx512 == b.has_avx512 and
        a.prefer_hw_gather_f32 == b.prefer_hw_gather_f32;
}

fn keyEql(a: Key, b: Key) bool {
    if (a.width != b.width or a.height != b.height) return false;
    if (a.opt_int != b.opt_int or a.default_boundary != b.default_boundary) return false;
    if (!fmtEql(a.dst_fmt, b.dst_fmt)) return false;
    if (a.profile.use_fma != b.profile.use_fma or a.profile.isa_family != b.profile.isa_family) return false;
    if (!cpuEql(a.cpu, b.cpu)) return false;
    if (!std.mem.eql(u8, a.expr, b.expr)) return false;
    if (a.src_fmts.len != b.src_fmts.len) return false;
    for (a.src_fmts, b.src_fmts) |fa, fb| {
        if (!fmtEql(fa, fb)) return false;
    }
    if (a.resolved_builtins.len != b.resolved_builtins.len) return false;
    for (a.resolved_builtins, b.resolved_builtins) |ba, bb| {
        if (!ba.eql(bb)) return false;
    }
    return true;
}

test "PlaneCache LRU hit, eviction, active ref survival, and key isolation" {
    const parser = @import("../expr/parser.zig");
    const lower_mod = @import("../expr/lower.zig");
    const optimize_mod = @import("../expr/optimize.zig");
    const analysis_mod = @import("../expr/analysis.zig");
    const codegen = @import("codegen.zig");

    const alloc = std.testing.allocator;
    const cpu = cpu_mod.detectWithOverride(null);
    if (cpu.isa == .scalar) return;
    const profile = interp.TargetProfile.fromCpu(cpu);

    var cache: PlaneCache(2) = .{};
    defer cache.deinit(alloc);
    const rt = cache.ensureRuntime() orelse return error.OutOfMemory;

    const u8_fmt: ast.SampleFormat = .{ .is_float = false, .bits_per_sample = 8 };
    const src_fmts = [_]ast.SampleFormat{u8_fmt};

    const compileOne = struct {
        fn run(
            a: std.mem.Allocator,
            c: *PlaneCache(2),
            r: *aj.Runtime,
            k: Key,
        ) !*Entry {
            if (c.get(k)) |hit| return hit;
            var prog = try parser.parse(a, k.expr, .{
                .mode = .expr,
                .num_clips = @intCast(k.src_fmts.len),
                .default_boundary = k.default_boundary,
                .opt_int = k.opt_int,
            });
            defer prog.deinit(a);
            try lower_mod.lower(a, &prog, k.profile);
            try optimize_mod.optimize(a, &prog, k.profile, .{
                .enable_value_changing = true,
                .enable_fma = k.profile.use_fma,
            });
            var info = try analysis_mod.analyze(a, &prog, k.src_fmts, k.dst_fmt, @intCast(k.width), @intCast(k.height), .{ .f32_lanes = k.cpu.f32_lanes });
            defer info.deinit(a);
            const props = try buildPropBindings(a, &prog);
            errdefer freePropBindings(a, props);
            const compiled = try codegen.compilePlane(a, r, k.cpu, &prog, &info, .{
                .width = k.width,
                .height = k.height,
                .src_fmts = k.src_fmts,
                .dst_fmt = k.dst_fmt,
            });
            var exec: CompiledPlane = .{ .jit = .{ .kernel = compiled.kernel, .props = props } };
            errdefer exec.deinit(a, r);
            return try c.putOrGet(a, k, exec);
        }
    }.run;

    const k1: Key = .{
        .expr = "x 1 +",
        .src_fmts = &src_fmts,
        .dst_fmt = u8_fmt,
        .width = 32,
        .height = 4,
        .opt_int = false,
        .default_boundary = .clamp,
        .cpu = cpu,
        .profile = profile,
    };
    var k2 = k1;
    k2.expr = "x 2 +";
    var k3 = k1;
    k3.expr = "x 3 +";
    var k1_wide = k1;
    k1_wide.width = 64;

    const e1 = try compileOne(alloc, &cache, rt, k1);
    const e1_hit = try compileOne(alloc, &cache, rt, k1);
    try std.testing.expectEqual(e1, e1_hit);
    try std.testing.expectEqual(1, cache.hits);
    cache.release(alloc, e1_hit);

    // Different width must miss even with identical expression.
    const e1_wide = try compileOne(alloc, &cache, rt, k1_wide);
    try std.testing.expect(e1_wide != e1);
    cache.release(alloc, e1_wide);

    // Touch k1_wide again, then insert k2 and k3 -> evicts e1 while e1 is still held by caller!
    const e2 = try compileOne(alloc, &cache, rt, k2);
    const e3 = try compileOne(alloc, &cache, rt, k3);
    cache.release(alloc, e2);
    cache.release(alloc, e3);

    // e1 must remain valid and executable even after LRU eviction because caller holds a ref!
    var src_buf: [32 * 4]u8 = @splat(10);
    var dst_buf: [32 * 4]u8 = @splat(0);
    const srcs = [_][*]const u8{&src_buf};
    const strides = [_]isize{32};
    e1.exec.jit.kernel.func(&dst_buf, 32, &srcs, &strides, 0, null);
    try std.testing.expectEqual(11, dst_buf[0]);
    try std.testing.expectEqual(11, dst_buf[127]);

    cache.release(alloc, e1);
}

test "PlaneCache resolved_builtins isolation, D5 i32_val distinction, and Phase A LUT sharing" {
    const parser = @import("../expr/parser.zig");
    const lower_mod = @import("../expr/lower.zig");
    const optimize_mod = @import("../expr/optimize.zig");
    const analysis_mod = @import("../expr/analysis.zig");
    const codegen = @import("codegen.zig");

    const alloc = std.testing.allocator;
    const cpu = cpu_mod.detectWithOverride(null);
    if (cpu.isa == .scalar) return;
    const profile = interp.TargetProfile.fromCpu(cpu);

    var cache: PlaneCache(8) = .{};
    defer cache.deinit(alloc);
    const rt = cache.ensureRuntime() orelse return error.OutOfMemory;

    const f32_fmt: ast.SampleFormat = .{ .is_float = true, .bits_per_sample = 32 };
    const u16_fmt: ast.SampleFormat = .{ .is_float = false, .bits_per_sample = 16 };
    const f32_srcs = [_]ast.SampleFormat{f32_fmt};
    const u16_srcs = [_]ast.SampleFormat{u16_fmt};

    // 1. Isolation: YUV444PS plane 0 (`range_min = 0.0`) vs plane 1 (`range_min = -0.5`) must NOT collide.
    const y_val = [_]builtins_mod.BuiltinVal{.{ .f32_bits = @bitCast(@as(f32, 0.0)), .i32_val = null }};
    const uv_val = [_]builtins_mod.BuiltinVal{.{ .f32_bits = @bitCast(@as(f32, -0.5)), .i32_val = null }};
    const k_y: Key = .{
        .expr = "x range_min +",
        .src_fmts = &f32_srcs,
        .dst_fmt = f32_fmt,
        .width = 16,
        .height = 2,
        .opt_int = false,
        .default_boundary = .clamp,
        .cpu = cpu,
        .profile = profile,
        .resolved_builtins = &y_val,
    };
    var k_uv = k_y;
    k_uv.resolved_builtins = &uv_val;
    try std.testing.expect(!keyEql(k_y, k_uv));

    // 2. D5/D1: same f32_bits (1.0) with i32_val = 1 vs i32_val = null must NOT collide.
    const int_one = [_]builtins_mod.BuiltinVal{.{ .f32_bits = @bitCast(@as(f32, 1.0)), .i32_val = 1 }};
    const flt_one = [_]builtins_mod.BuiltinVal{.{ .f32_bits = @bitCast(@as(f32, 1.0)), .i32_val = null }};
    var k_i1 = k_y;
    k_i1.resolved_builtins = &int_one;
    var k_f1 = k_y;
    k_f1.resolved_builtins = &flt_one;
    try std.testing.expect(!keyEql(k_i1, k_f1));
    try std.testing.expect(hashKey(k_i1) != hashKey(k_f1));

    // 3. Non-builtin expression key keeps resolved_builtins.len == 0 (zero allocation).
    const k_plain: Key = .{
        .expr = "x 1 +",
        .src_fmts = &f32_srcs,
        .dst_fmt = f32_fmt,
        .width = 16,
        .height = 2,
        .opt_int = false,
        .default_boundary = .clamp,
        .cpu = cpu,
        .profile = profile,
    };
    try std.testing.expectEqual(@as(usize, 0), k_plain.resolved_builtins.len);

    // 4. Phase A LUT sharing (C11/D3): costly 1D u16 gamma `x range_max / 2.2 pow range_max *`
    // builds a dimension-free LUT on Y (32x16) and hits on U/V (16x8) with identical `[65535]`.
    const y16_meta: builtins_mod.ClipMeta = .{ .is_float = false, .bits_per_sample = 16, .color_family = .yuv, .plane_idx = 0, .is_full_range = false };
    const rctx: builtins_mod.ResolveContext = .{ .src_metas = &.{y16_meta}, .dst_meta = y16_meta, .opt_int = false };
    var prog = try parser.parse(alloc, "x range_max / 2.2 pow range_max *", .{
        .mode = .expr,
        .num_clips = 1,
        .builtin_ctx = &rctx,
    });
    defer prog.deinit(alloc);
    try lower_mod.lower(alloc, &prog, profile);
    try optimize_mod.optimize(alloc, &prog, profile, .{ .enable_value_changing = true, .enable_fma = profile.use_fma });
    var info = try analysis_mod.analyze(alloc, &prog, &u16_srcs, u16_fmt, 32, 16, .{ .f32_lanes = cpu.f32_lanes });
    defer info.deinit(alloc);
    try std.testing.expectEqual(analysis_mod.LutKind.lut1d_u16, info.lut_kind);

    const lut = try lut_mod.Lut.build(alloc, rt, cpu, profile, &prog, &info, &u16_srcs, u16_fmt, 32, 16, null);
    try std.testing.expect(lut.simd_kernel == null);

    const u16_rmax = [_]builtins_mod.BuiltinVal{try builtins_mod.resolveMeta(.range_max, y16_meta)};
    const k_lut_dimfree: Key = .{
        .expr = "x range_max / 2.2 pow range_max *",
        .src_fmts = &u16_srcs,
        .dst_fmt = u16_fmt,
        .width = 0,
        .height = 0,
        .opt_int = false,
        .default_boundary = .clamp,
        .cpu = cpu,
        .profile = profile,
        .resolved_builtins = &u16_rmax,
    };
    const e_y = blk: {
        var exec: CompiledPlane = .{ .lut = lut };
        errdefer exec.deinit(alloc, rt);
        break :blk try cache.putOrGet(alloc, k_lut_dimfree, exec);
    };
    defer cache.release(alloc, e_y);

    // U/V plane lookup with identical `[65535]` resolved_builtins hits the exact same entry!
    const e_uv = cache.get(k_lut_dimfree) orelse return error.TestUnexpectedResult;
    defer cache.release(alloc, e_uv);
    try std.testing.expectEqual(e_y, e_uv);

    // Confirm JIT isolation on Y vs U/V in f32 too.
    const y_meta: builtins_mod.ClipMeta = .{ .is_float = true, .bits_per_sample = 32, .color_family = .yuv, .plane_idx = 0, .is_full_range = true };
    const uv_meta: builtins_mod.ClipMeta = .{ .is_float = true, .bits_per_sample = 32, .color_family = .yuv, .plane_idx = 1, .is_full_range = true };
    const compileF32 = struct {
        fn run(a: std.mem.Allocator, c: *PlaneCache(8), r: *aj.Runtime, k: Key, m: builtins_mod.ClipMeta) !*Entry {
            const ctx: builtins_mod.ResolveContext = .{ .src_metas = &.{m}, .dst_meta = m, .opt_int = false };
            var pr = try parser.parse(a, k.expr, .{ .mode = .expr, .num_clips = 1, .builtin_ctx = &ctx });
            defer pr.deinit(a);
            try lower_mod.lower(a, &pr, k.profile);
            try optimize_mod.optimize(a, &pr, k.profile, .{ .enable_value_changing = true, .enable_fma = k.profile.use_fma });
            var inf = try analysis_mod.analyze(a, &pr, k.src_fmts, k.dst_fmt, @intCast(k.width), @intCast(k.height), .{ .f32_lanes = k.cpu.f32_lanes });
            defer inf.deinit(a);
            const props = try buildPropBindings(a, &pr);
            errdefer freePropBindings(a, props);
            const compiled = try codegen.compilePlane(a, r, k.cpu, &pr, &inf, .{
                .width = k.width,
                .height = k.height,
                .src_fmts = k.src_fmts,
                .dst_fmt = k.dst_fmt,
            });
            var exec: CompiledPlane = .{ .jit = .{ .kernel = compiled.kernel, .props = props } };
            errdefer exec.deinit(a, r);
            return try c.putOrGet(a, k, exec);
        }
    }.run;
    const ej_y = try compileF32(alloc, &cache, rt, k_y, y_meta);
    defer cache.release(alloc, ej_y);
    const ej_uv = try compileF32(alloc, &cache, rt, k_uv, uv_meta);
    defer cache.release(alloc, ej_uv);
    try std.testing.expect(ej_y != ej_uv);
}

test "PlaneCache Phase B limited YUV16 LUT sharing (D3) and partial-variant refcount cleanup (F2)" {
    const parser = @import("../expr/parser.zig");
    const lower_mod = @import("../expr/lower.zig");
    const optimize_mod = @import("../expr/optimize.zig");
    const analysis_mod = @import("../expr/analysis.zig");

    const alloc = std.testing.allocator;
    const cpu = cpu_mod.detectWithOverride(null);
    if (cpu.isa == .scalar) return;
    const profile = interp.TargetProfile.fromCpu(cpu);

    var cache: PlaneCache(8) = .{};
    defer cache.deinit(alloc);
    const rt = cache.ensureRuntime() orelse return error.OutOfMemory;

    const u16_fmt: ast.SampleFormat = .{ .is_float = false, .bits_per_sample = 16 };
    const u16_srcs = [_]ast.SampleFormat{u16_fmt};
    const expr = "x plane_min - plane_max plane_min - / 2.2 pow range_max *";

    // D3: 16-bit limited Y resolves to [4096, 60160, 65535]; U and V both resolve to [4096, 61440, 65535].
    const y_meta: builtins_mod.ClipMeta = .{ .is_float = false, .bits_per_sample = 16, .color_family = .yuv, .plane_idx = 0, .is_full_range = false };
    const u_meta: builtins_mod.ClipMeta = .{ .is_float = false, .bits_per_sample = 16, .color_family = .yuv, .plane_idx = 1, .is_full_range = false };
    const v_meta: builtins_mod.ClipMeta = .{ .is_float = false, .bits_per_sample = 16, .color_family = .yuv, .plane_idx = 2, .is_full_range = false };

    const specs = [_]builtins_mod.BuiltinSpec{
        .{ .target = .{ .clip = 0 }, .kind = .plane_min },
        .{ .target = .{ .clip = 0 }, .kind = .plane_max },
        .{ .target = .{ .clip = 0 }, .kind = .range_max },
    };
    var rb_y: [3]builtins_mod.BuiltinVal = undefined;
    var rb_u: [3]builtins_mod.BuiltinVal = undefined;
    var rb_v: [3]builtins_mod.BuiltinVal = undefined;
    try builtins_mod.resolveAll(&specs, &.{y_meta}, y_meta, &rb_y);
    try builtins_mod.resolveAll(&specs, &.{u_meta}, u_meta, &rb_u);
    try builtins_mod.resolveAll(&specs, &.{v_meta}, v_meta, &rb_v);
    try std.testing.expectEqual(@as(?i32, 4096), rb_y[0].i32_val);
    try std.testing.expectEqual(@as(?i32, 60160), rb_y[1].i32_val);
    try std.testing.expectEqual(@as(?i32, 65535), rb_y[2].i32_val);
    try std.testing.expectEqual(@as(?i32, 4096), rb_u[0].i32_val);
    try std.testing.expectEqual(@as(?i32, 61440), rb_u[1].i32_val);
    try std.testing.expectEqual(@as(?i32, 65535), rb_u[2].i32_val);

    const buildLutEntry = struct {
        fn run(a: std.mem.Allocator, c: *PlaneCache(8), r: *aj.Runtime, c_info: cpu_mod.CpuInfo, prof: interp.TargetProfile, e: []const u8, m: builtins_mod.ClipMeta, rb: []const builtins_mod.BuiltinVal) !*Entry {
            const k: Key = .{
                .expr = e,
                .src_fmts = &.{.{ .is_float = false, .bits_per_sample = 16 }},
                .dst_fmt = .{ .is_float = false, .bits_per_sample = 16 },
                .width = 0,
                .height = 0,
                .opt_int = false,
                .default_boundary = .clamp,
                .cpu = c_info,
                .profile = prof,
                .resolved_builtins = rb,
            };
            if (c.get(k)) |hit| return hit;
            const ctx: builtins_mod.ResolveContext = .{ .src_metas = &.{m}, .dst_meta = m, .opt_int = false };
            var pr = try parser.parse(a, e, .{ .mode = .expr, .num_clips = 1, .builtin_ctx = &ctx });
            defer pr.deinit(a);
            try lower_mod.lower(a, &pr, prof);
            try optimize_mod.optimize(a, &pr, prof, .{ .enable_value_changing = true, .enable_fma = prof.use_fma });
            var inf = try analysis_mod.analyze(a, &pr, &.{.{ .is_float = false, .bits_per_sample = 16 }}, .{ .is_float = false, .bits_per_sample = 16 }, 32, 16, .{ .f32_lanes = c_info.f32_lanes });
            defer inf.deinit(a);
            const lut = try lut_mod.Lut.build(a, r, c_info, prof, &pr, &inf, &.{.{ .is_float = false, .bits_per_sample = 16 }}, .{ .is_float = false, .bits_per_sample = 16 }, 32, 16, null);
            var exec: CompiledPlane = .{ .lut = lut };
            errdefer exec.deinit(a, r);
            return try c.putOrGet(a, k, exec);
        }
    }.run;

    const e_y = try buildLutEntry(alloc, &cache, rt, cpu, profile, expr, y_meta, &rb_y);
    defer cache.release(alloc, e_y);
    const e_u = try buildLutEntry(alloc, &cache, rt, cpu, profile, expr, u_meta, &rb_u);
    defer cache.release(alloc, e_u);
    const e_v = try buildLutEntry(alloc, &cache, rt, cpu, profile, expr, v_meta, &rb_v);
    defer cache.release(alloc, e_v);

    try std.testing.expect(e_y != e_u);
    try std.testing.expectEqual(e_u, e_v);

    // F2: Exercise production `RangeVariants.deinit` directly with duplicate acquired cache refs
    // plus a heap `ScalarExec` and partial initialization (`init_count = 3` failing at slot 3).
    _ = u16_srcs;
    const TestRangeVariants = RangeVariants(PlaneCache(8), u8);
    const Partial = struct {
        fn build(a: std.mem.Allocator, c: *PlaneCache(8), shared: *Entry) !*TestRangeVariants {
            const rv = try a.create(TestRangeVariants);
            rv.* = .{ .clip_count = 2 };
            errdefer {
                rv.deinit(a, c);
                a.destroy(rv);
            }
            // Acquire two references to the exact same cached entry in slots 0 and 1.
            rv.variants[0] = .{ .cached = c.get(shared.key).? };
            rv.init_count += 1;
            rv.variants[1] = .{ .cached = c.get(shared.key).? };
            rv.init_count += 1;
            // Also populate slot 2 with a heap ScalarExec owning a parsed program + frame-prop bindings
            // so `ScalarExec.deinit` is verified leak-free under `std.testing.allocator`.
            {
                var sp = try parser.parse(a, "x[-1,0] x.Offset +", .{ .mode = .expr, .num_clips = 1 });
                var owns_sp = true;
                errdefer if (owns_sp) sp.deinit(a);
                const sprops = try buildPropBindings(a, &sp);
                var owns_sprops = true;
                errdefer if (owns_sprops) freePropBindings(a, sprops);
                const se = try a.create(ScalarExec);
                se.* = .{ .prog = sp, .props = sprops };
                owns_sp = false;
                owns_sprops = false;
                rv.variants[2] = .{ .scalar = se };
                rv.init_count += 1;
            }
            // Simulate failure while compiling slot 3 (`init_count == 3 < 4`).
            return error.OutOfMemory;
        }
    };
    const before_refs = e_u.ref_count;
    try std.testing.expectError(error.OutOfMemory, Partial.build(alloc, &cache, e_u));
    try std.testing.expectEqual(before_refs, e_u.ref_count);

    // F3: CompiledPlane.scalar caches lowered/optimized programs safely and frees them on deinit.
    {
        var sp = try parser.parse(alloc, "x[-1,0] 1 +", .{ .mode = .expr, .num_clips = 1 });
        errdefer sp.deinit(alloc);
        try lower_mod.lower(alloc, &sp, profile);
        try optimize_mod.optimize(alloc, &sp, profile, .{ .enable_value_changing = true, .enable_fma = profile.use_fma });
        const sprops = try buildPropBindings(alloc, &sp);
        errdefer freePropBindings(alloc, sprops);
        const f16_fmt: ast.SampleFormat = .{ .is_float = true, .bits_per_sample = 16 };
        const sk: Key = .{
            .expr = "x[-1,0] 1 +",
            .src_fmts = &.{f16_fmt},
            .dst_fmt = f16_fmt,
            .width = 16,
            .height = 4,
            .opt_int = false,
            .default_boundary = .clamp,
            .cpu = cpu,
            .profile = profile,
        };
        const se = blk: {
            var exec: CompiledPlane = .{ .scalar = .{ .prog = sp, .props = sprops } };
            errdefer exec.deinit(alloc, rt);
            break :blk try cache.putOrGet(alloc, sk, exec);
        };
        defer cache.release(alloc, se);
        const se_hit = cache.get(sk) orelse return error.TestUnexpectedResult;
        defer cache.release(alloc, se_hit);
        try std.testing.expectEqual(se, se_hit);
        try std.testing.expect(se_hit.exec == .scalar);
    }
}

test "PlaneCacheWithLutCap enforces LUT byte cap without evicting JIT entries and supports getOrDim" {
    const parser = @import("../expr/parser.zig");
    const lower_mod = @import("../expr/lower.zig");
    const optimize_mod = @import("../expr/optimize.zig");
    const analysis_mod = @import("../expr/analysis.zig");
    const codegen = @import("codegen.zig");

    const alloc = std.testing.allocator;
    const cpu = cpu_mod.detectWithOverride(null);
    if (cpu.isa == .scalar) return;
    const profile = interp.TargetProfile.fromCpu(cpu);

    // 8 total slots, but only 150 KiB LUT cap -> holds at most ONE 128 KiB u16 LUT, never two.
    var cache: PlaneCacheWithLutCap(8, 150 * 1024) = .{};
    defer cache.deinit(alloc);
    const rt = cache.ensureRuntime() orelse return error.OutOfMemory;

    const u8_fmt: ast.SampleFormat = .{ .is_float = false, .bits_per_sample = 8 };
    const u16_fmt: ast.SampleFormat = .{ .is_float = false, .bits_per_sample = 16 };

    // 1. Insert a JIT entry (0 LUT bytes).
    const jit_key: Key = .{
        .expr = "x 1 +",
        .src_fmts = &.{u8_fmt},
        .dst_fmt = u8_fmt,
        .width = 32,
        .height = 4,
        .opt_int = false,
        .default_boundary = .clamp,
        .cpu = cpu,
        .profile = profile,
    };
    const e_jit = blk: {
        var pr = try parser.parse(alloc, jit_key.expr, .{ .mode = .expr, .num_clips = 1 });
        defer pr.deinit(alloc);
        try lower_mod.lower(alloc, &pr, profile);
        try optimize_mod.optimize(alloc, &pr, profile, .{ .enable_value_changing = true, .enable_fma = profile.use_fma });
        var inf = try analysis_mod.analyze(alloc, &pr, jit_key.src_fmts, jit_key.dst_fmt, 32, 4, .{ .f32_lanes = cpu.f32_lanes });
        defer inf.deinit(alloc);
        const props = try buildPropBindings(alloc, &pr);
        errdefer freePropBindings(alloc, props);
        const compiled = try codegen.compilePlane(alloc, rt, cpu, &pr, &inf, .{ .width = 32, .height = 4, .src_fmts = jit_key.src_fmts, .dst_fmt = jit_key.dst_fmt });
        var exec: CompiledPlane = .{ .jit = .{ .kernel = compiled.kernel, .props = props } };
        errdefer exec.deinit(alloc, rt);
        break :blk try cache.putOrGet(alloc, jit_key, exec);
    };
    cache.release(alloc, e_jit);

    // 2. Insert two distinct 128 KiB dimension-free u16 LUTs: second must evict the first LUT (not e_jit!).
    const insertLut = struct {
        fn run(a: std.mem.Allocator, c: *PlaneCacheWithLutCap(8, 150 * 1024), r: *aj.Runtime, c_info: cpu_mod.CpuInfo, prof: interp.TargetProfile, expr_s: []const u8) !*Entry {
            const k: Key = .{
                .expr = expr_s,
                .src_fmts = &.{u16_fmt},
                .dst_fmt = u16_fmt,
                .width = 0,
                .height = 0,
                .opt_int = false,
                .default_boundary = .clamp,
                .cpu = c_info,
                .profile = prof,
            };
            var pr = try parser.parse(a, expr_s, .{ .mode = .expr, .num_clips = 1 });
            defer pr.deinit(a);
            try lower_mod.lower(a, &pr, prof);
            try optimize_mod.optimize(a, &pr, prof, .{ .enable_value_changing = true, .enable_fma = prof.use_fma });
            var inf = try analysis_mod.analyze(a, &pr, k.src_fmts, k.dst_fmt, 64, 16, .{ .f32_lanes = c_info.f32_lanes });
            defer inf.deinit(a);
            const lut = try lut_mod.Lut.build(a, r, c_info, prof, &pr, &inf, k.src_fmts, k.dst_fmt, 64, 16, null);
            var exec: CompiledPlane = .{ .lut = lut };
            errdefer exec.deinit(a, r);
            return try c.putOrGet(a, k, exec);
        }
    }.run;

    const e_lut1 = try insertLut(alloc, &cache, rt, cpu, profile, "x 65535 / 2.2 pow 65535 *");
    cache.release(alloc, e_lut1);
    try std.testing.expectEqual(@as(usize, 128 * 1024), cache.lut_bytes);

    const e_lut2 = try insertLut(alloc, &cache, rt, cpu, profile, "x 65535 / 1.8 pow 65535 *");
    defer cache.release(alloc, e_lut2);
    try std.testing.expectEqual(@as(usize, 128 * 1024), cache.lut_bytes);
    try std.testing.expectEqual(@as(usize, 2), cache.count);

    // JIT entry must still be cached (LUT cap did not touch it).
    const jit_hit = cache.get(jit_key) orelse return error.TestUnexpectedResult;
    defer cache.release(alloc, jit_hit);
    try std.testing.expectEqual(e_jit, jit_hit);

    // getOrDim must hit the dimension-free LUT entry in a single call.
    const exact_lut2: Key = .{
        .expr = "x 65535 / 1.8 pow 65535 *",
        .src_fmts = &.{u16_fmt},
        .dst_fmt = u16_fmt,
        .width = 64,
        .height = 16,
        .opt_int = false,
        .default_boundary = .clamp,
        .cpu = cpu,
        .profile = profile,
    };
    var dim_lut2 = exact_lut2;
    dim_lut2.width = 0;
    dim_lut2.height = 0;
    const lut_hit = cache.getOrDim(exact_lut2, dim_lut2) orelse return error.TestUnexpectedResult;
    defer cache.release(alloc, lut_hit);
    try std.testing.expectEqual(e_lut2, lut_hit);
}
