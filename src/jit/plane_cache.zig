//! Bounded thread-safe refcounted LRU cache for compiled `zexpr.Expr` planes.
//!
//! Caches dimension-specialized JIT kernels (with frame property bindings) and
//! precomputed 1D/2D LUTs across `Expr` instances (e.g. `FrameEval` callbacks
//! and identical U/V planes), sharing a single lazily-initialized `aj.Runtime`
//! so executable pages are reused without per-frame `mmap`/`munmap` syscalls.

const std = @import("std");
const ast = @import("../expr/ast.zig");
const interp = @import("../expr/interp.zig");
const aj = @import("asmjit.zig");
const cpu_mod = @import("cpu.zig");
const lut_mod = @import("lut.zig");

pub const default_capacity: usize = 32;

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
        out[i] = .{
            .clip = p.clip,
            .name_z = try allocator.dupeSentinel(u8, p.name, 0),
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
};

pub const CompiledPlane = union(enum) {
    lut: lut_mod.Lut,
    jit: struct {
        kernel: aj.CompiledKernel,
        props: []PropBinding,
    },

    pub fn deinit(self: *CompiledPlane, allocator: std.mem.Allocator, rt: ?*aj.Runtime) void {
        switch (self.*) {
            .lut => |*l| l.deinit(allocator, rt),
            .jit => |*j| {
                if (rt) |runtime| runtime.release(j.kernel.func);
                freePropBindings(allocator, j.props);
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

pub fn PlaneCache(comptime capacity: usize) type {
    comptime std.debug.assert(capacity > 0);
    return struct {
        const Self = @This();

        mutex: std.Io.Mutex = .init,
        rt: ?aj.Runtime = null,
        slots: [capacity]?*Entry = @splat(null),
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

            for (self.slots) |maybe_entry| {
                const e = maybe_entry orelse continue;
                if (e.hash == h and keyEql(e.key, key)) {
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

        /// Insert a freshly compiled plane into the LRU cache (taking ownership of
        /// `exec` on success) and return a refcounted handle for the caller.
        /// If another thread inserted the same key concurrently, destroys the
        /// duplicate `exec` and returns a handle to the existing entry.
        /// On OOM while allocating the cache entry, leaves `exec` untouched for the caller.
        pub fn putOrGet(self: *Self, allocator: std.mem.Allocator, key: Key, exec: CompiledPlane) std.mem.Allocator.Error!*Entry {
            const h = hashKey(key);

            const owned_expr = try allocator.dupe(u8, key.expr);
            errdefer allocator.free(owned_expr);

            const owned_src_fmts = try allocator.dupe(ast.SampleFormat, key.src_fmts);
            errdefer allocator.free(owned_src_fmts);

            const new_entry = try allocator.create(Entry);
            errdefer allocator.destroy(new_entry);

            var owned_key = key;
            owned_key.expr = owned_expr;
            owned_key.src_fmts = owned_src_fmts;

            var duplicate_exec: ?CompiledPlane = null;
            var evicted_dead: ?*Entry = null;
            var rt_ptr: ?*aj.Runtime = null;

            const result: *Entry = blk: {
                self.mutex.lockUncancelable(cache_io);
                defer self.mutex.unlock(cache_io);
                rt_ptr = self.runtimePtrLocked();

                for (self.slots) |maybe_entry| {
                    const e = maybe_entry orelse continue;
                    if (e.hash == h and keyEql(e.key, key)) {
                        self.stamp +%= 1;
                        e.last_used = self.stamp;
                        e.ref_count += 1;
                        duplicate_exec = exec;
                        break :blk e;
                    }
                }

                // Choose a free slot or the least-recently-used slot.
                var target_slot: usize = 0;
                var found_free = false;
                for (self.slots, 0..) |maybe_entry, i| {
                    if (maybe_entry == null) {
                        target_slot = i;
                        found_free = true;
                        break;
                    }
                }
                if (!found_free) {
                    var min_used = self.slots[0].?.last_used;
                    target_slot = 0;
                    for (self.slots[1..], 1..) |maybe_entry, i| {
                        const e = maybe_entry.?;
                        if (e.last_used < min_used) {
                            min_used = e.last_used;
                            target_slot = i;
                        }
                    }
                    const evicted = self.slots[target_slot].?;
                    evicted.ref_count -= 1;
                    if (evicted.ref_count == 0) {
                        evicted_dead = evicted;
                    }
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
                break :blk new_entry;
            };

            if (duplicate_exec) |*dup| {
                dup.deinit(allocator, rt_ptr);
                allocator.free(owned_expr);
                allocator.free(owned_src_fmts);
                allocator.destroy(new_entry);
            }
            if (evicted_dead) |dead| {
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
            for (&self.slots) |*slot| {
                if (slot.*) |e| {
                    e.ref_count -= 1;
                    std.debug.assert(e.ref_count == 0);
                    destroyEntry(allocator, e, rt_ptr);
                    slot.* = null;
                }
            }
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
            var info = try analysis_mod.analyze(a, &prog, k.src_fmts, k.dst_fmt, @intCast(k.width), @intCast(k.height), false);
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
