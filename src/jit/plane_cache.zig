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
/// 16 MiB cache-retained LUT payload budget (independent of evicted-but-pinned filter entries
/// or AsmJit executable mappings; allows up to 512 small JIT entries alongside large LUTs).
pub const default_max_lut_bytes: usize = 16 * 1024 * 1024;

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
    key: Key,
    exec: CompiledPlane,
    ref_count: usize,
    /// Intrusive singly-linked list used only after eviction to destroy dead entries outside the lock.
    next_dead: ?*Entry = null,
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

pub fn storageKey(key: Key, exec: *const CompiledPlane) Key {
    var out = key;
    switch (exec.*) {
        .lut => {
            out.width = 0;
            out.height = 0;
        },
        .jit, .scalar => {},
    }
    return out;
}

fn entryLutBytes(exec: *const CompiledPlane) usize {
    return switch (exec.*) {
        .lut => |*l| l.table.len,
        .jit, .scalar => 0,
    };
}

pub const CacheState = enum(u8) {
    open,
    closed,
};

pub const PutError = std.mem.Allocator.Error || error{CacheClosed};

pub const RuntimeLease = struct {
    rt: *aj.Runtime,
};

pub const RuntimeAcquireResult = union(enum) {
    lease: RuntimeLease,
    /// `aj.Runtime.init()` failed (e.g. ExecMemDenied or OOM); scalar fallback is permitted.
    unavailable,
    /// Cache has been closed by DSO teardown; caller must abort without admitting new entries.
    closed,
};

pub const HashPair = struct {
    exact: u64,
    dim_free: u64,
};

fn defaultHashPair(k: Key) HashPair {
    const base = hashBase(k);
    const h_exact = mixDims(base, k.width, k.height);
    const h_dim = if (k.width == 0 and k.height == 0) h_exact else mixDims(base, 0, 0);
    return .{ .exact = h_exact, .dim_free = h_dim };
}

pub fn PlaneCacheWithLutCap(comptime capacity: usize, comptime max_lut_bytes: usize) type {
    return PlaneCacheCustom(capacity, max_lut_bytes, defaultHashPair);
}

pub fn PlaneCacheCustom(
    comptime capacity: usize,
    comptime max_lut_bytes: usize,
    comptime hash_pair_fn: fn (Key) HashPair,
) type {
    comptime std.debug.assert(capacity > 0 and capacity <= std.math.maxInt(u16) - 1);
    const none_idx: u16 = std.math.maxInt(u16);
    const table_size: usize = std.math.ceilPowerOfTwoAssert(usize, capacity * 2);
    const table_mask: usize = table_size - 1;

    const SlotMeta = struct {
        entry: ?*Entry = null,
        hash: u64 = 0,
        prev_lru: u16 = none_idx,
        next_lru: u16 = none_idx,
        prev_lut: u16 = none_idx,
        next_lut: u16 = none_idx,
        next_free: u16 = none_idx,
        lut_bytes: usize = 0,
    };

    return struct {
        const Self = @This();

        mutex: std.Io.Mutex = .init,
        state: CacheState = .open,
        rt: ?aj.Runtime = null,
        live_external_refs: usize = 0,
        active_rt_users: usize = 0,

        /// Open-addressed hash index (`0` = empty, `slot + 1` = occupied) with backward-shift
        /// deletion (zero tombstones, load factor <= 0.50).
        index: [table_size]u16 = @splat(0),
        slots: [capacity]SlotMeta = @splat(.{}),
        next_fresh_slot: u16 = 0,
        free_head: u16 = none_idx,
        lru_head: u16 = none_idx,
        lru_tail: u16 = none_idx,
        lut_head: u16 = none_idx,
        lut_tail: u16 = none_idx,
        count: usize = 0,
        lut_bytes: usize = 0,
        hits: u64 = 0,

        inline fn lock(self: *Self) void {
            std.Io.Threaded.mutexLockUncancelable(&self.mutex);
        }

        inline fn unlock(self: *Self) void {
            std.Io.Threaded.mutexUnlock(&self.mutex);
        }

        fn allocSlotLocked(self: *Self) u16 {
            if (self.free_head != none_idx) {
                const s = self.free_head;
                self.free_head = self.slots[s].next_free;
                self.slots[s].next_free = none_idx;
                return s;
            }
            std.debug.assert(self.next_fresh_slot < capacity);
            const s = self.next_fresh_slot;
            self.next_fresh_slot += 1;
            return s;
        }

        fn freeSlotLocked(self: *Self, s: u16) void {
            self.slots[s] = .{ .next_free = self.free_head };
            self.free_head = s;
        }

        /// Intrusive doubly-linked list ops shared by the global LRU (`list = "lru"`) and the
        /// LUT-only LRU (`list = "lut"`): `prev_<list>`/`next_<list>` links, `<list>_head`/`<list>_tail`.
        fn pushFrontLocked(self: *Self, comptime list: []const u8, s: u16) void {
            const head = &@field(self, list ++ "_head");
            @field(self.slots[s], "prev_" ++ list) = none_idx;
            @field(self.slots[s], "next_" ++ list) = head.*;
            if (head.* != none_idx) {
                @field(self.slots[head.*], "prev_" ++ list) = s;
            } else {
                @field(self, list ++ "_tail") = s;
            }
            head.* = s;
        }

        fn unlinkLocked(self: *Self, comptime list: []const u8, s: u16) void {
            const p = @field(self.slots[s], "prev_" ++ list);
            const n = @field(self.slots[s], "next_" ++ list);
            if (p != none_idx) {
                @field(self.slots[p], "next_" ++ list) = n;
            } else {
                @field(self, list ++ "_head") = n;
            }
            if (n != none_idx) {
                @field(self.slots[n], "prev_" ++ list) = p;
            } else {
                @field(self, list ++ "_tail") = p;
            }
            @field(self.slots[s], "prev_" ++ list) = none_idx;
            @field(self.slots[s], "next_" ++ list) = none_idx;
        }

        fn moveToFrontLocked(self: *Self, comptime list: []const u8, s: u16) void {
            if (@field(self, list ++ "_head") == s) return;
            self.unlinkLocked(list, s);
            self.pushFrontLocked(list, s);
        }

        fn touchLocked(self: *Self, s: u16) void {
            self.moveToFrontLocked("lru", s);
            if (self.slots[s].lut_bytes > 0) {
                self.moveToFrontLocked("lut", s);
            }
        }

        /// Cache hit: move slot `s` to MRU and hand the caller a new reference to its entry.
        fn takeHitLocked(self: *Self, s: u16) *Entry {
            self.touchLocked(s);
            const e = self.slots[s].entry.?;
            e.ref_count += 1;
            self.live_external_refs += 1;
            return e;
        }

        fn findSlotLocked(self: *const Self, h: u64, key: Key) ?u16 {
            var pos: usize = @truncate(h & table_mask);
            while (true) {
                const raw = self.index[pos];
                if (raw == 0) return null;
                const s = raw - 1;
                if (self.slots[s].hash == h and keyEql(self.slots[s].entry.?.key, key)) {
                    return s;
                }
                pos = (pos + 1) & table_mask;
            }
        }

        fn indexInsertLocked(self: *Self, s: u16, h: u64) void {
            var pos: usize = @truncate(h & table_mask);
            while (self.index[pos] != 0) {
                pos = (pos + 1) & table_mask;
            }
            self.index[pos] = s + 1;
        }

        fn indexRemoveLocked(self: *Self, s: u16, h: u64) void {
            const target: u16 = s + 1;
            var hole: usize = @truncate(h & table_mask);
            while (self.index[hole] != target) {
                std.debug.assert(self.index[hole] != 0);
                hole = (hole + 1) & table_mask;
            }
            var cur = (hole + 1) & table_mask;
            while (self.index[cur] != 0) {
                const cand_slot = self.index[cur] - 1;
                const ideal: usize = @truncate(self.slots[cand_slot].hash & table_mask);
                // Shift `cur` backward into `hole` iff `hole` lies cyclically in `[ideal, cur)`.
                if (((hole -% ideal) & table_mask) < ((cur -% ideal) & table_mask)) {
                    self.index[hole] = self.index[cur];
                    hole = cur;
                }
                cur = (cur + 1) & table_mask;
            }
            self.index[hole] = 0;
        }

        fn evictSlotLocked(self: *Self, s: u16, dead_head: *?*Entry) void {
            const meta = self.slots[s];
            const e = meta.entry.?;
            self.indexRemoveLocked(s, meta.hash);
            self.unlinkLocked("lru", s);
            if (meta.lut_bytes > 0) {
                self.unlinkLocked("lut", s);
                self.lut_bytes -= meta.lut_bytes;
            }
            self.count -= 1;
            self.freeSlotLocked(s);
            e.ref_count -= 1;
            if (e.ref_count == 0) {
                e.next_dead = dead_head.*;
                dead_head.* = e;
            }
        }

        fn resetStorageLocked(self: *Self) void {
            @memset(&self.index, 0);
            self.next_fresh_slot = 0;
            self.free_head = none_idx;
            self.lru_head = none_idx;
            self.lru_tail = none_idx;
            self.lut_head = none_idx;
            self.lut_tail = none_idx;
            self.count = 0;
            self.lut_bytes = 0;
        }

        /// Lazily initialize the shared `aj.Runtime` and acquire an in-flight lease that keeps
        /// the runtime alive across compilation and insertion until `releaseRuntime()` is called.
        pub fn acquireRuntime(self: *Self) RuntimeAcquireResult {
            self.lock();
            defer self.unlock();
            if (self.state == .closed) return .closed;
            if (self.rt == null) {
                self.rt = aj.Runtime.init() catch null;
            }
            if (self.rt == null) return .unavailable;
            self.active_rt_users += 1;
            return .{ .lease = .{ .rt = &self.rt.? } };
        }

        /// Release one in-flight runtime lease acquired via `acquireRuntime()`.
        pub fn releaseRuntime(self: *Self) void {
            self.lock();
            defer self.unlock();
            std.debug.assert(self.active_rt_users > 0);
            self.active_rt_users -= 1;
        }

        fn runtimePtrLocked(self: *Self) ?*aj.Runtime {
            return if (self.rt != null) &self.rt.? else null;
        }

        /// Look up a compiled plane by exact key. On hit, moves the entry to MRU,
        /// increments `ref_count`, and returns it (caller must release it with `release`).
        pub fn get(self: *Self, key: Key) ?*Entry {
            const h = hash_pair_fn(key).exact;
            self.lock();
            defer self.unlock();
            if (self.state == .closed) return null;

            if (self.findSlotLocked(h, key)) |s| {
                self.hits += 1;
                return self.takeHitLocked(s);
            }
            return null;
        }

        /// Single-lock lookup preferring an exact `(width, height)` match and falling back
        /// to a dimension-free (`0x0`) LUT entry, hashing the key prefix only once.
        pub fn getOrDim(self: *Self, key: Key) ?*Entry {
            const hp = hash_pair_fn(key);
            const has_dims = key.width != 0 or key.height != 0;
            self.lock();
            defer self.unlock();
            if (self.state == .closed) return null;

            if (self.findSlotLocked(hp.exact, key)) |s| {
                self.hits += 1;
                return self.takeHitLocked(s);
            }
            if (has_dims) {
                var dim_key = key;
                dim_key.width = 0;
                dim_key.height = 0;
                if (self.findSlotLocked(hp.dim_free, dim_key)) |s| {
                    self.hits += 1;
                    return self.takeHitLocked(s);
                }
            }
            return null;
        }

        /// Insert a freshly compiled plane into the LRU cache (taking ownership of
        /// `exec` on success) and return a refcounted handle for the caller.
        /// If another thread inserted the same key concurrently, destroys the
        /// duplicate `exec` and returns a handle to the existing entry.
        /// If a single LUT payload exceeds `max_lut_bytes`, bypasses cache admission
        /// without evicting healthy entries and returns a caller-only entry (`ref_count = 1`).
        /// On `OutOfMemory` or `CacheClosed`, leaves `exec` untouched in caller ownership.
        pub fn putOrGet(self: *Self, allocator: std.mem.Allocator, key: Key, exec: CompiledPlane) PutError!*Entry {
            const h = hash_pair_fn(key).exact;
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
            var dead_head: ?*Entry = null;
            var rt_ptr: ?*aj.Runtime = null;
            var held_rt_user = false;

            const result: *Entry = blk: {
                self.lock();
                defer self.unlock();
                if (self.state == .closed) return error.CacheClosed;
                rt_ptr = self.runtimePtrLocked();

                if (self.findSlotLocked(h, key)) |s| {
                    const e = self.takeHitLocked(s);
                    duplicate_exec = exec;
                    self.active_rt_users += 1;
                    held_rt_user = true;
                    break :blk e;
                }

                // Oversize-admission bypass: a single LUT larger than the retention budget is
                // returned with caller-only ownership instead of wiping healthy cached entries.
                if (new_lut_bytes > max_lut_bytes) {
                    new_entry.* = .{
                        .key = owned_key,
                        .exec = exec,
                        .ref_count = 1,
                        .next_dead = null,
                    };
                    self.live_external_refs += 1;
                    break :blk new_entry;
                }

                // Enforce the retained LUT byte cap in O(1) per evicted LUT without touching JIT entries.
                if (new_lut_bytes > 0) {
                    while (self.lut_bytes + new_lut_bytes > max_lut_bytes) {
                        std.debug.assert(self.lut_tail != none_idx);
                        self.evictSlotLocked(self.lut_tail, &dead_head);
                    }
                }

                // Enforce the total slot cap in O(1) via global LRU tail.
                if (self.count == capacity) {
                    std.debug.assert(self.lru_tail != none_idx);
                    self.evictSlotLocked(self.lru_tail, &dead_head);
                }

                const slot = self.allocSlotLocked();
                new_entry.* = .{
                    .key = owned_key,
                    .exec = exec,
                    .ref_count = 2, // 1 for cache slot + 1 for caller
                    .next_dead = null,
                };
                self.slots[slot] = .{
                    .entry = new_entry,
                    .hash = h,
                    .lut_bytes = new_lut_bytes,
                };
                self.indexInsertLocked(slot, h);
                self.pushFrontLocked("lru", slot);
                if (new_lut_bytes > 0) {
                    self.pushFrontLocked("lut", slot);
                    self.lut_bytes += new_lut_bytes;
                }
                self.count += 1;
                self.live_external_refs += 1;
                if (dead_head != null) {
                    self.active_rt_users += 1;
                    held_rt_user = true;
                }
                break :blk new_entry;
            };

            defer if (held_rt_user) self.releaseRuntime();
            if (duplicate_exec) |*dup| {
                dup.deinit(allocator, rt_ptr);
                allocator.free(owned_expr);
                allocator.free(owned_src_fmts);
                if (owned_builtins.len > 0) allocator.free(owned_builtins);
                allocator.destroy(new_entry);
            }
            var cur_dead = dead_head;
            while (cur_dead) |dead| {
                const next = dead.next_dead;
                destroyEntry(allocator, dead, rt_ptr);
                cur_dead = next;
            }
            return result;
        }

        /// Drop one caller reference to `entry`. Destroys the entry once evicted
        /// (or uncached due to oversize bypass) and no active filters hold a reference.
        pub fn release(self: *Self, allocator: std.mem.Allocator, entry: *Entry) void {
            self.releaseImpl(allocator, entry, null, null);
        }

        fn releaseImpl(
            self: *Self,
            allocator: std.mem.Allocator,
            entry: *Entry,
            after_unlock_hook: ?*const fn (?*anyopaque) void,
            hook_ctx: ?*anyopaque,
        ) void {
            var destroy = false;
            var rt_ptr: ?*aj.Runtime = null;
            {
                self.lock();
                defer self.unlock();
                std.debug.assert(entry.ref_count > 0);
                std.debug.assert(self.live_external_refs > 0);
                entry.ref_count -= 1;
                self.live_external_refs -= 1;
                if (entry.ref_count == 0) {
                    destroy = true;
                    rt_ptr = self.runtimePtrLocked();
                    // Keep `self.rt` alive until deferred `destroyEntry` finishes outside the lock.
                    self.active_rt_users += 1;
                }
            }
            if (destroy) {
                defer self.releaseRuntime();
                if (after_unlock_hook) |hook| hook(hook_ctx);
                destroyEntry(allocator, entry, rt_ptr);
            }
        }

        /// Non-blocking DSO teardown hook: closes the cache, frees cached entry heap payloads,
        /// and destroys `self.rt` in a single bulk `JitAllocator::reset(kHard)` pass when quiescent.
        /// Returns `true` iff teardown ran to completion; returns `false` (skipping teardown safely)
        /// if the mutex is contended, the cache is already closed, or any external ref or in-flight
        /// runtime user is active (dirty process exit).
        pub fn shutdown(self: *Self, allocator: std.mem.Allocator) bool {
            if (!self.mutex.tryLock()) return false;
            defer self.unlock();
            if (self.state == .closed) return false;
            if (self.live_external_refs != 0 or self.active_rt_users != 0) return false;
            self.teardownLocked(allocator);
            return true;
        }

        /// Release all cached entries and the shared `JitRuntime` (for tests;
        /// callers must have released all active entries and runtime leases first).
        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            self.lock();
            defer self.unlock();
            std.debug.assert(self.live_external_refs == 0);
            std.debug.assert(self.active_rt_users == 0);
            self.teardownLocked(allocator);
        }

        /// Close the cache, destroy every cached entry and the shared runtime (quiescent callers only).
        fn teardownLocked(self: *Self, allocator: std.mem.Allocator) void {
            self.state = .closed;
            var s = self.lru_head;
            while (s != none_idx) {
                const next = self.slots[s].next_lru;
                const e = self.slots[s].entry.?;
                e.ref_count -= 1;
                std.debug.assert(e.ref_count == 0);
                // Pass `rt = null` so per-kernel `Runtime.release` (and its internal lock) is
                // skipped; `self.rt.deinit()` reclaims all executable blocks in one bulk reset.
                destroyEntry(allocator, e, null);
                self.slots[s] = .{};
                s = next;
            }
            self.resetStorageLocked();
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

/// Hash all dimension-independent key fields once so `getOrDim` derives exact and `0x0`
/// hashes without re-hashing `expr`, `src_fmts`, or `resolved_builtins`.
fn hashBase(k: Key) u64 {
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
    const cpu_bits = cpuFlagBits(k.cpu);
    h.update(std.mem.asBytes(&cpu_bits));
    // D5: hash resolved_builtins field by field (never raw struct bytes over `?i32` padding).
    h.update(std.mem.asBytes(&k.resolved_builtins.len));
    for (k.resolved_builtins) |bv| {
        bv.hashInto(&h);
    }
    return h.final();
}

fn mixDims(base: u64, width: usize, height: usize) u64 {
    var h = std.hash.Wyhash.init(base);
    h.update(std.mem.asBytes(&width));
    h.update(std.mem.asBytes(&height));
    return h.final();
}

fn fmtEql(a: ast.SampleFormat, b: ast.SampleFormat) bool {
    return a.is_float == b.is_float and a.bits_per_sample == b.bits_per_sample;
}

/// All boolean `CpuInfo` feature flags packed into one byte (shared by `hashBase` and `cpuEql`).
fn cpuFlagBits(c: cpu_mod.CpuInfo) u8 {
    return (@as(u8, @intFromBool(c.has_ssse3)) << 0) |
        (@as(u8, @intFromBool(c.has_sse4_1)) << 1) |
        (@as(u8, @intFromBool(c.has_fma)) << 2) |
        (@as(u8, @intFromBool(c.has_f16c)) << 3) |
        (@as(u8, @intFromBool(c.has_avx2)) << 4) |
        (@as(u8, @intFromBool(c.has_avx512)) << 5) |
        (@as(u8, @intFromBool(c.prefer_hw_gather_f32)) << 6) |
        (@as(u8, @intFromBool(c.prefer_alt_widen_load)) << 7);
}

fn cpuEql(a: cpu_mod.CpuInfo, b: cpu_mod.CpuInfo) bool {
    return a.isa == b.isa and
        a.vec_bytes == b.vec_bytes and
        a.f32_lanes == b.f32_lanes and
        cpuFlagBits(a) == cpuFlagBits(b);
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

// Test-only helpers (lazily analyzed, so the production library never compiles them).
const parser = @import("../expr/parser.zig");
const lower_mod = @import("../expr/lower.zig");
const optimize_mod = @import("../expr/optimize.zig");
const analysis_mod = @import("../expr/analysis.zig");
const codegen = @import("codegen.zig");

fn testLease(res: RuntimeAcquireResult) !*aj.Runtime {
    return switch (res) {
        .lease => |l| l.rt,
        else => error.OutOfMemory,
    };
}

const TestPlan = struct {
    prog: ast.Program,
    info: analysis_mod.Analysis,

    fn deinit(self: *TestPlan, a: std.mem.Allocator) void {
        self.info.deinit(a);
        self.prog.deinit(a);
    }
};

/// Parse, lower, optimize and analyze `k.expr` for a `w x h` plane, resolving built-ins
/// against `meta` (as clip 0 and the output) when given.
fn testPlan(a: std.mem.Allocator, k: Key, meta: ?builtins_mod.ClipMeta, w: i32, h: i32, fast_u8_lut: bool) !TestPlan {
    var metas: [1]builtins_mod.ClipMeta = undefined;
    var ctx: builtins_mod.ResolveContext = undefined;
    const ctx_ptr: ?*const builtins_mod.ResolveContext = if (meta) |m| blk: {
        metas = .{m};
        ctx = .{ .src_metas = &metas, .dst_meta = m, .opt_int = false };
        break :blk &ctx;
    } else null;
    var prog = try parser.parse(a, k.expr, .{
        .mode = .expr,
        .num_clips = @intCast(k.src_fmts.len),
        .default_boundary = k.default_boundary,
        .opt_int = k.opt_int,
        .builtin_ctx = ctx_ptr,
    });
    errdefer prog.deinit(a);
    try lower_mod.lower(a, &prog, k.profile);
    try optimize_mod.optimize(a, &prog, k.profile, .{ .enable_value_changing = true, .enable_fma = k.profile.use_fma });
    const info = try analysis_mod.analyze(a, &prog, k.src_fmts, k.dst_fmt, w, h, .{
        .has_fast_u8_lut = fast_u8_lut,
        .f32_lanes = k.cpu.f32_lanes,
    });
    return .{ .prog = prog, .info = info };
}

/// JIT-compile `k` at its own dimensions into a caller-owned `CompiledPlane.jit`.
fn testJit(a: std.mem.Allocator, r: *aj.Runtime, k: Key, meta: ?builtins_mod.ClipMeta) !CompiledPlane {
    var plan = try testPlan(a, k, meta, @intCast(k.width), @intCast(k.height), false);
    defer plan.deinit(a);
    const props = try buildPropBindings(a, &plan.prog);
    errdefer freePropBindings(a, props);
    const compiled = try codegen.compilePlane(a, r, k.cpu, &plan.prog, &plan.info, .{
        .width = k.width,
        .height = k.height,
        .src_fmts = k.src_fmts,
        .dst_fmt = k.dst_fmt,
    });
    return .{ .jit = .{ .kernel = compiled.kernel, .props = props } };
}

/// Build the LUT for `k` (analyzed as a `w x h` plane) into a caller-owned `Lut`.
fn testLut(a: std.mem.Allocator, r: *aj.Runtime, k: Key, meta: ?builtins_mod.ClipMeta, w: i32, h: i32, fast_u8_lut: bool) !lut_mod.Lut {
    var plan = try testPlan(a, k, meta, w, h, fast_u8_lut);
    defer plan.deinit(a);
    return lut_mod.Lut.build(a, r, k.cpu, k.profile, &plan.prog, &plan.info, k.src_fmts, k.dst_fmt, null);
}

/// A scalar `lut1d_u8` entry with a `len`-byte table filled with `fill`.
fn testTableLut(a: std.mem.Allocator, len: usize, fill: u8) !CompiledPlane {
    const tbl = try a.alignedAlloc(u8, .@"64", len);
    @memset(tbl, fill);
    return .{ .lut = .{ .kind = .lut1d_u8, .clips = .{ 0, 0 }, .dst_fmt = .{ .is_float = false, .bits_per_sample = 8 }, .table = tbl } };
}

/// `putOrGet` that destroys `exec` when admission fails (the error leaves it with the caller).
fn testAdmit(cache: anytype, a: std.mem.Allocator, r: ?*aj.Runtime, k: Key, exec: CompiledPlane) !*Entry {
    var owned = exec;
    errdefer owned.deinit(a, r);
    return try cache.putOrGet(a, k, owned);
}

test "PlaneCache LRU hit, eviction, active ref survival, and key isolation" {
    const alloc = std.testing.allocator;
    const cpu = cpu_mod.detectWithOverride(null);
    if (cpu.isa == .scalar) return;
    const profile = interp.TargetProfile.fromCpu(cpu);

    var cache: PlaneCache(2) = .{};
    defer cache.deinit(alloc);
    const rt = try testLease(cache.acquireRuntime());
    defer cache.releaseRuntime();

    const u8_fmt: ast.SampleFormat = .{ .is_float = false, .bits_per_sample = 8 };
    const src_fmts = [_]ast.SampleFormat{u8_fmt};

    const compileOne = struct {
        fn run(a: std.mem.Allocator, c: *PlaneCache(2), r: *aj.Runtime, k: Key) !*Entry {
            if (c.get(k)) |hit| return hit;
            return testAdmit(c, a, r, k, try testJit(a, r, k, null));
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
    // VapourSynth aligns row strides to 64 bytes on AVX-512 (32 on AVX2), so allocate 64B rows.
    const row_stride = 64;
    var src_buf: [row_stride * 4]u8 align(64) = @splat(10);
    var dst_buf: [row_stride * 4]u8 align(64) = @splat(0);
    const srcs = [_][*]const u8{&src_buf};
    const strides = [_]isize{row_stride};
    e1.exec.jit.kernel.func(&dst_buf, row_stride, &srcs, &strides, 0, null);
    try std.testing.expectEqual(11, dst_buf[0]);
    try std.testing.expectEqual(11, dst_buf[3 * row_stride + 31]);

    cache.release(alloc, e1);
}

test "PlaneCache resolved_builtins isolation, D5 i32_val distinction, and Phase A LUT sharing" {
    const alloc = std.testing.allocator;
    const cpu = cpu_mod.detectWithOverride(null);
    if (cpu.isa == .scalar) return;
    const profile = interp.TargetProfile.fromCpu(cpu);

    var cache: PlaneCache(8) = .{};
    defer cache.deinit(alloc);
    const rt = try testLease(cache.acquireRuntime());
    defer cache.releaseRuntime();

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
    try std.testing.expect(defaultHashPair(k_i1).exact != defaultHashPair(k_f1).exact);

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
    {
        var plan = try testPlan(alloc, k_lut_dimfree, y16_meta, 32, 16, false);
        defer plan.deinit(alloc);
        try std.testing.expectEqual(analysis_mod.LutKind.lut1d_u16, plan.info.lut_kind);
    }
    const lut = try testLut(alloc, rt, k_lut_dimfree, y16_meta, 32, 16, false);
    try std.testing.expect(lut.simd_kernel == null);
    const e_y = try testAdmit(&cache, alloc, rt, k_lut_dimfree, .{ .lut = lut });
    defer cache.release(alloc, e_y);

    // U/V plane lookup with identical `[65535]` resolved_builtins hits the exact same entry!
    const e_uv = cache.get(k_lut_dimfree) orelse return error.TestUnexpectedResult;
    defer cache.release(alloc, e_uv);
    try std.testing.expectEqual(e_y, e_uv);

    // Confirm JIT isolation on Y vs U/V in f32 too.
    const y_meta: builtins_mod.ClipMeta = .{ .is_float = true, .bits_per_sample = 32, .color_family = .yuv, .plane_idx = 0, .is_full_range = true };
    const uv_meta: builtins_mod.ClipMeta = .{ .is_float = true, .bits_per_sample = 32, .color_family = .yuv, .plane_idx = 1, .is_full_range = true };
    const ej_y = try testAdmit(&cache, alloc, rt, k_y, try testJit(alloc, rt, k_y, y_meta));
    defer cache.release(alloc, ej_y);
    const ej_uv = try testAdmit(&cache, alloc, rt, k_uv, try testJit(alloc, rt, k_uv, uv_meta));
    defer cache.release(alloc, ej_uv);
    try std.testing.expect(ej_y != ej_uv);
}

test "PlaneCache Phase B limited YUV16 LUT sharing (D3) and partial-variant refcount cleanup (F2)" {
    const alloc = std.testing.allocator;
    const cpu = cpu_mod.detectWithOverride(null);
    if (cpu.isa == .scalar) return;
    const profile = interp.TargetProfile.fromCpu(cpu);

    var cache: PlaneCache(8) = .{};
    defer cache.deinit(alloc);
    const rt = try testLease(cache.acquireRuntime());
    defer cache.releaseRuntime();

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
            return testAdmit(c, a, r, k, .{ .lut = try testLut(a, r, k, m, 32, 16, false) });
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
    const alloc = std.testing.allocator;
    const cpu = cpu_mod.detectWithOverride(null);
    if (cpu.isa == .scalar) return;
    const profile = interp.TargetProfile.fromCpu(cpu);

    // 8 total slots, but only 150 KiB LUT cap -> holds at most ONE 128 KiB u16 LUT, never two.
    var cache: PlaneCacheWithLutCap(8, 150 * 1024) = .{};
    defer cache.deinit(alloc);
    const rt = try testLease(cache.acquireRuntime());
    defer cache.releaseRuntime();

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
    const e_jit = try testAdmit(&cache, alloc, rt, jit_key, try testJit(alloc, rt, jit_key, null));
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
            return testAdmit(c, a, r, k, .{ .lut = try testLut(a, r, k, null, 64, 16, false) });
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
    const lut_hit = cache.getOrDim(exact_lut2) orelse return error.TestUnexpectedResult;
    defer cache.release(alloc, lut_hit);
    try std.testing.expectEqual(e_lut2, lut_hit);
}

test "PlaneCache shutdown guards in-flight runtime lease, deferred release, busy lock, and late closed admission" {
    const alloc = std.testing.allocator;
    const cpu = cpu_mod.detectWithOverride(null);
    if (cpu.isa == .scalar) return;
    const profile = interp.TargetProfile.fromCpu(cpu);
    const u8_fmt: ast.SampleFormat = .{ .is_float = false, .bits_per_sample = 8 };
    const src_fmts = [_]ast.SampleFormat{u8_fmt};

    const base_key: Key = .{
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
    var k2 = base_key;
    k2.expr = "x 2 +";

    // 1. Cold compiler paused after `acquireRuntime()`: shutdown sees active_rt_users == 1
    //    and skips teardown; once lease is released, shutdown drains and destroys `rt`.
    {
        var cache: PlaneCache(2) = .{};
        defer cache.deinit(alloc);

        const lease_rt = try testLease(cache.acquireRuntime());
        try std.testing.expectEqual(@as(usize, 1), cache.active_rt_users);
        try std.testing.expect(!cache.shutdown(alloc));
        try std.testing.expectEqual(CacheState.open, cache.state);
        try std.testing.expect(cache.rt != null);

        // Compilation still succeeds safely against the borrowed runtime.
        const e1 = try testAdmit(&cache, alloc, lease_rt, base_key, try testJit(alloc, lease_rt, base_key, null));
        cache.releaseRuntime();

        // While caller holds `e1` (live_external_refs == 1), shutdown must still refuse.
        try std.testing.expectEqual(@as(usize, 1), cache.live_external_refs);
        try std.testing.expect(!cache.shutdown(alloc));
        try std.testing.expectEqual(CacheState.open, cache.state);

        // Release caller ref; now quiescent -> shutdown drains entry + destroys runtime.
        cache.release(alloc, e1);
        try std.testing.expectEqual(@as(usize, 0), cache.live_external_refs);
        try std.testing.expectEqual(@as(usize, 0), cache.active_rt_users);
        try std.testing.expect(cache.shutdown(alloc));
        try std.testing.expectEqual(CacheState.closed, cache.state);
        try std.testing.expectEqual(@as(usize, 0), cache.count);
        try std.testing.expect(cache.rt == null);
        // Second shutdown is an idempotent no-op returning false.
        try std.testing.expect(!cache.shutdown(alloc));
    }

    // 2. Final `release()` of an evicted entry paused after unlock and before `destroyEntry`:
    //    `shutdown()` invoked inside the hook must observe active_rt_users == 1 and skip teardown
    //    so `destroyEntry` safely releases the kernel against the still-alive runtime.
    {
        const Cache1 = PlaneCache(1);
        var cache: Cache1 = .{};
        defer cache.deinit(alloc);

        const lease_rt = try testLease(cache.acquireRuntime());
        const e1 = try testAdmit(&cache, alloc, lease_rt, base_key, try testJit(alloc, lease_rt, base_key, null));
        const e2 = try testAdmit(&cache, alloc, lease_rt, k2, try testJit(alloc, lease_rt, k2, null));
        cache.releaseRuntime();
        // Drop e2's caller ref (still cached in slot 0); e1 is evicted and pinned only by caller.
        cache.release(alloc, e2);
        try std.testing.expectEqual(@as(usize, 1), e1.ref_count);

        const HookCtx = struct {
            c: *Cache1,
            a: std.mem.Allocator,
            shutdown_ran: bool = false,
            seen_active_users: usize = 0,
            rt_alive: bool = false,
        };
        var hctx: HookCtx = .{ .c = &cache, .a = alloc };
        const hookFn = struct {
            fn call(raw: ?*anyopaque) void {
                const ctx: *HookCtx = @ptrCast(@alignCast(raw.?));
                ctx.seen_active_users = ctx.c.active_rt_users;
                ctx.shutdown_ran = ctx.c.shutdown(ctx.a);
                ctx.rt_alive = ctx.c.rt != null;
            }
        }.call;

        cache.releaseImpl(alloc, e1, hookFn, &hctx);
        try std.testing.expectEqual(@as(usize, 1), hctx.seen_active_users);
        try std.testing.expect(!hctx.shutdown_ran);
        try std.testing.expect(hctx.rt_alive);
        try std.testing.expectEqual(CacheState.open, cache.state);

        // Once releaseImpl finishes, active_rt_users drops back to 0 and shutdown succeeds.
        try std.testing.expectEqual(@as(usize, 0), cache.active_rt_users);
        try std.testing.expect(cache.shutdown(alloc));
        try std.testing.expectEqual(CacheState.closed, cache.state);
        try std.testing.expect(cache.rt == null);
    }

    // 3. Contended mutex during dirty exit: shutdown must not block and returns false.
    {
        var cache: PlaneCache(2) = .{};
        defer cache.deinit(alloc);
        cache.lock();
        try std.testing.expect(!cache.shutdown(alloc));
        cache.unlock();
        try std.testing.expectEqual(CacheState.open, cache.state);
        try std.testing.expect(cache.shutdown(alloc));
    }

    // 4. Late admission after `shutdown()`: `acquireRuntime` returns `.closed`, `get`/`getOrDim`
    //    return null, and late scalar `putOrGet` returns `error.CacheClosed`, leaving the cache
    //    empty and leaving payload ownership with the caller (verified leak-free by testing.allocator).
    {
        var cache: PlaneCache(4) = .{};
        defer cache.deinit(alloc);
        try std.testing.expect(cache.shutdown(alloc));

        try std.testing.expectEqual(RuntimeAcquireResult.closed, cache.acquireRuntime());
        try std.testing.expect(cache.get(base_key) == null);
        try std.testing.expect(cache.getOrDim(base_key) == null);

        var sp = try parser.parse(alloc, "x 1 +", .{ .mode = .expr, .num_clips = 1 });
        var owns_sp = true;
        errdefer if (owns_sp) sp.deinit(alloc);
        const sprops = try buildPropBindings(alloc, &sp);
        var owns_sprops = true;
        errdefer if (owns_sprops) freePropBindings(alloc, sprops);
        var scalar_exec: CompiledPlane = .{ .scalar = .{ .prog = sp, .props = sprops } };
        owns_sp = false;
        owns_sprops = false;
        defer scalar_exec.deinit(alloc, null);

        try std.testing.expectError(error.CacheClosed, cache.putOrGet(alloc, base_key, scalar_exec));
        try std.testing.expectEqual(@as(usize, 0), cache.count);
        try std.testing.expectEqual(@as(usize, 0), cache.live_external_refs);
    }
}

test "PlaneCacheCustom matches reference model under wrap-around clusters, exact/0x0 hash collisions, and multi-LUT eviction" {
    const alloc = std.testing.allocator;
    const cpu = cpu_mod.detectWithOverride(null);
    const profile = interp.TargetProfile.fromCpu(cpu);
    const u8_fmt: ast.SampleFormat = .{ .is_float = false, .bits_per_sample = 8 };
    const src_fmts = [_]ast.SampleFormat{u8_fmt};

    // Deliberately hash everything into bucket 6 or 7 of an 8-bucket table (capacity=4, table_size=8)
    // to force clusters that wrap across index 7 -> 0 -> 1 -> 2, and force exact == dim_free hash collisions!
    const wrapCollideHash = struct {
        fn run(k: Key) HashPair {
            const bucket: u64 = if (k.expr.len > 0 and (k.expr[0] & 1) == 0) 7 else 6;
            const h: u64 = (bucket | (0x5a00 + @as(u64, k.expr.len % 2) * 0x100));
            return .{ .exact = h, .dim_free = h };
        }
    }.run;

    const TestCache = PlaneCacheCustom(4, 300, wrapCollideHash);
    var cache: TestCache = .{};
    defer cache.deinit(alloc);

    const RefEntry = struct {
        expr: []const u8,
        w: usize,
        h: usize,
        lut_bytes: usize,
        ptr: *Entry,
    };
    var ref_all: std.ArrayList(RefEntry) = .empty;
    defer ref_all.deinit(alloc);

    const makeKey = struct {
        fn k(expr_s: []const u8, w: usize, h: usize, c_info: cpu_mod.CpuInfo, prof: interp.TargetProfile) Key {
            return .{
                .expr = expr_s,
                .src_fmts = &src_fmts,
                .dst_fmt = u8_fmt,
                .width = w,
                .height = h,
                .opt_int = false,
                .default_boundary = .clamp,
                .cpu = c_info,
                .profile = prof,
            };
        }
    }.k;

    const makeExec = struct {
        fn build(a: std.mem.Allocator, expr_s: []const u8, lut_len: usize) !CompiledPlane {
            if (lut_len > 0) return testTableLut(a, lut_len, 0);
            var pr = try parser.parse(a, expr_s, .{ .mode = .expr, .num_clips = 1 });
            errdefer pr.deinit(a);
            const props = try buildPropBindings(a, &pr);
            return .{ .scalar = .{ .prog = pr, .props = props } };
        }
    }.build;

    const verifyModel = struct {
        fn check(c: *const TestCache, ref: []const RefEntry) !void {
            const none_idx: u16 = std.math.maxInt(u16);
            try std.testing.expectEqual(ref.len, c.count);

            // 1. Check global LRU forward (head -> tail) AND backward (tail -> head).
            var idx: usize = 0;
            var cur = c.lru_head;
            var prev: u16 = none_idx;
            var ref_lut_total: usize = 0;
            while (cur != none_idx) : (idx += 1) {
                try std.testing.expect(idx < ref.len);
                try std.testing.expectEqual(prev, c.slots[cur].prev_lru);
                try std.testing.expectEqual(ref[idx].ptr, c.slots[cur].entry.?);
                try std.testing.expectEqual(ref[idx].lut_bytes, c.slots[cur].lut_bytes);
                ref_lut_total += ref[idx].lut_bytes;
                prev = cur;
                cur = c.slots[cur].next_lru;
            }
            try std.testing.expectEqual(ref.len, idx);
            try std.testing.expectEqual(prev, c.lru_tail);
            try std.testing.expectEqual(ref_lut_total, c.lut_bytes);

            // 2. Check LUT-only LRU forward AND backward order matches relative order of LUTs in `ref`.
            var lut_cur = c.lut_head;
            var lut_prev: u16 = none_idx;
            for (ref) |re| {
                if (re.lut_bytes == 0) continue;
                try std.testing.expect(lut_cur != none_idx);
                try std.testing.expectEqual(lut_prev, c.slots[lut_cur].prev_lut);
                try std.testing.expectEqual(re.ptr, c.slots[lut_cur].entry.?);
                lut_prev = lut_cur;
                lut_cur = c.slots[lut_cur].next_lut;
            }
            try std.testing.expectEqual(none_idx, lut_cur);
            try std.testing.expectEqual(lut_prev, c.lut_tail);

            // 3. Verify open-addressed table probe integrity for every live slot (no broken backward-shift chains).
            var occupied_buckets: usize = 0;
            for (c.index) |raw| {
                if (raw != 0) occupied_buckets += 1;
            }
            try std.testing.expectEqual(ref.len, occupied_buckets);
            for (ref) |re| {
                const k = re.ptr.key;
                const found = c.findSlotLocked(wrapCollideHash(k).exact, k) orelse return error.TestUnexpectedResult;
                try std.testing.expectEqual(re.ptr, c.slots[found].entry.?);
            }
        }
    }.check;

    const refTouch = struct {
        fn run(ref: *std.ArrayList(RefEntry), ptr: *Entry) void {
            for (ref.items, 0..) |item, i| {
                if (item.ptr == ptr) {
                    const saved = ref.orderedRemove(i);
                    ref.insertAssumeCapacity(0, saved);
                    return;
                }
            }
            unreachable;
        }
    }.run;

    const refInsert = struct {
        fn run(a: std.mem.Allocator, ref: *std.ArrayList(RefEntry), re: RefEntry) !void {
            if (re.lut_bytes > 0) {
                while (true) {
                    var total: usize = 0;
                    for (ref.items) |it| total += it.lut_bytes;
                    if (total + re.lut_bytes <= 300) break;
                    // Evict LRU LUT (last LUT in ref).
                    var victim: ?usize = null;
                    var j = ref.items.len;
                    while (j > 0) {
                        j -= 1;
                        if (ref.items[j].lut_bytes > 0) {
                            victim = j;
                            break;
                        }
                    }
                    _ = ref.orderedRemove(victim.?);
                }
            }
            if (ref.items.len == 4) {
                _ = ref.pop();
            }
            try ref.insert(a, 0, re);
        }
    }.run;

    // Step 1: Insert 0x0 LUT "x 1 +" (100B), 0x0 LUT "x 2 +" (100B), exact JIT "x 1 +" (32x4, collides hash with 0x0!), and JIT "x 3 +" (16x4).
    const e_lut1 = try testAdmit(&cache, alloc, null, makeKey("x 1 +", 0, 0, cpu, profile), try makeExec(alloc, "x 1 +", 100));
    try refInsert(alloc, &ref_all, .{ .expr = "x 1 +", .w = 0, .h = 0, .lut_bytes = 100, .ptr = e_lut1 });
    cache.release(alloc, e_lut1);
    try verifyModel(&cache, ref_all.items);

    const e_lut2 = try testAdmit(&cache, alloc, null, makeKey("x 2 +", 0, 0, cpu, profile), try makeExec(alloc, "x 2 +", 100));
    try refInsert(alloc, &ref_all, .{ .expr = "x 2 +", .w = 0, .h = 0, .lut_bytes = 100, .ptr = e_lut2 });
    cache.release(alloc, e_lut2);
    try verifyModel(&cache, ref_all.items);

    // Exact (32x4) entry shares hash with 0x0 LUT "x 1 +"!
    const e_exact1 = try testAdmit(&cache, alloc, null, makeKey("x 1 +", 32, 4, cpu, profile), try makeExec(alloc, "x 1 +", 0));
    try refInsert(alloc, &ref_all, .{ .expr = "x 1 +", .w = 32, .h = 4, .lut_bytes = 0, .ptr = e_exact1 });
    cache.release(alloc, e_exact1);
    try verifyModel(&cache, ref_all.items);

    // Step 2: Verify `getOrDim` prefers exact (32x4) over colliding 0x0, while (64x8) falls back to 0x0!
    {
        const hit_exact = cache.getOrDim(makeKey("x 1 +", 32, 4, cpu, profile)) orelse return error.TestUnexpectedResult;
        try std.testing.expectEqual(e_exact1, hit_exact);
        refTouch(&ref_all, hit_exact);
        cache.release(alloc, hit_exact);
        try verifyModel(&cache, ref_all.items);

        const hit_dim = cache.getOrDim(makeKey("x 1 +", 64, 8, cpu, profile)) orelse return error.TestUnexpectedResult;
        try std.testing.expectEqual(e_lut1, hit_dim);
        refTouch(&ref_all, hit_dim);
        cache.release(alloc, hit_dim);
        try verifyModel(&cache, ref_all.items);
    }

    // Step 3: Insert JIT "x 3 +" -> fills capacity=4 (all wrapped in buckets 6,7,0,1).
    const e_jit3 = try testAdmit(&cache, alloc, null, makeKey("x 3 +", 16, 4, cpu, profile), try makeExec(alloc, "x 3 +", 0));
    try refInsert(alloc, &ref_all, .{ .expr = "x 3 +", .w = 16, .h = 4, .lut_bytes = 0, .ptr = e_jit3 });
    cache.release(alloc, e_jit3);
    try verifyModel(&cache, ref_all.items);

    // Step 4: Duplicate insert of "x 2 +" (0x0) must dedup, touch both LRU lists, and destroy duplicate cleanly.
    {
        const dup = try testAdmit(&cache, alloc, null, makeKey("x 2 +", 0, 0, cpu, profile), try makeExec(alloc, "x 2 +", 100));
        try std.testing.expectEqual(e_lut2, dup);
        refTouch(&ref_all, dup);
        cache.release(alloc, dup);
        try verifyModel(&cache, ref_all.items);
    }

    // Step 5: Multi-LUT eviction in ONE insert! Insert a 250B LUT (budget=300B, current=200B -> must evict BOTH e_lut1 and e_lut2!).
    // This triggers two backward-shift deletions inside a wrapped cluster while preserving e_exact1 and e_jit3!
    const e_lut3 = try testAdmit(&cache, alloc, null, makeKey("x 4 +", 0, 0, cpu, profile), try makeExec(alloc, "x 4 +", 250));
    try refInsert(alloc, &ref_all, .{ .expr = "x 4 +", .w = 0, .h = 0, .lut_bytes = 250, .ptr = e_lut3 });
    cache.release(alloc, e_lut3);
    try verifyModel(&cache, ref_all.items);
    try std.testing.expectEqual(@as(usize, 3), cache.count);
    try std.testing.expectEqual(@as(usize, 250), cache.lut_bytes);

    // Step 6: Insert two more JIT entries to force capacity eviction across wrapped buckets.
    for ([_][]const u8{ "x 5 +", "x 6 +" }) |expr_s| {
        const ej = try testAdmit(&cache, alloc, null, makeKey(expr_s, 8, 8, cpu, profile), try makeExec(alloc, expr_s, 0));
        try refInsert(alloc, &ref_all, .{ .expr = expr_s, .w = 8, .h = 8, .lut_bytes = 0, .ptr = ej });
        cache.release(alloc, ej);
        try verifyModel(&cache, ref_all.items);
    }
}

test "PlaneCacheWithLutCap bypasses admission for oversized single LUT without evicting healthy entries" {
    const alloc = std.testing.allocator;
    const cpu = cpu_mod.detectWithOverride(null);
    const profile = interp.TargetProfile.fromCpu(cpu);
    const u8_fmt: ast.SampleFormat = .{ .is_float = false, .bits_per_sample = 8 };
    const src_fmts = [_]ast.SampleFormat{u8_fmt};

    // Configurable 1 KiB LUT cap: 256B LUT fits, 2 KiB LUT exceeds cap -> must bypass without evicting!
    var cache: PlaneCacheWithLutCap(4, 1024) = .{};
    defer cache.deinit(alloc);

    const k_small: Key = .{
        .expr = "x 1 +",
        .src_fmts = &src_fmts,
        .dst_fmt = u8_fmt,
        .width = 0,
        .height = 0,
        .opt_int = false,
        .default_boundary = .clamp,
        .cpu = cpu,
        .profile = profile,
    };
    const e_small = try testAdmit(&cache, alloc, null, k_small, try testTableLut(alloc, 256, 1));
    cache.release(alloc, e_small);
    try std.testing.expectEqual(@as(usize, 1), cache.count);
    try std.testing.expectEqual(@as(usize, 256), cache.lut_bytes);

    var k_big = k_small;
    k_big.expr = "x 2 +";
    const e_big = try testAdmit(&cache, alloc, null, k_big, try testTableLut(alloc, 2048, 2));
    // Oversized entry bypasses cache: ref_count == 1 (caller only), healthy entry untouched!
    try std.testing.expectEqual(@as(usize, 1), e_big.ref_count);
    try std.testing.expectEqual(@as(usize, 1), cache.count);
    try std.testing.expectEqual(@as(usize, 256), cache.lut_bytes);
    try std.testing.expectEqual(@as(usize, 1), cache.live_external_refs);
    try std.testing.expect(cache.get(k_big) == null);
    const small_still = cache.get(k_small) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(e_small, small_still);
    cache.release(alloc, small_still);
    // Releasing caller-only oversized entry destroys it cleanly.
    cache.release(alloc, e_big);
    try std.testing.expectEqual(@as(usize, 0), cache.live_external_refs);

    // Closed cache must still reject oversized admission with `error.CacheClosed`.
    try std.testing.expect(cache.shutdown(alloc));
    var big_closed = try testTableLut(alloc, 2048, 3);
    defer big_closed.deinit(alloc, null);
    try std.testing.expectError(error.CacheClosed, cache.putOrGet(alloc, k_big, big_closed));
    try std.testing.expectEqual(@as(usize, 0), cache.count);
    try std.testing.expectEqual(@as(usize, 0), cache.live_external_refs);
}

test "PlaneCache Phase 3 dimension-free SIMD u8 LUT build counts and entry identity across YUV420P8 planes" {
    const alloc = std.testing.allocator;
    const cpu = cpu_mod.detectWithOverride(null);
    if (!aj.Builder.cpuSupportsFastLutU8(cpu)) return;
    const profile = interp.TargetProfile.fromCpu(cpu);
    const u8_fmt: ast.SampleFormat = .{ .is_float = false, .bits_per_sample = 8 };
    const src_fmts = [_]ast.SampleFormat{u8_fmt};

    // Helper mirroring static/dynamic Expr admission via the shared production `storageKey()`.
    const admitU8Lut = struct {
        fn run(
            a: std.mem.Allocator,
            c: *PlaneCache(16),
            r: *aj.Runtime,
            c_info: cpu_mod.CpuInfo,
            prof: interp.TargetProfile,
            expr_s: []const u8,
            w: usize,
            h: usize,
            meta: ?builtins_mod.ClipMeta,
            rb: []const builtins_mod.BuiltinVal,
            build_count: *usize,
        ) !*Entry {
            const plane_key: Key = .{
                .expr = expr_s,
                .src_fmts = &src_fmts,
                .dst_fmt = u8_fmt,
                .width = w,
                .height = h,
                .opt_int = false,
                .default_boundary = .clamp,
                .cpu = c_info,
                .profile = prof,
                .resolved_builtins = rb,
            };
            if (c.getOrDim(plane_key)) |hit| return hit;

            {
                var plan = try testPlan(a, plane_key, meta, @intCast(w), @intCast(h), aj.Builder.cpuSupportsFastLutU8(c_info));
                defer plan.deinit(a);
                try std.testing.expectEqual(analysis_mod.LutKind.lut1d_u8, plan.info.lut_kind);
            }
            const lut = try testLut(a, r, plane_key, meta, @intCast(w), @intCast(h), aj.Builder.cpuSupportsFastLutU8(c_info));
            build_count.* += 1;
            // Review 7 #1: scalar fallback must never false-pass this SIMD sharing test.
            try std.testing.expect(lut.simd_kernel != null);

            var exec: CompiledPlane = .{ .lut = lut };
            errdefer exec.deinit(a, r);
            const store_key = storageKey(plane_key, &exec);
            try std.testing.expectEqual(@as(usize, 0), store_key.width);
            try std.testing.expectEqual(@as(usize, 0), store_key.height);
            return try c.putOrGet(a, store_key, exec);
        }
    }.run;

    // Case 1: Literal YUV420P8 gamma `x 255 / 2.2 pow 255 *`:
    //   34x8 (Y) + 17x4 (U) + 17x4 (V) -> exactly 1 build shared across all 3 planes,
    //   and 0 additional builds at 64x16 / 32x8!
    {
        var cache: PlaneCache(16) = .{};
        defer cache.deinit(alloc);
        const rt = try testLease(cache.acquireRuntime());
        defer cache.releaseRuntime();

        var builds: usize = 0;
        const expr = "x 255 / 2.2 pow 255 *";
        const e_y = try admitU8Lut(alloc, &cache, rt, cpu, profile, expr, 34, 8, null, &.{}, &builds);
        defer cache.release(alloc, e_y);
        const e_u = try admitU8Lut(alloc, &cache, rt, cpu, profile, expr, 17, 4, null, &.{}, &builds);
        defer cache.release(alloc, e_u);
        const e_v = try admitU8Lut(alloc, &cache, rt, cpu, profile, expr, 17, 4, null, &.{}, &builds);
        defer cache.release(alloc, e_v);

        try std.testing.expectEqual(@as(usize, 1), builds);
        try std.testing.expectEqual(@as(usize, 1), cache.count);
        try std.testing.expectEqual(e_y, e_u);
        try std.testing.expectEqual(e_y, e_v);
        try std.testing.expect(e_y.exec == .lut and e_y.exec.lut.simd_kernel != null);

        // New resolution (64x16 Y, 32x8 U/V) adds 0 builds and hits the exact same entry.
        const e_y2 = try admitU8Lut(alloc, &cache, rt, cpu, profile, expr, 64, 16, null, &.{}, &builds);
        defer cache.release(alloc, e_y2);
        const e_u2 = try admitU8Lut(alloc, &cache, rt, cpu, profile, expr, 32, 8, null, &.{}, &builds);
        defer cache.release(alloc, e_u2);
        try std.testing.expectEqual(@as(usize, 1), builds);
        try std.testing.expectEqual(e_y, e_y2);
        try std.testing.expectEqual(e_y, e_u2);
    }

    // Case 2: YUV420P8 `x 255 / 2.2 pow plane_max *` (static Create compiles both limited & full variants):
    //   Full-range resolves to 255 on Y/U/V (1 shared entry); limited resolves to 235 (Y) and 240 (U/V)
    //   -> exactly 3 distinct SIMD entries total (down from 4 baseline), and 0 additional builds at 64x16/32x8!
    {
        var cache: PlaneCache(16) = .{};
        defer cache.deinit(alloc);
        const rt = try testLease(cache.acquireRuntime());
        defer cache.releaseRuntime();

        var builds: usize = 0;
        const expr = "x 255 / 2.2 pow plane_max *";
        const p_dims = [_][2]usize{ .{ 34, 8 }, .{ 17, 4 }, .{ 17, 4 } };
        var full_entries: [3]*Entry = undefined;
        var lim_entries: [3]*Entry = undefined;
        for (0..3) |p| {
            const m_full: builtins_mod.ClipMeta = .{ .is_float = false, .bits_per_sample = 8, .color_family = .yuv, .plane_idx = @intCast(p), .is_full_range = true };
            const m_lim: builtins_mod.ClipMeta = .{ .is_float = false, .bits_per_sample = 8, .color_family = .yuv, .plane_idx = @intCast(p), .is_full_range = false };
            const rb_full = [_]builtins_mod.BuiltinVal{try builtins_mod.resolveMeta(.plane_max, m_full)};
            const rb_lim = [_]builtins_mod.BuiltinVal{try builtins_mod.resolveMeta(.plane_max, m_lim)};
            full_entries[p] = try admitU8Lut(alloc, &cache, rt, cpu, profile, expr, p_dims[p][0], p_dims[p][1], m_full, &rb_full, &builds);
            lim_entries[p] = try admitU8Lut(alloc, &cache, rt, cpu, profile, expr, p_dims[p][0], p_dims[p][1], m_lim, &rb_lim, &builds);
        }
        defer {
            for (full_entries) |e| cache.release(alloc, e);
            for (lim_entries) |e| cache.release(alloc, e);
        }

        try std.testing.expectEqual(@as(usize, 3), builds);
        try std.testing.expectEqual(@as(usize, 3), cache.count);
        // Full-range 255 is shared across Y, U, V.
        try std.testing.expectEqual(full_entries[0], full_entries[1]);
        try std.testing.expectEqual(full_entries[0], full_entries[2]);
        // Limited-range Y (235) is distinct from limited-range U/V (240), and U/V share one entry.
        try std.testing.expect(lim_entries[0] != lim_entries[1]);
        try std.testing.expect(lim_entries[0] != full_entries[0]);
        try std.testing.expect(lim_entries[1] != full_entries[0]);
        try std.testing.expectEqual(lim_entries[1], lim_entries[2]);
        // Every distinct entry must hold a live SIMD kernel (not scalar fallback).
        try std.testing.expect(full_entries[0].exec == .lut and full_entries[0].exec.lut.simd_kernel != null);
        try std.testing.expect(lim_entries[0].exec == .lut and lim_entries[0].exec.lut.simd_kernel != null);
        try std.testing.expect(lim_entries[1].exec == .lut and lim_entries[1].exec.lut.simd_kernel != null);

        // Repeating at 64x16 / 32x8 adds 0 builds and hits the exact same 3 entries.
        const p_dims2 = [_][2]usize{ .{ 64, 16 }, .{ 32, 8 }, .{ 32, 8 } };
        for (0..3) |p| {
            const m_full: builtins_mod.ClipMeta = .{ .is_float = false, .bits_per_sample = 8, .color_family = .yuv, .plane_idx = @intCast(p), .is_full_range = true };
            const m_lim: builtins_mod.ClipMeta = .{ .is_float = false, .bits_per_sample = 8, .color_family = .yuv, .plane_idx = @intCast(p), .is_full_range = false };
            const rb_full = [_]builtins_mod.BuiltinVal{try builtins_mod.resolveMeta(.plane_max, m_full)};
            const rb_lim = [_]builtins_mod.BuiltinVal{try builtins_mod.resolveMeta(.plane_max, m_lim)};
            const ef = try admitU8Lut(alloc, &cache, rt, cpu, profile, expr, p_dims2[p][0], p_dims2[p][1], m_full, &rb_full, &builds);
            defer cache.release(alloc, ef);
            const el = try admitU8Lut(alloc, &cache, rt, cpu, profile, expr, p_dims2[p][0], p_dims2[p][1], m_lim, &rb_lim, &builds);
            defer cache.release(alloc, el);
            try std.testing.expectEqual(full_entries[p], ef);
            try std.testing.expectEqual(lim_entries[p], el);
        }
        try std.testing.expectEqual(@as(usize, 3), builds);
    }
}
