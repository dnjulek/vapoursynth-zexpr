//! Flat arena DAG / SSA IR types for `zexpr`.

const std = @import("std");
const builtins_mod = @import("builtins.zig");

pub const NodeIdx = u32;
pub const ClipId = u16;
pub const invalid_node: NodeIdx = std.math.maxInt(NodeIdx);

pub const BoundaryMode = enum(u8) {
    clamp = 0,
    mirror = 1,
};

pub const ExprMode = enum(u8) {
    expr,
    select,
    prop_expr,
};

pub const SampleFormat = struct {
    is_float: bool,
    bits_per_sample: u8,

    pub fn isSupportedV1(self: SampleFormat) bool {
        if (self.is_float) {
            return self.bits_per_sample == 16 or self.bits_per_sample == 32;
        }
        return (self.bits_per_sample >= 8 and self.bits_per_sample <= 16) or self.bits_per_sample == 32;
    }

    pub fn bytesPerSample(self: SampleFormat) u8 {
        if (self.is_float) {
            return if (self.bits_per_sample == 16) 2 else 4;
        }
        if (self.bits_per_sample <= 8) return 1;
        if (self.bits_per_sample <= 16) return 2;
        return 4;
    }

    pub fn peakInt(self: SampleFormat) u32 {
        std.debug.assert(!self.is_float and ((self.bits_per_sample >= 8 and self.bits_per_sample <= 16) or self.bits_per_sample == 32));
        if (self.bits_per_sample == 32) return std.math.maxInt(u32);
        return (@as(u32, 1) << @intCast(self.bits_per_sample)) - 1;
    }

    pub fn peakF32(self: SampleFormat) f32 {
        if (self.is_float) return 1.0;
        // Largest exact f32 < 2^31 (`0x7FFF_FF80`), keeping `cvt_round_f32_to_i32` in-range on x86 and AArch64.
        if (self.bits_per_sample == 32) return 2147483520.0;
        return @floatFromInt(self.peakInt());
    }
};

pub const UnaryOp = enum(u8) {
    neg,
    abs,
    sqrt,
    rsqrt,
    sgn,
    trunc,
    floor,
    ceil,
    round,
    not,
    bitnot,
    exp,
    exp2,
    log,
    log2,
    sin,
    cos,
    tan,
    atan,
    asin,
    acos,
    /// Explicit i32 -> f32 conversion used under `opt=1` to preserve float typing across rewrites.
    to_f32,
};

pub const BinaryOp = enum(u8) {
    add,
    sub,
    mul,
    div,
    /// Atomic modulo op (`a - trunc(a / b) * b` unfused). Never decomposed in the optimizer
    /// so FMA fusion cannot touch it.
    mod,
    min,
    max,
    pow,
    atan2,
    copysign,
    lt,
    le,
    gt,
    ge,
    eq,
    and_op,
    or_op,
    xor_op,
    bitand,
    bitor,
    bitxor,
};

pub const TernaryOp = enum(u8) {
    /// `a > 0 ? b : c` (ordered comparison; NaN takes `c`).
    select,
    /// Fused multiply-add variants created by the optimizer when the target has FMA:
    /// `fmadd(a, b, c) = a * b + c`
    fmadd,
    /// `fmsub(a, b, c) = a * b - c`
    fmsub,
    /// `fnmadd(a, b, c) = c - a * b`
    fnmadd,
    /// `fnmsub(a, b, c) = -(a * b) - c`
    fnmsub,
    /// `linstep(lo, hi, x) = clamp((x - lo) / (hi - lo), 0, 1)`; always expanded in `lower.zig` Phase B
    /// after Phase A constant folding so literal and constant-expression bounds share the exact-endpoint path.
    linstep,
};

pub const ArgExtremaKind = enum(u8) {
    argmin,
    argmax,
};

pub const PropSpec = union(enum) {
    frame_prop: struct {
        clip: ClipId,
        name: []u8,
    },
    /// Built-in constant slot used by scalar filters (`Select`, `PropExpr`) reusing `.load_prop` (D2).
    builtin: builtins_mod.BuiltinSpec,
};

pub const ArgSortSpec = struct {
    args_start: u32,
    count: u16,
};

pub const Node = union(enum) {
    const_f32: f32,
    /// Used in `opt=1` integer evaluation mode for exact 32-bit integer literals.
    const_i32: i32,
    load_clip: ClipId,
    load_rel: struct {
        clip: ClipId,
        dx: i32,
        dy: i32,
        mode: BoundaryMode,
    },
    load_abs: struct {
        clip: ClipId,
        x: NodeIdx,
        y: NodeIdx,
        mode: BoundaryMode,
    },
    load_prop: u32,
    coord_x,
    coord_y,
    frame_n,
    plane_width,
    plane_height,
    unary: struct {
        op: UnaryOp,
        a: NodeIdx,
    },
    binary: struct {
        op: BinaryOp,
        a: NodeIdx,
        b: NodeIdx,
    },
    ternary: struct {
        op: TernaryOp,
        a: NodeIdx,
        b: NodeIdx,
        c: NodeIdx,
    },
    arg_extrema: struct {
        kind: ArgExtremaKind,
        args_start: u32,
        count: u16,
    },
    arg_sort_elem: struct {
        sort_id: u32,
        rank: u16,
    },
};

const PropKey = struct {
    clip: ClipId,
    name: []const u8,
};

const PropKeyContext = struct {
    pub fn hash(_: @This(), k: PropKey) u64 {
        var h = std.hash.Wyhash.init(@as(u64, k.clip));
        h.update(k.name);
        return h.final();
    }

    pub fn eql(_: @This(), a: PropKey, b: PropKey) bool {
        return a.clip == b.clip and std.mem.eql(u8, a.name, b.name);
    }
};

const BuiltinSpecContext = struct {
    pub fn hash(_: @This(), k: builtins_mod.BuiltinSpec) u64 {
        var h = std.hash.Wyhash.init(0);
        k.hashInto(&h);
        return h.final();
    }

    pub fn eql(_: @This(), a: builtins_mod.BuiltinSpec, b: builtins_mod.BuiltinSpec) bool {
        return a.eql(b);
    }
};

pub const Program = struct {
    nodes: std.ArrayList(Node) = .empty,
    root: NodeIdx = invalid_node,
    props: std.ArrayList(PropSpec) = .empty,
    prop_map: std.HashMapUnmanaged(PropKey, u32, PropKeyContext, std.hash_map.default_max_load_percentage) = .empty,
    builtin_prop_map: std.HashMapUnmanaged(builtins_mod.BuiltinSpec, u32, BuiltinSpecContext, std.hash_map.default_max_load_percentage) = .empty,
    builtins: std.ArrayList(builtins_mod.BuiltinSpec) = .empty,
    builtin_map: std.HashMapUnmanaged(builtins_mod.BuiltinSpec, u32, BuiltinSpecContext, std.hash_map.default_max_load_percentage) = .empty,
    extra_args: std.ArrayList(NodeIdx) = .empty,
    arg_sorts: std.ArrayList(ArgSortSpec) = .empty,
    opt_int: bool = false,
    mode: ExprMode = .expr,

    pub fn deinit(self: *Program, allocator: std.mem.Allocator) void {
        self.prop_map.deinit(allocator);
        self.builtin_prop_map.deinit(allocator);
        self.builtin_map.deinit(allocator);
        self.builtins.deinit(allocator);
        for (self.props.items) |prop| {
            switch (prop) {
                .frame_prop => |fp| allocator.free(fp.name),
                .builtin => {},
            }
        }
        self.props.deinit(allocator);
        self.extra_args.deinit(allocator);
        self.arg_sorts.deinit(allocator);
        self.nodes.deinit(allocator);
        self.* = .{};
    }

    pub fn appendNode(self: *Program, allocator: std.mem.Allocator, node: Node) std.mem.Allocator.Error!NodeIdx {
        const idx: NodeIdx = @intCast(self.nodes.items.len);
        try self.nodes.append(allocator, node);
        return idx;
    }

    pub fn internProp(self: *Program, allocator: std.mem.Allocator, clip: ClipId, name: []const u8) std.mem.Allocator.Error!u32 {
        if (self.prop_map.get(.{ .clip = clip, .name = name })) |existing| {
            return existing;
        }
        const duped = try allocator.dupe(u8, name);
        errdefer allocator.free(duped);
        const idx: u32 = @intCast(self.props.items.len);
        try self.props.append(allocator, .{ .frame_prop = .{ .clip = clip, .name = duped } });
        errdefer _ = self.props.pop();
        try self.prop_map.put(allocator, .{ .clip = clip, .name = duped }, idx);
        return idx;
    }

    pub fn internBuiltinProp(self: *Program, allocator: std.mem.Allocator, spec: builtins_mod.BuiltinSpec) std.mem.Allocator.Error!u32 {
        if (self.builtin_prop_map.get(spec)) |existing| {
            return existing;
        }
        const idx: u32 = @intCast(self.props.items.len);
        try self.props.append(allocator, .{ .builtin = spec });
        errdefer _ = self.props.pop();
        try self.builtin_prop_map.put(allocator, spec, idx);
        return idx;
    }

    pub fn internBuiltin(self: *Program, allocator: std.mem.Allocator, spec: builtins_mod.BuiltinSpec) std.mem.Allocator.Error!u32 {
        if (self.builtin_map.get(spec)) |existing| {
            return existing;
        }
        const idx: u32 = @intCast(self.builtins.items.len);
        try self.builtins.append(allocator, spec);
        errdefer _ = self.builtins.pop();
        try self.builtin_map.put(allocator, spec, idx);
        return idx;
    }
};
