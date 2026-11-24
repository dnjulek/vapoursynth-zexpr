//! Per-clip, per-frame, per-plane, and output (`dst`) built-in constants (vsexprtools `ExprToken` 7).
//!
//! Supported built-ins:
//! - Format-only (5): `range_min`, `range_max`, `range_size`, `mask_max`, `neutral`
//! - `_Range`-dependent on integer targets only (2): `plane_min`, `plane_max`
//!
//! Rules:
//! - Targets: `<clip>:<const>` (`x`..`w`, `srcN`), `dst:<const>`, or bare `<const>` (= `x:<const>`).
//! - Integer `8..16`-bit targets produce exact `i32` (`i32_val != null`, emitting `.const_i32` under `opt=1`).
//! - Float (`f16`/`f32`) targets always produce `.const_f32` (`i32_val = null` even for `0.0`/`1.0`, D1).
//! - Integer targets outside `8..16`-bit (`17..32`-bit) are rejected with `error.UnsupportedIntBitDepthConstant` (C4/F5).

const std = @import("std");
const ast = @import("ast.zig");

pub const BuiltinTarget = union(enum) {
    clip: ast.ClipId,
    dst,

    pub fn eql(a: BuiltinTarget, b: BuiltinTarget) bool {
        return switch (a) {
            .clip => |ca| switch (b) {
                .clip => |cb| ca == cb,
                .dst => false,
            },
            .dst => b == .dst,
        };
    }

    pub fn hashInto(self: BuiltinTarget, h: *std.hash.Wyhash) void {
        switch (self) {
            .clip => |cid| {
                const tag: u8 = 0;
                h.update(std.mem.asBytes(&tag));
                h.update(std.mem.asBytes(&cid));
            },
            .dst => {
                const tag: u8 = 1;
                h.update(std.mem.asBytes(&tag));
            },
        }
    }
};

pub const BuiltinKind = enum(u8) {
    range_min,
    range_max,
    range_size,
    mask_max,
    neutral,
    plane_min,
    plane_max,
};

pub const BuiltinSpec = struct {
    target: BuiltinTarget,
    kind: BuiltinKind,

    pub fn eql(a: BuiltinSpec, b: BuiltinSpec) bool {
        return a.kind == b.kind and a.target.eql(b.target);
    }

    pub fn hashInto(self: BuiltinSpec, h: *std.hash.Wyhash) void {
        const k: u8 = @backingInt(self.kind);
        h.update(std.mem.asBytes(&k));
        self.target.hashInto(h);
    }
};

pub const BuiltinVal = struct {
    f32_bits: u32,
    i32_val: ?i32,

    pub fn f32Val(self: BuiltinVal) f32 {
        return @bitCast(self.f32_bits);
    }

    /// Explicit field-by-field equality (D5: never compare raw struct bytes because `?i32` has padding).
    pub fn eql(a: BuiltinVal, b: BuiltinVal) bool {
        return a.f32_bits == b.f32_bits and a.i32_val == b.i32_val;
    }

    /// Explicit field-by-field hash (D5: avoids hashing undefined padding bytes in `?i32`).
    pub fn hashInto(self: BuiltinVal, h: *std.hash.Wyhash) void {
        h.update(std.mem.asBytes(&self.f32_bits));
        const has_i32: u8 = @intFromBool(self.i32_val != null);
        h.update(std.mem.asBytes(&has_i32));
        const iv: i32 = self.i32_val orelse 0;
        h.update(std.mem.asBytes(&iv));
    }
};

pub const ColorFamily = enum(u8) {
    gray = 1,
    rgb = 2,
    yuv = 3,
};

/// Format and range metadata for a single clip/plane target, decoupled from `ast.SampleFormat`
/// so `Select` and `PropExpr` can inspect any valid VapourSynth format including 17..31-bit clips (F5).
pub const ClipMeta = struct {
    is_float: bool,
    bits_per_sample: u8,
    color_family: ColorFamily,
    plane_idx: u8,
    is_full_range: bool,
};

/// Context passed to `parser.parse` via `ParseOptions.builtin_ctx` for compile-time literal resolution (D6).
pub const ResolveContext = struct {
    src_metas: []const ClipMeta,
    dst_meta: ClipMeta,
    opt_int: bool,
};

pub const ResolveError = error{
    UnsupportedIntBitDepthConstant,
};

pub fn matchKind(s: []const u8) ?BuiltinKind {
    const map = std.StaticStringMap(BuiltinKind).initComptime(.{
        .{ "range_min", BuiltinKind.range_min },
        .{ "range_max", BuiltinKind.range_max },
        .{ "range_size", BuiltinKind.range_size },
        .{ "mask_max", BuiltinKind.mask_max },
        .{ "neutral", BuiltinKind.neutral },
        .{ "plane_min", BuiltinKind.plane_min },
        .{ "plane_max", BuiltinKind.plane_max },
    });
    return map.get(s);
}

pub fn kindName(kind: BuiltinKind) []const u8 {
    return switch (kind) {
        .range_min => "range_min",
        .range_max => "range_max",
        .range_size => "range_size",
        .mask_max => "mask_max",
        .neutral => "neutral",
        .plane_min => "plane_min",
        .plane_max => "plane_max",
    };
}

pub fn isRangeKind(kind: BuiltinKind) bool {
    return kind == .plane_min or kind == .plane_max;
}

/// Returns the clip ID whose `_Range` frame property this built-in depends on, or `null` if range-independent.
/// C1 + D7: Float targets are always full range (`null`), and `dst:` reads clip `0`'s frame properties
/// so `x:` and `dst:` deduplicate onto clip `0`'s single variant bit.
pub fn rangeClipId(spec: BuiltinSpec, src_fmts: []const ast.SampleFormat, dst_fmt: ast.SampleFormat) ?ast.ClipId {
    if (!isRangeKind(spec.kind)) return null;
    return switch (spec.target) {
        .dst => if (dst_fmt.is_float) null else 0,
        .clip => |cid| if (src_fmts[cid].is_float) null else cid,
    };
}

/// Pure host-free `_Range` resolver (C7 + C12):
/// 1. Float formats are always full range.
/// 2. Exact `_Range` integer: `0` = limited (`false`), `1` = full (`true`); other values treated as missing.
/// 3. Pre-R74 legacy `_ColorRange` fallback: `1` = limited (`false`), `0` = full (`true`); other values ignored.
/// 4. Missing default: RGB = full (`true`), YUV/GRAY = limited (`false`).
pub fn resolveRangeFlag(
    opt_range: ?i64,
    opt_legacy_color_range: ?i64,
    is_float: bool,
    color_family: ColorFamily,
) bool {
    if (is_float) return true;
    if (opt_range) |r| {
        if (r == 0) return false;
        if (r == 1) return true;
    }
    if (opt_legacy_color_range) |cr| {
        if (cr == 1) return false;
        if (cr == 0) return true;
    }
    return color_family == .rgb;
}

/// Resolve a built-in constant directly from its target's `ClipMeta`.
pub fn resolveMeta(kind: BuiltinKind, meta: ClipMeta) ResolveError!BuiltinVal {
    const is_chroma = meta.color_family == .yuv and meta.plane_idx > 0;
    if (meta.is_float) {
        const f: f32 = switch (kind) {
            .range_min, .plane_min => if (is_chroma) -0.5 else 0.0,
            .range_max, .plane_max => if (is_chroma) 0.5 else 1.0,
            .range_size, .mask_max => 1.0,
            .neutral => 0.0,
        };
        // D1: Float targets always leave `i32_val = null` so `opt=1` emits `.const_f32` even for `0.0`/`1.0`.
        return .{ .f32_bits = @bitCast(f), .i32_val = null };
    }

    // C4 + F5: Value-range constants support 8..16-bit integer targets only.
    if (meta.bits_per_sample < 8 or meta.bits_per_sample > 16) {
        return error.UnsupportedIntBitDepthConstant;
    }

    const b: u5 = @intCast(meta.bits_per_sample);
    const shift8: u5 = @intCast(meta.bits_per_sample - 8);
    const peak: i32 = (@as(i32, 1) << b) - 1;

    const iv: i32 = switch (kind) {
        .range_min => 0,
        .range_max => peak,
        .range_size => @as(i32, 1) << b,
        .mask_max => peak,
        .neutral => @as(i32, 1) << @intCast(meta.bits_per_sample - 1),
        .plane_min => if (meta.is_full_range) 0 else @as(i32, 16) << shift8,
        .plane_max => if (meta.is_full_range) peak else (if (is_chroma) @as(i32, 240) else @as(i32, 235)) << shift8,
    };
    const fv: f32 = @floatFromInt(iv);
    return .{ .f32_bits = @bitCast(fv), .i32_val = iv };
}

pub fn resolveOne(spec: BuiltinSpec, src_metas: []const ClipMeta, dst_meta: ClipMeta) ResolveError!BuiltinVal {
    const meta = switch (spec.target) {
        .clip => |cid| src_metas[cid],
        .dst => dst_meta,
    };
    return resolveMeta(spec.kind, meta);
}

pub fn resolveAll(
    specs: []const BuiltinSpec,
    src_metas: []const ClipMeta,
    dst_meta: ClipMeta,
    out: []BuiltinVal,
) ResolveError!void {
    std.debug.assert(specs.len == out.len);
    for (specs, 0..) |spec, i| {
        out[i] = try resolveOne(spec, src_metas, dst_meta);
    }
}

test "resolveRangeFlag handles _Range, invalid values, legacy _ColorRange, float, and defaults" {
    // Float is always full regardless of props.
    try std.testing.expect(resolveRangeFlag(0, 1, true, .yuv));
    try std.testing.expect(resolveRangeFlag(null, null, true, .gray));

    // Exact _Range = 0 (limited) / 1 (full) wins over _ColorRange and family default.
    try std.testing.expect(!resolveRangeFlag(0, null, false, .rgb));
    try std.testing.expect(!resolveRangeFlag(0, 0, false, .rgb));
    try std.testing.expect(resolveRangeFlag(1, null, false, .yuv));
    try std.testing.expect(resolveRangeFlag(1, 1, false, .gray));

    // Invalid _Range values (-1, 2) fall back to _ColorRange / family default (C7).
    try std.testing.expect(resolveRangeFlag(-1, null, false, .rgb));
    try std.testing.expect(!resolveRangeFlag(2, null, false, .yuv));
    try std.testing.expect(resolveRangeFlag(2, 0, false, .yuv));
    try std.testing.expect(!resolveRangeFlag(-1, 1, false, .rgb));

    // Legacy _ColorRange fallback (pre-R74): 1 = limited, 0 = full; invalid ignored.
    try std.testing.expect(!resolveRangeFlag(null, 1, false, .rgb));
    try std.testing.expect(resolveRangeFlag(null, 0, false, .yuv));
    try std.testing.expect(!resolveRangeFlag(null, 99, false, .yuv));
    try std.testing.expect(resolveRangeFlag(null, -2, false, .rgb));

    // Missing defaults: RGB = full, YUV/GRAY = limited.
    try std.testing.expect(resolveRangeFlag(null, null, false, .rgb));
    try std.testing.expect(!resolveRangeFlag(null, null, false, .yuv));
    try std.testing.expect(!resolveRangeFlag(null, null, false, .gray));
}

test "resolveMeta matches vsexprtools ExprToken across integer, float, luma/chroma, limited/full" {
    // 8-bit limited YUV luma vs chroma
    const y8_lim: ClipMeta = .{ .is_float = false, .bits_per_sample = 8, .color_family = .yuv, .plane_idx = 0, .is_full_range = false };
    const uv8_lim: ClipMeta = .{ .is_float = false, .bits_per_sample = 8, .color_family = .yuv, .plane_idx = 1, .is_full_range = false };
    try std.testing.expectEqual(@as(?i32, 0), (try resolveMeta(.range_min, y8_lim)).i32_val);
    try std.testing.expectEqual(@as(?i32, 255), (try resolveMeta(.range_max, y8_lim)).i32_val);
    try std.testing.expectEqual(@as(?i32, 256), (try resolveMeta(.range_size, y8_lim)).i32_val);
    try std.testing.expectEqual(@as(?i32, 255), (try resolveMeta(.mask_max, y8_lim)).i32_val);
    try std.testing.expectEqual(@as(?i32, 128), (try resolveMeta(.neutral, y8_lim)).i32_val);
    try std.testing.expectEqual(@as(?i32, 16), (try resolveMeta(.plane_min, y8_lim)).i32_val);
    try std.testing.expectEqual(@as(?i32, 235), (try resolveMeta(.plane_max, y8_lim)).i32_val);
    try std.testing.expectEqual(@as(?i32, 16), (try resolveMeta(.plane_min, uv8_lim)).i32_val);
    try std.testing.expectEqual(@as(?i32, 240), (try resolveMeta(.plane_max, uv8_lim)).i32_val);

    // 10-bit limited & 16-bit limited/full
    const y10_lim: ClipMeta = .{ .is_float = false, .bits_per_sample = 10, .color_family = .yuv, .plane_idx = 0, .is_full_range = false };
    const uv16_lim: ClipMeta = .{ .is_float = false, .bits_per_sample = 16, .color_family = .yuv, .plane_idx = 2, .is_full_range = false };
    const y16_full: ClipMeta = .{ .is_float = false, .bits_per_sample = 16, .color_family = .yuv, .plane_idx = 0, .is_full_range = true };
    try std.testing.expectEqual(@as(?i32, 64), (try resolveMeta(.plane_min, y10_lim)).i32_val);
    try std.testing.expectEqual(@as(?i32, 940), (try resolveMeta(.plane_max, y10_lim)).i32_val);
    try std.testing.expectEqual(@as(?i32, 512), (try resolveMeta(.neutral, y10_lim)).i32_val);
    try std.testing.expectEqual(@as(?i32, 1024), (try resolveMeta(.range_size, y10_lim)).i32_val);
    try std.testing.expectEqual(@as(?i32, 4096), (try resolveMeta(.plane_min, uv16_lim)).i32_val);
    try std.testing.expectEqual(@as(?i32, 61440), (try resolveMeta(.plane_max, uv16_lim)).i32_val);
    try std.testing.expectEqual(@as(?i32, 0), (try resolveMeta(.plane_min, y16_full)).i32_val);
    try std.testing.expectEqual(@as(?i32, 65535), (try resolveMeta(.plane_max, y16_full)).i32_val);

    // Float luma & chroma: D1 requires `i32_val == null` even for 0.0 and 1.0!
    const fy: ClipMeta = .{ .is_float = true, .bits_per_sample = 32, .color_family = .yuv, .plane_idx = 0, .is_full_range = true };
    const fuv: ClipMeta = .{ .is_float = true, .bits_per_sample = 16, .color_family = .yuv, .plane_idx = 1, .is_full_range = true };
    const frgb: ClipMeta = .{ .is_float = true, .bits_per_sample = 32, .color_family = .rgb, .plane_idx = 1, .is_full_range = true };
    for ([_]BuiltinKind{ .range_min, .range_max, .range_size, .mask_max, .neutral, .plane_min, .plane_max }) |k| {
        try std.testing.expectEqual(@as(?i32, null), (try resolveMeta(k, fy)).i32_val);
        try std.testing.expectEqual(@as(?i32, null), (try resolveMeta(k, fuv)).i32_val);
        try std.testing.expectEqual(@as(?i32, null), (try resolveMeta(k, frgb)).i32_val);
    }
    try std.testing.expectEqual(@as(f32, 0.0), (try resolveMeta(.range_min, fy)).f32Val());
    try std.testing.expectEqual(@as(f32, 1.0), (try resolveMeta(.range_max, fy)).f32Val());
    try std.testing.expectEqual(@as(f32, -0.5), (try resolveMeta(.range_min, fuv)).f32Val());
    try std.testing.expectEqual(@as(f32, 0.5), (try resolveMeta(.range_max, fuv)).f32Val());
    try std.testing.expectEqual(@as(f32, 1.0), (try resolveMeta(.range_size, fuv)).f32Val());
    try std.testing.expectEqual(@as(f32, 1.0), (try resolveMeta(.mask_max, fuv)).f32Val());
    try std.testing.expectEqual(@as(f32, 0.0), (try resolveMeta(.neutral, fuv)).f32Val());
    try std.testing.expectEqual(@as(f32, -0.5), (try resolveMeta(.plane_min, fuv)).f32Val());
    try std.testing.expectEqual(@as(f32, 0.5), (try resolveMeta(.plane_max, fuv)).f32Val());
    // RGB plane 1 is NOT chroma!
    try std.testing.expectEqual(@as(f32, 0.0), (try resolveMeta(.range_min, frgb)).f32Val());
    try std.testing.expectEqual(@as(f32, 1.0), (try resolveMeta(.range_max, frgb)).f32Val());

    // C4 + F5: >16-bit integer targets (24-bit, 32-bit) rejected for all 7 built-ins.
    const u24_meta: ClipMeta = .{ .is_float = false, .bits_per_sample = 24, .color_family = .gray, .plane_idx = 0, .is_full_range = false };
    const u32_meta: ClipMeta = .{ .is_float = false, .bits_per_sample = 32, .color_family = .gray, .plane_idx = 0, .is_full_range = false };
    for ([_]BuiltinKind{ .range_min, .range_max, .range_size, .mask_max, .neutral, .plane_min, .plane_max }) |k| {
        try std.testing.expectError(error.UnsupportedIntBitDepthConstant, resolveMeta(k, u24_meta));
        try std.testing.expectError(error.UnsupportedIntBitDepthConstant, resolveMeta(k, u32_meta));
    }
}

test "rangeClipId deduplicates x and dst onto clip 0 and ignores float targets" {
    const u8_fmt: ast.SampleFormat = .{ .is_float = false, .bits_per_sample = 8 };
    const f32_fmt: ast.SampleFormat = .{ .is_float = true, .bits_per_sample = 32 };

    // Integer x & y, integer dst -> x: and dst: both return clip 0, y: returns clip 1
    const int_srcs = [_]ast.SampleFormat{ u8_fmt, u8_fmt };
    try std.testing.expectEqual(@as(?ast.ClipId, null), rangeClipId(.{ .target = .{ .clip = 0 }, .kind = .range_max }, &int_srcs, u8_fmt));
    try std.testing.expectEqual(@as(?ast.ClipId, 0), rangeClipId(.{ .target = .{ .clip = 0 }, .kind = .plane_min }, &int_srcs, u8_fmt));
    try std.testing.expectEqual(@as(?ast.ClipId, 0), rangeClipId(.{ .target = .dst, .kind = .plane_max }, &int_srcs, u8_fmt));
    try std.testing.expectEqual(@as(?ast.ClipId, 1), rangeClipId(.{ .target = .{ .clip = 1 }, .kind = .plane_min }, &int_srcs, u8_fmt));

    // Float x, integer dst -> x: returns null, dst: returns clip 0 (E2!)
    const mixed_srcs = [_]ast.SampleFormat{ f32_fmt, u8_fmt };
    try std.testing.expectEqual(@as(?ast.ClipId, null), rangeClipId(.{ .target = .{ .clip = 0 }, .kind = .plane_min }, &mixed_srcs, u8_fmt));
    try std.testing.expectEqual(@as(?ast.ClipId, 0), rangeClipId(.{ .target = .dst, .kind = .plane_max }, &mixed_srcs, u8_fmt));

    // Integer x, float dst -> x: returns clip 0, dst: returns null
    try std.testing.expectEqual(@as(?ast.ClipId, 0), rangeClipId(.{ .target = .{ .clip = 0 }, .kind = .plane_min }, &int_srcs, f32_fmt));
    try std.testing.expectEqual(@as(?ast.ClipId, null), rangeClipId(.{ .target = .dst, .kind = .plane_max }, &int_srcs, f32_fmt));
}

test "BuiltinVal field-by-field hash and equality distinguish i32_val presence" {
    const as_int: BuiltinVal = .{ .f32_bits = @bitCast(@as(f32, 1.0)), .i32_val = 1 };
    const as_flt: BuiltinVal = .{ .f32_bits = @bitCast(@as(f32, 1.0)), .i32_val = null };
    try std.testing.expect(!as_int.eql(as_flt));
    var h1 = std.hash.Wyhash.init(0);
    var h2 = std.hash.Wyhash.init(0);
    as_int.hashInto(&h1);
    as_flt.hashInto(&h2);
    try std.testing.expect(h1.final() != h2.final());
}
