//! Shared VapourSynth adapter helpers handling `vapoursynth-zig` binding pitfalls.

const std = @import("std");
const vapoursynth = @import("vapoursynth");
const ast = @import("../expr/ast.zig");
const builtins = @import("../expr/builtins.zig");
const plane_cache = @import("../jit/plane_cache.zig");

const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;

pub const ArgError = error{Invalid};

pub fn fail(map_out: ZAPI.ZMap(?*vs.Map), comptime prefix: []const u8, comptime fmt: []const u8, args: anytype) ArgError {
    var buf: [512]u8 = undefined;
    var w: std.Io.Writer = .fixed(buf[0 .. buf.len - 1]);
    w.print(prefix ++ ": " ++ fmt, args) catch {};
    buf[w.end] = 0;
    const msg: [:0]const u8 = buf[0..w.end :0];
    map_out.setError(msg);
    return error.Invalid;
}

pub const PropBinding = plane_cache.PropBinding;
pub const freePropBindings = plane_cache.freePropBindings;
pub const buildPropBindings = plane_cache.buildPropBindings;

/// Per-frame scalar slot binding for `Select` and `PropExpr` (D2: `.load_prop` serves both
/// frame properties and built-in constants without modifying `ast.Node`).
pub const ScalarPropBinding = union(enum) {
    frame_prop: PropBinding,
    builtin: builtins.BuiltinSpec,
};

pub fn freeScalarPropBindings(allocator: std.mem.Allocator, props: []ScalarPropBinding) void {
    for (props) |p| {
        switch (p) {
            .frame_prop => |fp| allocator.free(fp.name_z),
            .builtin => {},
        }
    }
    allocator.free(props);
}

pub fn buildScalarPropBindings(
    allocator: std.mem.Allocator,
    prog: *const ast.Program,
) std.mem.Allocator.Error![]ScalarPropBinding {
    const out = try allocator.alloc(ScalarPropBinding, prog.props.items.len);
    var built: usize = 0;
    errdefer {
        for (out[0..built]) |p| {
            switch (p) {
                .frame_prop => |fp| allocator.free(fp.name_z),
                .builtin => {},
            }
        }
        allocator.free(out);
    }
    for (prog.props.items, 0..) |p, i| {
        out[i] = switch (p) {
            .frame_prop => |fp| .{ .frame_prop = .{
                .clip = fp.clip,
                .name_z = try allocator.dupeSentinel(u8, fp.name, 0),
            } },
            .builtin => |spec| .{ .builtin = spec },
        };
        built += 1;
    }
    return out;
}

pub fn colorFamilyFromVs(cf: vs.ColorFamily) ?builtins.ColorFamily {
    return switch (cf) {
        .Gray => .gray,
        .RGB => .rgb,
        .YUV => .yuv,
        else => null,
    };
}

/// C7 + D11: Unified per-frame `_Range` resolver (`_Range` exact int -> legacy `_ColorRange` -> family default).
pub fn readFrameIsFullRange(zapi: ZAPI, props: ?*const vs.Map, vf: vs.VideoFormat) bool {
    const cf = colorFamilyFromVs(vf.colorFamily) orelse .gray;
    if (vf.sampleType == .Float) return true;
    const opt_range: ?i64 = if (props != null) getOptInt64(zapi, props, "_Range") else null;
    const opt_legacy: ?i64 = if (props != null and (opt_range == null or (opt_range.? != 0 and opt_range.? != 1)))
        getOptInt64(zapi, props, "_ColorRange")
    else
        null;
    return builtins.resolveRangeFlag(opt_range, opt_legacy, false, cf);
}

/// F5: Build `ClipMeta` directly from `VideoFormat` without calling `toSampleFormat`, so `Select`/`PropExpr`
/// continue accepting 17..31-bit integer clips (e.g. `GRAY24`).
pub fn clipMetaFromVi(vf: vs.VideoFormat, plane_idx: u8, is_full_range: bool) ?builtins.ClipMeta {
    const cf = colorFamilyFromVs(vf.colorFamily) orelse return null;
    return .{
        .is_float = vf.sampleType == .Float,
        .bits_per_sample = @intCast(vf.bitsPerSample),
        .color_family = cf,
        .plane_idx = plane_idx,
        .is_full_range = if (vf.sampleType == .Float) true else is_full_range,
    };
}

pub fn clipMetaFromFrame(
    zapi: ZAPI,
    frame: ?*const vs.Frame,
    vf: vs.VideoFormat,
    plane_idx: u8,
    read_range: bool,
) ?builtins.ClipMeta {
    const cf = colorFamilyFromVs(vf.colorFamily) orelse return null;
    const is_full = if (vf.sampleType == .Float)
        true
    else if (read_range)
        readFrameIsFullRange(zapi, zapi.vsapi.getFramePropertiesRO.?(frame), vf)
    else
        (cf == .rgb);
    return .{
        .is_float = vf.sampleType == .Float,
        .bits_per_sample = @intCast(vf.bitsPerSample),
        .color_family = cf,
        .plane_idx = plane_idx,
        .is_full_range = is_full,
    };
}

pub inline fn clampFrameIdx(n: c_int, num_frames: i32) c_int {
    if (num_frames <= 0) return n;
    if (n >= num_frames) return @intCast(num_frames - 1);
    return n;
}

/// Safe `mapGetData` wrapper that allows empty strings `""` (ZMap.getData returns null for len == 0).
pub fn getDataAllowEmpty(zapi: ZAPI, map: ?*const vs.Map, key: [*:0]const u8, idx: usize) ?[]const u8 {
    var err: vs.MapPropertyError = .Success;
    const size = zapi.vsapi.mapGetDataSize.?(map, key, @intCast(idx), &err);
    if (err != .Success or size < 0) return null;
    if (size == 0) return "";
    const ptr = zapi.vsapi.mapGetData.?(map, key, @intCast(idx), &err);
    if (err != .Success or ptr == null) return null;
    return ptr.?[0..@intCast(size)];
}

/// Read all nodes from a `vnode[]` parameter with full `errdefer` cleanup.
pub fn getNodes(
    allocator: std.mem.Allocator,
    zapi: ZAPI,
    map: ?*const vs.Map,
    key: [*:0]const u8,
) (std.mem.Allocator.Error || error{ Missing, TooManyClips })![]*vs.Node {
    const count_raw = zapi.vsapi.mapNumElements.?(map, key);
    if (count_raw <= 0) return error.Missing;
    const count: usize = @intCast(count_raw);
    if (count > std.math.maxInt(ast.ClipId)) return error.TooManyClips;

    const nodes = try allocator.alloc(*vs.Node, count);
    var acquired: usize = 0;
    errdefer {
        for (nodes[0..acquired]) |n| zapi.freeNode(n);
        allocator.free(nodes);
    }

    for (0..count) |i| {
        var err: vs.MapPropertyError = .Success;
        const node = zapi.vsapi.mapGetNode.?(map, key, @intCast(i), &err);
        if (err != .Success or node == null) return error.Missing;
        nodes[i] = node.?;
        acquired += 1;
    }
    return nodes;
}

/// Read an exact `i64` optional integer argument without ZMap's saturating narrowing.
pub fn getOptInt64(zapi: ZAPI, map: ?*const vs.Map, key: [*:0]const u8) ?i64 {
    var err: vs.MapPropertyError = .Success;
    const val = zapi.vsapi.mapGetInt.?(map, key, 0, &err);
    if (err != .Success) return null;
    return val;
}

/// Convert a VapourSynth `VideoFormat` to `ast.SampleFormat`, returning null for unsupported formats
/// (8..16 and 32-bit integer, 16/32-bit float; 17..31-bit integer is rejected).
pub fn toSampleFormat(vf: vs.VideoFormat) ?ast.SampleFormat {
    const sf: ast.SampleFormat = .{
        .is_float = vf.sampleType == .Float,
        .bits_per_sample = @intCast(vf.bitsPerSample),
    };
    if (!sf.isSupportedV1()) return null;
    return sf;
}

/// D11 frame property reader:
/// `ptInt` -> `ptFloat` -> first unsigned byte (`u8`) of `ptData` (if `size > 0`) -> `missing_default`.
pub fn readFrameProp(zapi: ZAPI, props: ?*const vs.Map, name_z: [*:0]const u8, missing_default: f32) f32 {
    if (props == null) return missing_default;
    const api = zapi.vsapi;

    var err: vs.MapPropertyError = .Success;
    const iv = api.mapGetInt.?(props, name_z, 0, &err);
    if (err == .Success) return @floatFromInt(iv);

    if (err == .Type) {
        err = .Success;
        const fv = api.mapGetFloat.?(props, name_z, 0, &err);
        if (err == .Success) return @floatCast(fv);

        if (err == .Type) {
            err = .Success;
            const sz = api.mapGetDataSize.?(props, name_z, 0, &err);
            if (err == .Success and sz > 0) {
                const ptr = api.mapGetData.?(props, name_z, 0, &err);
                if (err == .Success and ptr != null) {
                    const byte: u8 = @bitCast(ptr.?[0]);
                    return @floatFromInt(byte);
                }
            }
        }
    }

    return missing_default;
}

var cached_core_align: std.atomic.Value(u32) = .init(0);

pub fn resetCoreAlignmentCache() void {
    cached_core_align.store(0, .release);
}

/// S5: Probe the running core's frame row alignment once per process via a 1x1 GRAY8 frame.
pub fn probeCoreAlignment(zapi: ZAPI) u32 {
    const cached = cached_core_align.load(.acquire);
    if (cached != 0) return cached;

    const api = zapi.vsapi;
    var fmt: vs.VideoFormat = undefined;
    if (api.getVideoFormatByID.?(&fmt, .Gray8, zapi.core) == 0) {
        return 32;
    }
    const frame = api.newVideoFrame.?(&fmt, 1, 1, null, zapi.core) orelse return 32;
    defer api.freeFrame.?(frame);
    const stride = api.getStride.?(frame, 0);
    const align_bytes: u32 = if (stride <= 0) 32 else @intCast(stride);
    cached_core_align.store(align_bytes, .release);
    return align_bytes;
}
