//! Shared VapourSynth adapter helpers handling `vapoursynth-zig` binding pitfalls.

const std = @import("std");
const vapoursynth = @import("vapoursynth");
const ast = @import("../expr/ast.zig");
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
