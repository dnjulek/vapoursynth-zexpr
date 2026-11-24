//! `zexpr.Select` VapourSynth filter adapter (`.Parallel`, interpreted per S10).
//!
//! Contracts: B2 (`@ptrFromInt(idx + 1)`), D8/X2 (clamp NaN/<=0 and +inf/>=len-1 before
//! half-away `@round`), Y6 (`@ptrCast` `frame_data` to `*[4]?*anyopaque`), Y17 (strict
//! validation of `clip_src` and `prop_src`), Y19 (`.StrictSpatial`/`.FrameReuseLastOnly`).

const std = @import("std");
const vapoursynth = @import("vapoursynth");
const ast = @import("../expr/ast.zig");
const builtins = @import("../expr/builtins.zig");
const parser = @import("../expr/parser.zig");
const lower_mod = @import("../expr/lower.zig");
const optimize_mod = @import("../expr/optimize.zig");
const interp = @import("../expr/interp.zig");
const cpu_mod = @import("../jit/cpu.zig");
const common = @import("common.zig");

const vs = vapoursynth.vapoursynth4;
const vsh = vapoursynth.vshelper;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;
pub const filter_name = "Select";
pub const params = "clip_src:vnode[];prop_src:vnode[];expr:data[];";

const max_planes = 3;

const PlaneExpr = struct {
    prog: ast.Program,
    props: []common.ScalarPropBinding,

    fn deinit(self: *PlaneExpr) void {
        self.prog.deinit(allocator);
        common.freeScalarPropBindings(allocator, self.props);
    }
};

const SelectData = struct {
    clip_src: []*vs.Node,
    prop_src: []*vs.Node,
    prop_num_frames: []i32,
    out_vi: vs.VideoInfo,
    profile: interp.TargetProfile,
    planes: [max_planes]PlaneExpr,
    num_planes: usize,

    fn deinit(self: *SelectData, zapi: ZAPI) void {
        for (self.planes[0..self.num_planes]) |*pe| pe.deinit();
        for (self.clip_src) |n| zapi.freeNode(n);
        allocator.free(self.clip_src);
        for (self.prop_src) |n| zapi.freeNode(n);
        allocator.free(self.prop_src);
        allocator.free(self.prop_num_frames);
        allocator.destroy(self);
    }
};

/// D8 / X2: Clamp NaN/<=0 -> 0 and +inf/>=len-1 -> len-1 BEFORE half-away `@round` + `@intFromFloat`.
pub fn selectClipIndex(val: f32, num_clips: usize) usize {
    if (num_clips <= 1) return 0;
    const max_idx_f: f32 = @floatFromInt(num_clips - 1);
    if (std.math.isNan(val) or val <= 0.0) return 0;
    if (val >= max_idx_f) return num_clips - 1;
    return @intFromFloat(@round(val));
}

fn selectGetFrame(
    n: c_int,
    reason: vs.ActivationReason,
    instance_data: ?*anyopaque,
    frame_data: ?*?*anyopaque,
    frame_ctx: ?*vs.FrameContext,
    core: ?*vs.Core,
    vsapi: ?*const vs.API,
) callconv(.c) ?*const vs.Frame {
    const d: *const SelectData = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, frame_ctx);
    const slots: *[4]?*anyopaque = @ptrCast(frame_data.?);

    switch (reason) {
        .Initial => {
            slots.* = .{ null, null, null, null };
            for (d.prop_src, d.prop_num_frames) |node, nf| {
                zapi.requestFrameFilter(common.clampFrameIdx(n, nf), node);
            }
            return null;
        },
        .AllFramesReady => {
            if (slots[0] == null) {
                // Pass 1: Evaluate plane expressions from `prop_src` frames and request selected `clip_src`.
                const n_props = d.prop_src.len;
                var stack_pframes: [32]?*const vs.Frame = @splat(null);
                const heap_pframes: ?[]?*const vs.Frame = if (n_props > stack_pframes.len)
                    allocator.alloc(?*const vs.Frame, n_props) catch {
                        zapi.setFilterError("Select: out of memory");
                        return null;
                    }
                else
                    null;
                defer if (heap_pframes) |hp| allocator.free(hp);
                const prop_frames: []?*const vs.Frame = if (heap_pframes) |hp| hp else stack_pframes[0..n_props];
                @memset(prop_frames, null);
                defer {
                    for (prop_frames) |mf| if (mf) |f| zapi.freeFrame(f);
                }

                for (d.prop_src, d.prop_num_frames, 0..) |node, nf, i| {
                    prop_frames[i] = zapi.getFrameFilter(common.clampFrameIdx(n, nf), node);
                }

                var chosen: [max_planes]usize = .{ 0, 0, 0 };
                for (0..d.num_planes) |p| {
                    const pe = &d.planes[p];
                    var prop_stack: [64]f32 = undefined;
                    const prop_heap: ?[]f32 = if (pe.props.len > prop_stack.len)
                        allocator.alloc(f32, pe.props.len) catch {
                            zapi.setFilterError("Select: out of memory");
                            return null;
                        }
                    else
                        null;
                    defer if (prop_heap) |ph| allocator.free(ph);
                    const prop_vals: []f32 = if (prop_heap) |ph| ph else prop_stack[0..pe.props.len];

                    for (pe.props, 0..) |pb, pi| {
                        switch (pb) {
                            .frame_prop => |fp| {
                                const fprops = zapi.vsapi.getFramePropertiesRO.?(prop_frames[fp.clip]);
                                prop_vals[pi] = common.readFrameProp(zapi, fprops, fp.name_z.ptr, 0.0);
                            },
                            .builtin => |spec| {
                                const meta = switch (spec.target) {
                                    .dst => common.clipMetaFromVi(d.out_vi.format, @intCast(p), true),
                                    .clip => |cid| blk_m: {
                                        const f = prop_frames[cid];
                                        const vf = zapi.vsapi.getVideoFrameFormat.?(f).*;
                                        break :blk_m common.clipMetaFromFrame(zapi, f, vf, @intCast(p), builtins.isRangeKind(spec.kind));
                                    },
                                } orelse {
                                    zapi.setFilterError("Select: unsupported clip format for built-in constant");
                                    return null;
                                };
                                const bv = builtins.resolveMeta(spec.kind, meta) catch {
                                    zapi.setFilterError("Select: built-in constants are unsupported on >16-bit integer clips");
                                    return null;
                                };
                                prop_vals[pi] = bv.f32Val();
                            },
                        }
                    }

                    // V6: Select exposes full clip width/height on every plane (matches akarin & cranexpr).
                    const val = interp.evalScalar(allocator, &pe.prog, d.profile, .{
                        .width = d.out_vi.width,
                        .height = d.out_vi.height,
                        .frame_n = @intCast(n),
                        .props = prop_vals,
                    }) catch {
                        zapi.setFilterError("Select: out of memory");
                        return null;
                    };

                    const idx = selectClipIndex(val, d.clip_src.len);
                    chosen[p] = idx;
                    // B2: Store `idx + 1` so clip index 0 is non-null!
                    slots[p] = @ptrFromInt(idx + 1);
                }

                // Request deduplicated selected `clip_src` frames.
                for (0..d.num_planes) |p| {
                    const idx = chosen[p];
                    var already_req = false;
                    for (0..p) |prev| {
                        if (chosen[prev] == idx) {
                            already_req = true;
                            break;
                        }
                    }
                    if (!already_req) {
                        zapi.requestFrameFilter(n, d.clip_src[idx]);
                    }
                }
                return null;
            } else {
                // Pass 2: Decode selected clip indices, assemble output frame, clear slots.
                var chosen: [max_planes]usize = .{ 0, 0, 0 };
                for (0..d.num_planes) |p| {
                    chosen[p] = @intFromPtr(slots[p].?) - 1;
                }
                slots.* = .{ null, null, null, null };

                var unique_idxs: [max_planes]usize = undefined;
                var unique_frames: [max_planes]?*const vs.Frame = .{ null, null, null };
                var n_unique: usize = 0;
                defer {
                    for (unique_frames[0..n_unique]) |mf| if (mf) |f| zapi.freeFrame(f);
                }

                var plane_srcs: [max_planes]?*const vs.Frame = .{ null, null, null };
                const plane_indices = [max_planes]c_int{ 0, 1, 2 };

                for (0..d.num_planes) |p| {
                    const idx = chosen[p];
                    var found: ?*const vs.Frame = null;
                    for (0..n_unique) |ui| {
                        if (unique_idxs[ui] == idx) {
                            found = unique_frames[ui];
                            break;
                        }
                    }
                    if (found == null) {
                        const f = zapi.getFrameFilter(n, d.clip_src[idx]);
                        unique_idxs[n_unique] = idx;
                        unique_frames[n_unique] = f;
                        n_unique += 1;
                        found = f;
                    }
                    plane_srcs[p] = found;
                }

                const dst = zapi.vsapi.newVideoFrame2.?(
                    &d.out_vi.format,
                    d.out_vi.width,
                    d.out_vi.height,
                    &plane_srcs,
                    &plane_indices,
                    plane_srcs[0],
                    core,
                ) orelse {
                    zapi.setFilterError("Select: failed to allocate output frame");
                    return null;
                };
                return dst;
            }
        },
        .Error => {
            slots.* = .{ null, null, null, null };
            return null;
        },
    }
}

fn selectFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *SelectData = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, null);
    d.deinit(zapi);
}

pub fn create(
    in: ?*const vs.Map,
    out: ?*vs.Map,
    user_data: ?*anyopaque,
    core: ?*vs.Core,
    vsapi: ?*const vs.API,
) callconv(.c) void {
    _ = user_data;
    const zapi = ZAPI.init(vsapi, core, null);
    const map_out = zapi.initZMap(out);
    createImpl(zapi, in, out, map_out, vsapi) catch |err| switch (err) {
        error.Invalid => {},
        error.OutOfMemory => map_out.setError("Select: out of memory"),
    };
}

fn createImpl(
    zapi: ZAPI,
    in: ?*const vs.Map,
    out: ?*vs.Map,
    map_out: ZAPI.ZMap(?*vs.Map),
    vsapi: ?*const vs.API,
) (common.ArgError || std.mem.Allocator.Error)!void {
    const clip_src = common.getNodes(allocator, zapi, in, "clip_src") catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.TooManyClips => return common.fail(map_out, filter_name, "too many clip_src clips (max 65535)", .{}),
        error.Missing => return common.fail(map_out, filter_name, "at least one clip_src is required", .{}),
    };
    var owns_clips = true;
    errdefer if (owns_clips) {
        for (clip_src) |n| zapi.freeNode(n);
        allocator.free(clip_src);
    };

    const prop_src = common.getNodes(allocator, zapi, in, "prop_src") catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.TooManyClips => return common.fail(map_out, filter_name, "too many prop_src clips (max 65535)", .{}),
        error.Missing => return common.fail(map_out, filter_name, "at least one prop_src is required", .{}),
    };
    var owns_props = true;
    errdefer if (owns_props) {
        for (prop_src) |n| zapi.freeNode(n);
        allocator.free(prop_src);
    };

    // Validate `clip_src`: constant and identical VideoInfo + equal numFrames (Y17).
    const vi0 = zapi.getVideoInfo(clip_src[0]);
    if (!vsh.isConstantVideoFormat(vi0)) {
        return common.fail(map_out, filter_name, "clip_src must have constant format and dimensions", .{});
    }
    for (clip_src[1..], 1..) |node, i| {
        const vi = zapi.getVideoInfo(node);
        if (!vsh.isConstantVideoFormat(vi) or !vsh.isSameVideoInfo(vi0, vi) or vi.numFrames != vi0.numFrames) {
            return common.fail(map_out, filter_name, "clip_src[{d}] must have identical format, dimensions, and length to clip_src[0]", .{i});
        }
    }

    // Validate `prop_src`: constant format (reject variable-format prop_src, Y17).
    const prop_num_frames = try allocator.alloc(i32, prop_src.len);
    var owns_prop_nf = true;
    errdefer if (owns_prop_nf) allocator.free(prop_num_frames);

    for (prop_src, 0..) |node, i| {
        const vi = zapi.getVideoInfo(node);
        if (!vsh.isConstantVideoFormat(vi)) {
            return common.fail(map_out, filter_name, "prop_src[{d}] must have constant format", .{i});
        }
        prop_num_frames[i] = vi.numFrames;
    }

    const num_planes: usize = @intCast(vi0.format.numPlanes);
    const expr_count_raw = vsapi.?.mapNumElements.?(in, "expr");
    if (expr_count_raw <= 0 or expr_count_raw > vi0.format.numPlanes) {
        return common.fail(map_out, filter_name, "expr must have between 1 and {d} elements", .{num_planes});
    }
    const expr_count: usize = @intCast(expr_count_raw);

    const profile: interp.TargetProfile = .{
        .use_fma = false,
        .isa_family = cpu_mod.IsaFamily.host(),
    };

    var planes: [max_planes]PlaneExpr = undefined;
    var built: usize = 0;
    var owns_planes = true;
    errdefer if (owns_planes) {
        for (planes[0..built]) |*pe| pe.deinit();
    };

    for (0..num_planes) |p| {
        const src_idx = if (p < expr_count) p else expr_count - 1;
        const expr_str = common.getDataAllowEmpty(zapi, in, "expr", src_idx) orelse {
            return common.fail(map_out, filter_name, "failed to read expr[{d}]", .{src_idx});
        };
        if (expr_str.len == 0) {
            return common.fail(map_out, filter_name, "expr[{d}] cannot be empty", .{src_idx});
        }

        var prog = parser.parse(allocator, expr_str, .{
            .mode = .select,
            .num_clips = @intCast(prop_src.len),
            .default_boundary = .clamp,
            .opt_int = false,
        }) catch |err| {
            return common.fail(map_out, filter_name, "parse error on plane {d}: {s}", .{ p, @errorName(err) });
        };
        errdefer prog.deinit(allocator);

        // C2 + C4/F5: Reject range-dependent `dst:` constants and >16-bit integer targets in Create,
        // inspecting only targeted clips so unrelated GRAY24 inputs remain accepted.
        for (prog.builtins.items) |spec| {
            if (spec.target == .dst and builtins.isRangeKind(spec.kind)) {
                return common.fail(
                    map_out,
                    filter_name,
                    "plane {d}: dst:{s} is unsupported in Select",
                    .{ p, builtins.kindName(spec.kind) },
                );
            }
            const vf = switch (spec.target) {
                .dst => vi0.format,
                .clip => |cid| zapi.getVideoInfo(prop_src[cid]).format,
            };
            if (common.colorFamilyFromVs(vf.colorFamily) == null) {
                return common.fail(map_out, filter_name, "plane {d}: unsupported clip color family for built-in '{s}'", .{ p, builtins.kindName(spec.kind) });
            }
            if (vf.sampleType == .Integer and (vf.bitsPerSample < 8 or vf.bitsPerSample > 16)) {
                return common.fail(
                    map_out,
                    filter_name,
                    "plane {d}: built-in '{s}' is unsupported on {d}-bit integer clip",
                    .{ p, builtins.kindName(spec.kind), vf.bitsPerSample },
                );
            }
        }

        try lower_mod.lower(allocator, &prog, profile);
        try optimize_mod.optimize(allocator, &prog, profile, .{
            .enable_value_changing = true,
            .enable_fma = false,
        });

        const bindings = try common.buildScalarPropBindings(allocator, &prog);
        planes[p] = .{ .prog = prog, .props = bindings };
        built += 1;
    }

    const total_deps = clip_src.len + prop_src.len;
    const deps = try allocator.alloc(vs.FilterDependency, total_deps);
    defer allocator.free(deps);

    const data = try allocator.create(SelectData);
    owns_clips = false;
    owns_props = false;
    owns_prop_nf = false;
    owns_planes = false;
    data.* = .{
        .clip_src = clip_src,
        .prop_src = prop_src,
        .prop_num_frames = prop_num_frames,
        .out_vi = vi0.*,
        .profile = profile,
        .planes = planes,
        .num_planes = num_planes,
    };

    var di: usize = 0;
    for (clip_src) |node| {
        deps[di] = .{ .source = node, .requestPattern = .StrictSpatial };
        di += 1;
    }
    for (prop_src, prop_num_frames) |node, nf| {
        deps[di] = .{
            .source = node,
            .requestPattern = if (nf >= data.out_vi.numFrames) .StrictSpatial else .FrameReuseLastOnly,
        };
        di += 1;
    }

    zapi.createVideoFilter(out, filter_name, &data.out_vi, selectGetFrame, selectFree, .Parallel, deps, data);
}
