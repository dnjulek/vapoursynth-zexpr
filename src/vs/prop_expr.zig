//! `zexpr.PropExpr` VapourSynth filter adapter (`.Parallel`, interpreted per S10).
//!
//! Supports both akarin style (`dict=lambda: {...}`) with exact `i64`/`f64` constant
//! pass-through (Y18) and leak-free cleanup on all exits (Y20), and cranexpr style
//! (`**kwargs`). Evaluates all expressions atomically against original input frames
//! before mutating `dst = copyFrame(src[0])`.

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
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;
pub const filter_name = "PropExpr";
pub const params = "clips:vnode[];dict:func:opt;any";

const CompiledExpr = struct {
    prog: ast.Program,
    props: []common.ScalarPropBinding,

    fn deinit(self: *CompiledExpr) void {
        self.prog.deinit(allocator);
        common.freeScalarPropBindings(allocator, self.props);
    }
};

const DataElem = union(enum) {
    delete,
    expr: CompiledExpr,
};

const KeySpec = struct {
    key_z: [:0]u8,
    value: union(enum) {
        const_ints: []i64,
        const_floats: []f64,
        data_elems: []DataElem,
    },

    fn deinit(self: *KeySpec) void {
        allocator.free(self.key_z);
        switch (self.value) {
            .const_ints => |s| allocator.free(s),
            .const_floats => |s| allocator.free(s),
            .data_elems => |elems| {
                for (elems) |*e| {
                    if (e.* == .expr) e.expr.deinit();
                }
                allocator.free(elems);
            },
        }
    }
};

const EvalResult = union(enum) {
    delete,
    int_val: i64,
    float_val: f64,
};

const PropExprData = struct {
    nodes: []*vs.Node,
    clip_num_frames: []i32,
    out_vi: vs.VideoInfo,
    profile: interp.TargetProfile,
    specs: []KeySpec,

    fn deinit(self: *PropExprData, zapi: ZAPI) void {
        for (self.specs) |*s| s.deinit();
        allocator.free(self.specs);
        for (self.nodes) |n| zapi.freeNode(n);
        allocator.free(self.nodes);
        allocator.free(self.clip_num_frames);
        allocator.destroy(self);
    }
};

fn evalResultFromF32(val: f32) EvalResult {
    // Store `Int` when `val` is finite and equals an exact `i64` round-trip, else `Float`.
    // Note: 2^63 (0x1.0p63) is exact in f32 while i64.max rounds up to 2^63, so strict `< 0x1.0p63` is required.
    const min_i64_f: f32 = -@as(f32, 0x1.0p63);
    const max_i64_excl_f: f32 = @as(f32, 0x1.0p63);
    if (std.math.isFinite(val) and @trunc(val) == val and val >= min_i64_f and val < max_i64_excl_f) {
        return .{ .int_val = @intFromFloat(val) };
    }
    return .{ .float_val = @floatCast(val) };
}

fn propExprGetFrame(
    n: c_int,
    reason: vs.ActivationReason,
    instance_data: ?*anyopaque,
    frame_data: ?*?*anyopaque,
    frame_ctx: ?*vs.FrameContext,
    core: ?*vs.Core,
    vsapi: ?*const vs.API,
) callconv(.c) ?*const vs.Frame {
    _ = frame_data;
    const d: *const PropExprData = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, frame_ctx);

    switch (reason) {
        .Initial => {
            common.requestInputs(zapi, n, d.nodes, d.clip_num_frames);
            return null;
        },
        .AllFramesReady => {
            var inputs: common.InputFrames = .{};
            defer inputs.deinit(zapi);
            if (!inputs.fetch(zapi, n, d.nodes, d.clip_num_frames)) {
                zapi.setFilterError("PropExpr: out of memory");
                return null;
            }
            const src_frames = inputs.frames;

            // Phase 1: Atomically evaluate ALL property updates against original input frames.
            const results = allocator.alloc(EvalResult, d.specs.len) catch {
                zapi.setFilterError("PropExpr: out of memory");
                return null;
            };
            defer allocator.free(results);

            const frame_u: usize = if (n >= 0) @intCast(n) else 0;
            const fw: i32 = zapi.vsapi.getFrameWidth.?(src_frames[0], 0);
            const fh: i32 = zapi.vsapi.getFrameHeight.?(src_frames[0], 0);

            for (d.specs, 0..) |*spec, si| {
                results[si] = switch (spec.value) {
                    .const_ints => |arr| .{ .int_val = arr[frame_u % arr.len] },
                    .const_floats => |arr| .{ .float_val = arr[frame_u % arr.len] },
                    .data_elems => |elems| blk: {
                        const elem = &elems[frame_u % elems.len];
                        switch (elem.*) {
                            .delete => break :blk .delete,
                            .expr => |*ce| {
                                var prop_buf: common.Scratch(f32, 64) = .{};
                                defer prop_buf.deinit();
                                const prop_vals = prop_buf.get(ce.props.len) orelse {
                                    zapi.setFilterError("PropExpr: out of memory");
                                    return null;
                                };

                                for (ce.props, 0..) |pb, pi| {
                                    switch (pb) {
                                        .frame_prop => |fp| {
                                            const fprops = zapi.vsapi.getFramePropertiesRO.?(src_frames[fp.clip]);
                                            prop_vals[pi] = common.readFrameProp(zapi, fprops, fp.name_z.ptr, 0.0);
                                        },
                                        .builtin => |bspec| {
                                            const cid: usize = switch (bspec.target) {
                                                .clip => |c| c,
                                                .dst => 0,
                                            };
                                            const f = src_frames[cid];
                                            const vf = zapi.vsapi.getVideoFrameFormat.?(f).*;
                                            const meta = common.clipMetaFromFrame(zapi, f, vf, 0, builtins.isRangeKind(bspec.kind)) orelse {
                                                common.setFilterErrorFmt(zapi, filter_name, "key '{s}': unsupported clip format for built-in constant '{s}'", .{ spec.key_z, @tagName(bspec.kind) });
                                                return null;
                                            };
                                            const bv = builtins.resolveMeta(bspec.kind, meta) catch {
                                                common.setFilterErrorFmt(zapi, filter_name, "key '{s}': built-in constant '{s}' is unsupported on >16-bit integer clips", .{ spec.key_z, @tagName(bspec.kind) });
                                                return null;
                                            };
                                            prop_vals[pi] = bv.f32Val();
                                        },
                                    }
                                }

                                const val = interp.evalScalar(allocator, &ce.prog, d.profile, .{
                                    .width = fw,
                                    .height = fh,
                                    .frame_n = @intCast(n),
                                    .props = prop_vals,
                                }) catch {
                                    zapi.setFilterError("PropExpr: out of memory");
                                    return null;
                                };
                                break :blk evalResultFromF32(val);
                            },
                        }
                    },
                };
            }

            // Phase 2: Copy src[0] and mutate its properties atomically.
            const dst = zapi.vsapi.copyFrame.?(src_frames[0], core) orelse {
                zapi.setFilterError("PropExpr: failed to copy frame");
                return null;
            };
            const dst_props = zapi.vsapi.getFramePropertiesRW.?(dst);
            for (d.specs, results) |*spec, res| {
                switch (res) {
                    .delete => _ = zapi.vsapi.mapDeleteKey.?(dst_props, spec.key_z.ptr),
                    .int_val => |iv| _ = zapi.vsapi.mapSetInt.?(dst_props, spec.key_z.ptr, iv, .Replace),
                    .float_val => |fv| _ = zapi.vsapi.mapSetFloat.?(dst_props, spec.key_z.ptr, fv, .Replace),
                }
            }
            return dst;
        },
        .Error => return null,
    }
}

fn propExprFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *PropExprData = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, null);
    d.deinit(zapi);
}

fn parseDataKeySpec(
    zapi: ZAPI,
    map: ?*const vs.Map,
    key_z_in: [*:0]const u8,
    nodes: []*vs.Node,
    profile: interp.TargetProfile,
    map_out: ZAPI.ZMap(?*vs.Map),
) !KeySpec {
    const nelems_raw = zapi.vsapi.mapNumElements.?(map, key_z_in);
    if (nelems_raw <= 0) {
        return common.fail(map_out, filter_name, "property '{s}' cannot be empty", .{std.mem.span(key_z_in)});
    }
    const nelems: usize = @intCast(nelems_raw);
    const key_z = try allocator.dupeSentinel(u8, std.mem.span(key_z_in), 0);
    errdefer allocator.free(key_z);

    const elems = try allocator.alloc(DataElem, nelems);
    var built: usize = 0;
    errdefer {
        for (elems[0..built]) |*e| {
            if (e.* == .expr) e.expr.deinit();
        }
        allocator.free(elems);
    }

    for (0..nelems) |ei| {
        const expr_str = common.getDataAllowEmpty(zapi, map, key_z_in, ei) orelse {
            return common.fail(map_out, filter_name, "failed to read string for property '{s}'", .{std.mem.span(key_z_in)});
        };
        if (expr_str.len == 0) {
            elems[ei] = .delete;
            built += 1;
            continue;
        }
        var prog = parser.parse(allocator, expr_str, .{
            .mode = .prop_expr,
            .num_clips = @intCast(nodes.len),
            .default_boundary = .clamp,
            .opt_int = false,
        }) catch |err| {
            return common.fail(map_out, filter_name, "parse error for property '{s}': {s}", .{ std.mem.span(key_z_in), @errorName(err) });
        };
        errdefer prog.deinit(allocator);

        // C4/F5: Reject >16-bit integer targets in Create when the targeted clip has a known constant format;
        // variable-format clips are checked per frame in `propExprGetFrame`.
        for (prog.builtins.items) |spec| {
            const cid: usize = switch (spec.target) {
                .clip => |c| c,
                .dst => 0,
            };
            const vf = zapi.getVideoInfo(nodes[cid]).format;
            if (vf.colorFamily != .Undefined) {
                if (common.colorFamilyFromVs(vf.colorFamily) == null) {
                    return common.fail(map_out, filter_name, "property '{s}': unsupported clip color family for built-in '{s}'", .{ std.mem.span(key_z_in), builtins.kindName(spec.kind) });
                }
                if (vf.sampleType == .Integer and (vf.bitsPerSample < 8 or vf.bitsPerSample > 16)) {
                    return common.fail(
                        map_out,
                        filter_name,
                        "property '{s}': built-in '{s}' is unsupported on {d}-bit integer clip",
                        .{ std.mem.span(key_z_in), builtins.kindName(spec.kind), vf.bitsPerSample },
                    );
                }
            }
        }

        try lower_mod.lower(allocator, &prog, profile);
        try optimize_mod.optimize(allocator, &prog, profile, .{
            .enable_value_changing = true,
            .enable_fma = false,
        });
        const bindings = try common.buildScalarPropBindings(allocator, &prog);
        elems[ei] = .{ .expr = .{ .prog = prog, .props = bindings } };
        built += 1;
    }

    return .{
        .key_z = key_z,
        .value = .{ .data_elems = elems },
    };
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
    createImpl(zapi, in, out, map_out) catch |err| switch (err) {
        error.Invalid => {},
        error.OutOfMemory => map_out.setError("PropExpr: out of memory"),
    };
}

fn createImpl(
    zapi: ZAPI,
    in: ?*const vs.Map,
    out: ?*vs.Map,
    map_out: ZAPI.ZMap(?*vs.Map),
) (common.ArgError || std.mem.Allocator.Error)!void {
    const nodes = common.getNodes(allocator, zapi, in, "clips") catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.TooManyClips => return common.fail(map_out, filter_name, "too many input clips (max 65535)", .{}),
        error.Missing => return common.fail(map_out, filter_name, "at least one clip is required", .{}),
    };
    var owns_nodes = true;
    errdefer if (owns_nodes) {
        for (nodes) |n| zapi.freeNode(n);
        allocator.free(nodes);
    };

    const clip_num_frames = try allocator.alloc(i32, nodes.len);
    var owns_nf = true;
    errdefer if (owns_nf) allocator.free(clip_num_frames);

    for (nodes, 0..) |node, i| {
        clip_num_frames[i] = zapi.getVideoInfo(node).numFrames;
    }

    const vi0 = zapi.getVideoInfo(nodes[0]);
    const has_dict = zapi.vsapi.mapGetType.?(in, "dict") != .Unset;
    const in_num_keys_raw = zapi.vsapi.mapNumKeys.?(in);
    const in_num_keys: usize = if (in_num_keys_raw > 0) @intCast(in_num_keys_raw) else 0;

    var extra_kw_count: usize = 0;
    for (0..in_num_keys) |ki| {
        const k = zapi.vsapi.mapGetKey.?(in, @intCast(ki)).?;
        const ks = std.mem.span(k);
        if (!std.mem.eql(u8, ks, "clips") and !std.mem.eql(u8, ks, "dict")) {
            extra_kw_count += 1;
        }
    }

    if (has_dict and extra_kw_count > 0) {
        return common.fail(map_out, filter_name, "cannot specify both dict and keyword arguments", .{});
    }
    if (!has_dict and extra_kw_count == 0) {
        return common.fail(map_out, filter_name, "at least one property expression (via dict or keyword arguments) is required", .{});
    }

    const profile: interp.TargetProfile = .{
        .use_fma = false,
        .isa_family = cpu_mod.IsaFamily.host(),
    };

    var specs_list: std.ArrayList(KeySpec) = .empty;
    var owns_specs = true;
    errdefer if (owns_specs) {
        for (specs_list.items) |*s| s.deinit();
        specs_list.deinit(allocator);
    };

    if (has_dict) {
        // Akarin style: call `dict` function, freeing function + both temporary maps on every exit (Y20).
        var err: vs.MapPropertyError = .Success;
        const func = zapi.vsapi.mapGetFunction.?(in, "dict", 0, &err) orelse {
            return common.fail(map_out, filter_name, "dict must be a callable function", .{});
        };
        defer zapi.vsapi.freeFunction.?(func);

        const call_in = zapi.vsapi.createMap.?() orelse return error.OutOfMemory;
        defer zapi.vsapi.freeMap.?(call_in);

        const call_out = zapi.vsapi.createMap.?() orelse return error.OutOfMemory;
        defer zapi.vsapi.freeMap.?(call_out);

        zapi.vsapi.callFunction.?(func, call_in, call_out);
        if (zapi.vsapi.mapGetError.?(call_out)) |fn_err| {
            return common.fail(map_out, filter_name, "dict callback error: {s}", .{std.mem.span(fn_err)});
        }

        const nkeys_raw = zapi.vsapi.mapNumKeys.?(call_out);
        if (nkeys_raw <= 0) {
            return common.fail(map_out, filter_name, "dict callback must return at least one property", .{});
        }
        const nkeys: usize = @intCast(nkeys_raw);

        for (0..nkeys) |ki| {
            const k = zapi.vsapi.mapGetKey.?(call_out, @intCast(ki)).?;
            const ptype = zapi.vsapi.mapGetType.?(call_out, k);
            const nelems_raw = zapi.vsapi.mapNumElements.?(call_out, k);
            if (nelems_raw <= 0) {
                return common.fail(map_out, filter_name, "property '{s}' has empty value", .{std.mem.span(k)});
            }
            const nelems: usize = @intCast(nelems_raw);

            switch (ptype) {
                .Int => {
                    // Y18: Exact i64 constant pass-through!
                    const key_z = try allocator.dupeSentinel(u8, std.mem.span(k), 0);
                    errdefer allocator.free(key_z);
                    const vals = try allocator.alloc(i64, nelems);
                    errdefer allocator.free(vals);
                    for (0..nelems) |ei| {
                        var e2: vs.MapPropertyError = .Success;
                        vals[ei] = zapi.vsapi.mapGetInt.?(call_out, k, @intCast(ei), &e2);
                    }
                    try specs_list.append(allocator, .{ .key_z = key_z, .value = .{ .const_ints = vals } });
                },
                .Float => {
                    // Y18: Exact f64 constant pass-through!
                    const key_z = try allocator.dupeSentinel(u8, std.mem.span(k), 0);
                    errdefer allocator.free(key_z);
                    const vals = try allocator.alloc(f64, nelems);
                    errdefer allocator.free(vals);
                    for (0..nelems) |ei| {
                        var e2: vs.MapPropertyError = .Success;
                        vals[ei] = zapi.vsapi.mapGetFloat.?(call_out, k, @intCast(ei), &e2);
                    }
                    try specs_list.append(allocator, .{ .key_z = key_z, .value = .{ .const_floats = vals } });
                },
                .Data => {
                    var spec = try parseDataKeySpec(zapi, call_out, k, nodes, profile, map_out);
                    errdefer spec.deinit();
                    try specs_list.append(allocator, spec);
                },
                else => {
                    return common.fail(map_out, filter_name, "unsupported value type for property '{s}' (expected int, float, or string)", .{std.mem.span(k)});
                },
            }
        }
    } else {
        // Cranexpr style: extra keyword arguments in `in`, requiring string values.
        for (0..in_num_keys) |ki| {
            const k = zapi.vsapi.mapGetKey.?(in, @intCast(ki)).?;
            const ks = std.mem.span(k);
            if (std.mem.eql(u8, ks, "clips") or std.mem.eql(u8, ks, "dict")) continue;
            if (zapi.vsapi.mapGetType.?(in, k) != .Data) {
                return common.fail(map_out, filter_name, "keyword argument '{s}' must be a string expression", .{ks});
            }
            var spec = try parseDataKeySpec(zapi, in, k, nodes, profile, map_out);
            errdefer spec.deinit();
            try specs_list.append(allocator, spec);
        }
    }

    const owned_specs = try specs_list.toOwnedSlice(allocator);
    owns_specs = false;
    var owns_owned_specs = true;
    errdefer if (owns_owned_specs) {
        for (owned_specs) |*s| s.deinit();
        allocator.free(owned_specs);
    };

    const deps = try allocator.alloc(vs.FilterDependency, nodes.len);
    defer allocator.free(deps);

    const data = try allocator.create(PropExprData);
    owns_nodes = false;
    owns_nf = false;
    owns_owned_specs = false;
    data.* = .{
        .nodes = nodes,
        .clip_num_frames = clip_num_frames,
        .out_vi = vi0.*,
        .profile = profile,
        .specs = owned_specs,
    };

    common.fillDeps(deps, nodes, clip_num_frames, data.out_vi.numFrames);

    zapi.createVideoFilter(out, filter_name, &data.out_vi, propExprGetFrame, propExprFree, .Parallel, deps, data);
}
