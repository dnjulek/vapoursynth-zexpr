//! `zexpr.Expr` VapourSynth filter adapter.

const std = @import("std");
const vapoursynth = @import("vapoursynth");
const ast = @import("../expr/ast.zig");
const parser = @import("../expr/parser.zig");
const lower_mod = @import("../expr/lower.zig");
const optimize_mod = @import("../expr/optimize.zig");
const analysis_mod = @import("../expr/analysis.zig");
const interp = @import("../expr/interp.zig");
const aj = @import("../jit/asmjit.zig");
const cpu_mod = @import("../jit/cpu.zig");
const codegen = @import("../jit/codegen.zig");
const lut_mod = @import("../jit/lut.zig");
const plane_cache = @import("../jit/plane_cache.zig");
const common = @import("common.zig");

const vs = vapoursynth.vapoursynth4;
const vsh = vapoursynth.vshelper;
const ZAPI = vapoursynth.ZAPI;

const allocator = std.heap.c_allocator;
pub const filter_name = "Expr";
pub const params = "clips:vnode[];expr:data[];format:int:opt;opt:int:opt;boundary:int:opt;";

const max_planes = 3;

var global_plane_cache: plane_cache.PlaneCache(plane_cache.default_capacity) = .{};

const PlaneExec = union(enum) {
    copy,
    undefined_plane,
    cached: *plane_cache.Entry,
    lut_unshared: lut_mod.Lut,
    scalar: struct {
        prog: ast.Program,
        props: []common.PropBinding,
    },

    fn deinit(self: *PlaneExec) void {
        switch (self.*) {
            .copy, .undefined_plane => {},
            .cached => |entry| global_plane_cache.release(allocator, entry),
            .lut_unshared => |*l| l.deinit(allocator, null),
            .scalar => |*s| {
                s.prog.deinit(allocator);
                common.freePropBindings(allocator, s.props);
            },
        }
        self.* = .copy;
    }
};

const ExprData = struct {
    nodes: []*vs.Node,
    clip_num_frames: []i32,
    src_fmts: []ast.SampleFormat,
    out_vi: vs.VideoInfo,
    dst_fmt: ast.SampleFormat,
    profile: interp.TargetProfile,
    planes: [max_planes]PlaneExec,
    num_planes: usize,

    fn deinit(self: *ExprData, zapi: ZAPI) void {
        for (self.planes[0..self.num_planes]) |*pe| {
            pe.deinit();
        }
        for (self.nodes) |n| zapi.freeNode(n);
        allocator.free(self.nodes);
        allocator.free(self.clip_num_frames);
        allocator.free(self.src_fmts);
        allocator.destroy(self);
    }
};

fn exprGetFrame(
    n: c_int,
    reason: vs.ActivationReason,
    instance_data: ?*anyopaque,
    frame_data: ?*?*anyopaque,
    frame_ctx: ?*vs.FrameContext,
    core: ?*vs.Core,
    vsapi: ?*const vs.API,
) callconv(.c) ?*const vs.Frame {
    _ = frame_data;
    const d: *const ExprData = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, frame_ctx);

    switch (reason) {
        .Initial => {
            for (d.nodes, d.clip_num_frames) |node, nf| {
                zapi.requestFrameFilter(common.clampFrameIdx(n, nf), node);
            }
            return null;
        },
        .AllFramesReady => {
            const n_clips = d.nodes.len;
            var stack_frames: [32]?*const vs.Frame = @splat(null);
            const heap_frames: ?[]?*const vs.Frame = if (n_clips > stack_frames.len)
                allocator.alloc(?*const vs.Frame, n_clips) catch {
                    zapi.setFilterError("Expr: out of memory");
                    return null;
                }
            else
                null;
            defer if (heap_frames) |hf| allocator.free(hf);
            const src_frames: []?*const vs.Frame = if (heap_frames) |hf| hf else stack_frames[0..n_clips];
            @memset(src_frames, null);

            defer {
                for (src_frames) |maybe_f| {
                    if (maybe_f) |f| zapi.freeFrame(f);
                }
            }

            for (d.nodes, d.clip_num_frames, 0..) |node, nf, i| {
                src_frames[i] = zapi.getFrameFilter(common.clampFrameIdx(n, nf), node);
            }

            // Build dst frame using newVideoFrame2: copy planes reference src_frames[0], others null.
            var plane_srcs: [max_planes]?*const vs.Frame = .{ null, null, null };
            const plane_indices = [max_planes]c_int{ 0, 1, 2 };
            for (0..d.num_planes) |p| {
                if (d.planes[p] == .copy) {
                    plane_srcs[p] = src_frames[0];
                }
            }
            const dst_frame = zapi.vsapi.newVideoFrame2.?(
                &d.out_vi.format,
                d.out_vi.width,
                d.out_vi.height,
                &plane_srcs,
                &plane_indices,
                src_frames[0],
                core,
            ) orelse {
                zapi.setFilterError("Expr: failed to allocate output frame");
                return null;
            };

            var stack_ptrs: [32][*]const u8 = undefined;
            var stack_strides: [32]isize = undefined;
            const heap_ptrs: ?[][*]const u8 = if (n_clips > stack_ptrs.len)
                allocator.alloc([*]const u8, n_clips) catch {
                    zapi.freeFrame(dst_frame);
                    zapi.setFilterError("Expr: out of memory");
                    return null;
                }
            else
                null;
            defer if (heap_ptrs) |hp| allocator.free(hp);
            const heap_strides: ?[]isize = if (n_clips > stack_strides.len)
                allocator.alloc(isize, n_clips) catch {
                    zapi.freeFrame(dst_frame);
                    zapi.setFilterError("Expr: out of memory");
                    return null;
                }
            else
                null;
            defer if (heap_strides) |hs| allocator.free(hs);

            const src_ptrs: [][*]const u8 = if (heap_ptrs) |hp| hp else stack_ptrs[0..n_clips];
            const src_strides: []isize = if (heap_strides) |hs| hs else stack_strides[0..n_clips];

            for (0..d.num_planes) |p| {
                switch (d.planes[p]) {
                    .copy, .undefined_plane => continue,
                    .cached => |entry| switch (entry.exec) {
                        .lut => |*l| {
                            const dst_ptr = zapi.vsapi.getWritePtr.?(dst_frame, @intCast(p)).?;
                            const dst_stride: isize = zapi.vsapi.getStride.?(dst_frame, @intCast(p));
                            const pw: usize = @intCast(zapi.vsapi.getFrameWidth.?(dst_frame, @intCast(p)));
                            const ph: usize = @intCast(zapi.vsapi.getFrameHeight.?(dst_frame, @intCast(p)));
                            for (0..n_clips) |ci| {
                                src_ptrs[ci] = zapi.vsapi.getReadPtr.?(src_frames[ci], @intCast(p)).?;
                                src_strides[ci] = zapi.vsapi.getStride.?(src_frames[ci], @intCast(p));
                            }
                            l.apply(dst_ptr, dst_stride, src_ptrs, src_strides, pw, ph);
                        },
                        .jit => |j| {
                            const dst_ptr = zapi.vsapi.getWritePtr.?(dst_frame, @intCast(p)).?;
                            const dst_stride: isize = zapi.vsapi.getStride.?(dst_frame, @intCast(p));
                            for (0..n_clips) |ci| {
                                src_ptrs[ci] = zapi.vsapi.getReadPtr.?(src_frames[ci], @intCast(p)).?;
                                src_strides[ci] = zapi.vsapi.getStride.?(src_frames[ci], @intCast(p));
                            }
                            var prop_stack: [64]f32 = undefined;
                            const prop_heap: ?[]f32 = if (j.props.len > prop_stack.len)
                                allocator.alloc(f32, j.props.len) catch {
                                    zapi.freeFrame(dst_frame);
                                    zapi.setFilterError("Expr: out of memory");
                                    return null;
                                }
                            else
                                null;
                            defer if (prop_heap) |ph| allocator.free(ph);
                            const prop_vals: []f32 = if (prop_heap) |ph| ph else prop_stack[0..j.props.len];

                            for (j.props, 0..) |pb, pi| {
                                const fprops = zapi.vsapi.getFramePropertiesRO.?(src_frames[pb.clip]);
                                prop_vals[pi] = common.readFrameProp(zapi, fprops, pb.name_z.ptr, std.math.nan(f32));
                            }

                            j.kernel.func(
                                dst_ptr,
                                dst_stride,
                                src_ptrs.ptr,
                                src_strides.ptr,
                                @intCast(n),
                                if (prop_vals.len > 0) prop_vals.ptr else null,
                            );
                        },
                    },
                    .lut_unshared => |*l| {
                        const dst_ptr = zapi.vsapi.getWritePtr.?(dst_frame, @intCast(p)).?;
                        const dst_stride: isize = zapi.vsapi.getStride.?(dst_frame, @intCast(p));
                        const pw: usize = @intCast(zapi.vsapi.getFrameWidth.?(dst_frame, @intCast(p)));
                        const ph: usize = @intCast(zapi.vsapi.getFrameHeight.?(dst_frame, @intCast(p)));
                        for (0..n_clips) |ci| {
                            src_ptrs[ci] = zapi.vsapi.getReadPtr.?(src_frames[ci], @intCast(p)).?;
                            src_strides[ci] = zapi.vsapi.getStride.?(src_frames[ci], @intCast(p));
                        }
                        l.apply(dst_ptr, dst_stride, src_ptrs, src_strides, pw, ph);
                    },
                    .scalar => |*s| {
                        const dst_ptr = zapi.vsapi.getWritePtr.?(dst_frame, @intCast(p)).?;
                        const dst_stride: isize = zapi.vsapi.getStride.?(dst_frame, @intCast(p));
                        const pw: usize = @intCast(zapi.vsapi.getFrameWidth.?(dst_frame, @intCast(p)));
                        const ph: usize = @intCast(zapi.vsapi.getFrameHeight.?(dst_frame, @intCast(p)));
                        for (0..n_clips) |ci| {
                            src_ptrs[ci] = zapi.vsapi.getReadPtr.?(src_frames[ci], @intCast(p)).?;
                            src_strides[ci] = zapi.vsapi.getStride.?(src_frames[ci], @intCast(p));
                        }
                        var prop_stack: [64]f32 = undefined;
                        const prop_heap: ?[]f32 = if (s.props.len > prop_stack.len)
                            allocator.alloc(f32, s.props.len) catch {
                                zapi.freeFrame(dst_frame);
                                zapi.setFilterError("Expr: out of memory");
                                return null;
                            }
                        else
                            null;
                        defer if (prop_heap) |ph_buf| allocator.free(ph_buf);
                        const prop_vals: []f32 = if (prop_heap) |ph_buf| ph_buf else prop_stack[0..s.props.len];

                        for (s.props, 0..) |pb, pi| {
                            const fprops = zapi.vsapi.getFramePropertiesRO.?(src_frames[pb.clip]);
                            prop_vals[pi] = common.readFrameProp(zapi, fprops, pb.name_z.ptr, std.math.nan(f32));
                        }

                        interp.evalPlane(allocator, &s.prog, d.profile, .{
                            .dst = dst_ptr,
                            .dst_stride = dst_stride,
                            .dst_fmt = d.dst_fmt,
                            .srcs = src_ptrs,
                            .src_strides = src_strides,
                            .src_fmts = d.src_fmts,
                            .width = pw,
                            .height = ph,
                            .frame_n = @intCast(n),
                            .props = prop_vals,
                        }) catch {
                            zapi.freeFrame(dst_frame);
                            zapi.setFilterError("Expr: out of memory");
                            return null;
                        };
                    },
                }
            }

            return dst_frame;
        },
        .Error => return null,
    }
}

fn exprFree(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    const d: *ExprData = @ptrCast(@alignCast(instance_data));
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
    createImpl(zapi, in, out, map_out, core, vsapi) catch |err| switch (err) {
        error.Invalid => {},
        error.OutOfMemory => map_out.setError("Expr: out of memory"),
    };
}

fn createImpl(
    zapi: ZAPI,
    in: ?*const vs.Map,
    out: ?*vs.Map,
    map_out: ZAPI.ZMap(?*vs.Map),
    core: ?*vs.Core,
    vsapi: ?*const vs.API,
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

    const vi0 = zapi.getVideoInfo(nodes[0]);
    if (!vsh.isConstantVideoFormat(vi0)) {
        return common.fail(map_out, filter_name, "only constant format input is supported", .{});
    }

    const src_fmts = try allocator.alloc(ast.SampleFormat, nodes.len);
    var owns_src_fmts = true;
    errdefer if (owns_src_fmts) allocator.free(src_fmts);

    const clip_num_frames = try allocator.alloc(i32, nodes.len);
    var owns_clip_nf = true;
    errdefer if (owns_clip_nf) allocator.free(clip_num_frames);

    for (nodes, 0..) |node, i| {
        const vi = zapi.getVideoInfo(node);
        if (!vsh.isConstantVideoFormat(vi)) {
            return common.fail(map_out, filter_name, "clip {d} must have constant format and dimensions", .{i});
        }
        const sf = common.toSampleFormat(vi.format) orelse {
            return common.fail(map_out, filter_name, "clip {d} has unsupported sample format ({d}-bit)", .{ i, vi.format.bitsPerSample });
        };
        if (vi.width != vi0.width or vi.height != vi0.height or
            vi.format.numPlanes != vi0.format.numPlanes or
            vi.format.subSamplingW != vi0.format.subSamplingW or
            vi.format.subSamplingH != vi0.format.subSamplingH)
        {
            return common.fail(map_out, filter_name, "clip {d} dimensions, plane count, or subsampling do not match clip 0", .{i});
        }
        src_fmts[i] = sf;
        clip_num_frames[i] = vi.numFrames;
    }

    // Validate optional `format=` override (D14).
    var out_vi = vi0.*;
    if (common.getOptInt64(zapi, in, "format")) |fmt_id_64| {
        if (fmt_id_64 < 0 or fmt_id_64 > std.math.maxInt(u32)) {
            return common.fail(map_out, filter_name, "invalid format id", .{});
        }
        var new_fmt: vs.VideoFormat = undefined;
        const fmt_enum: vs.PresetVideoFormat = @fromBackingInt(@intCast(@as(u32, @intCast(fmt_id_64))));
        if (vsapi.?.getVideoFormatByID.?(&new_fmt, fmt_enum, core) == 0 or new_fmt.colorFamily == .Undefined) {
            return common.fail(map_out, filter_name, "invalid format id", .{});
        }
        if (common.toSampleFormat(new_fmt) == null) {
            return common.fail(map_out, filter_name, "unsupported output format ({d}-bit)", .{new_fmt.bitsPerSample});
        }
        if (new_fmt.colorFamily != vi0.format.colorFamily or
            new_fmt.numPlanes != vi0.format.numPlanes or
            new_fmt.subSamplingW != vi0.format.subSamplingW or
            new_fmt.subSamplingH != vi0.format.subSamplingH)
        {
            return common.fail(map_out, filter_name, "output format must match input color family, plane count, and subsampling", .{});
        }
        out_vi.format = new_fmt;
    }

    const dst_fmt = common.toSampleFormat(out_vi.format) orelse {
        return common.fail(map_out, filter_name, "unsupported output sample format", .{});
    };

    // Validate `opt` and `boundary`.
    const opt_val = common.getOptInt64(zapi, in, "opt") orelse 0;
    if (opt_val != 0 and opt_val != 1) {
        return common.fail(map_out, filter_name, "opt must be 0 or 1", .{});
    }
    const boundary_val = common.getOptInt64(zapi, in, "boundary") orelse 0;
    if (boundary_val != 0 and boundary_val != 1) {
        return common.fail(map_out, filter_name, "boundary must be 0 (clamp) or 1 (mirror)", .{});
    }
    const default_boundary: ast.BoundaryMode = if (boundary_val == 1) .mirror else .clamp;

    // Read `expr[]` array (allowing empty strings!).
    const num_planes: usize = @intCast(out_vi.format.numPlanes);
    const expr_count_raw = vsapi.?.mapNumElements.?(in, "expr");
    if (expr_count_raw <= 0 or expr_count_raw > out_vi.format.numPlanes) {
        return common.fail(map_out, filter_name, "expr must have between 1 and {d} elements", .{num_planes});
    }
    const expr_count: usize = @intCast(expr_count_raw);

    var plane_exprs: [max_planes][]const u8 = .{ "", "", "" };
    for (0..num_planes) |p| {
        const src_idx = if (p < expr_count) p else expr_count - 1;
        plane_exprs[p] = common.getDataAllowEmpty(zapi, in, "expr", src_idx) orelse {
            return common.fail(map_out, filter_name, "failed to read expr[{d}]", .{src_idx});
        };
    }

    // Detect host CPU + probe core alignment (S5, L4).
    const env_override = cpu_mod.readEnvOverride();
    var cpu = cpu_mod.detectWithOverride(env_override);
    const core_align = common.probeCoreAlignment(zapi);
    if (cpu.vec_bytes > core_align) {
        // Never assume more alignment than the host core provides.
        if (core_align >= 32) {
            cpu = cpu_mod.detectWithOverride(3);
        } else if (core_align >= 16) {
            cpu = cpu_mod.detectWithOverride(2);
        } else {
            cpu = cpu_mod.detectWithOverride(0);
        }
    }

    // Oracle profile: when forced via ZEXPR_CPU_LEVEL=0, use_fma = false; otherwise match host_cpu.has_fma.
    const profile: interp.TargetProfile = if (env_override != null and env_override.? == 0)
        .{ .use_fma = false, .isa_family = cpu_mod.IsaFamily.host() }
    else if (cpu.isa == .scalar)
        .{ .use_fma = cpu_mod.detectWithOverride(null).has_fma, .isa_family = cpu_mod.IsaFamily.host() }
    else
        interp.TargetProfile.fromCpu(cpu);

    const force_scalar = env_override != null and env_override.? == 0;

    var planes: [max_planes]PlaneExec = .{ .copy, .copy, .copy };
    var owns_planes = true;
    errdefer if (owns_planes) {
        for (planes[0..num_planes]) |*pe| {
            pe.deinit();
        }
    };

    var plane_keys: [max_planes]plane_cache.Key = undefined;
    var needs_compile: [max_planes]bool = .{ false, false, false };
    var any_miss = false;

    const same_sample_as_src0 =
        out_vi.format.sampleType == vi0.format.sampleType and
        out_vi.format.bitsPerSample == vi0.format.bitsPerSample;

    // Phase 1: check LRU cache for each non-empty plane (instant hit in FrameEval / shared U/V).
    for (0..num_planes) |p| {
        const expr_str = plane_exprs[p];
        if (expr_str.len == 0) {
            planes[p] = if (same_sample_as_src0) .copy else .undefined_plane;
            continue;
        }
        const pw: usize = @intCast(if (p == 0) out_vi.width else out_vi.width >> @intCast(out_vi.format.subSamplingW));
        const ph: usize = @intCast(if (p == 0) out_vi.height else out_vi.height >> @intCast(out_vi.format.subSamplingH));
        plane_keys[p] = .{
            .expr = expr_str,
            .src_fmts = src_fmts,
            .dst_fmt = dst_fmt,
            .width = pw,
            .height = ph,
            .opt_int = opt_val == 1,
            .default_boundary = default_boundary,
            .cpu = cpu,
            .profile = profile,
        };
        if (!force_scalar) {
            if (global_plane_cache.get(plane_keys[p])) |entry| {
                planes[p] = .{ .cached = entry };
                continue;
            }
        }
        needs_compile[p] = true;
        any_miss = true;
    }

    // Phase 2: parse all missed planes before compiling any, so syntax errors on plane 1/2
    // fail without wasting JIT compilation or evicting live cache entries.
    var parsed_progs: [max_planes]?ast.Program = .{ null, null, null };
    defer {
        for (&parsed_progs) |*mp| {
            if (mp.*) |*pr| pr.deinit(allocator);
        }
    }
    for (0..num_planes) |p| {
        if (!needs_compile[p]) continue;
        parsed_progs[p] = parser.parse(allocator, plane_exprs[p], .{
            .mode = .expr,
            .num_clips = @intCast(nodes.len),
            .default_boundary = default_boundary,
            .opt_int = opt_val == 1,
        }) catch |err| {
            return common.fail(map_out, filter_name, "parse error on plane {d}: {s}", .{ p, @errorName(err) });
        };
    }

    // Phase 3: compile missed planes against the shared JitRuntime and populate the LRU cache.
    const rt_ptr: ?*aj.Runtime = if (any_miss and cpu.isa != .scalar) global_plane_cache.ensureRuntime() else null;
    var execmem_warned = false;

    for (0..num_planes) |p| {
        if (!needs_compile[p]) continue;

        // Re-check cache in case an earlier plane in this same Create compiled an identical key
        // (e.g., YUV420 U and V planes sharing the same expression and dimensions).
        if (!force_scalar) {
            if (global_plane_cache.get(plane_keys[p])) |entry| {
                planes[p] = .{ .cached = entry };
                continue;
            }
        }

        var prog = parsed_progs[p].?;
        parsed_progs[p] = null;
        errdefer prog.deinit(allocator);

        try lower_mod.lower(allocator, &prog, profile);
        try optimize_mod.optimize(allocator, &prog, profile, .{
            .enable_value_changing = true,
            .enable_fma = profile.use_fma,
        });

        const pw = plane_keys[p].width;
        const ph = plane_keys[p].height;

        const has_fast_u8_lut = rt_ptr != null and aj.Builder.cpuSupportsFastLutU8(cpu);
        var info = try analysis_mod.analyze(allocator, &prog, src_fmts, dst_fmt, @intCast(pw), @intCast(ph), has_fast_u8_lut);
        defer info.deinit(allocator);

        // Automatic 1D/2D LUT fast path (B4, Y14) unless forced to scalar via ZEXPR_CPU_LEVEL=0.
        if (!force_scalar and info.lut_kind != .none) {
            var lut_execmem_denied = false;
            const lut = try lut_mod.Lut.build(
                allocator,
                rt_ptr,
                cpu,
                profile,
                &prog,
                &info,
                src_fmts,
                dst_fmt,
                pw,
                ph,
                &lut_execmem_denied,
            );
            prog.deinit(allocator);
            if (lut_execmem_denied) {
                if (!execmem_warned) {
                    execmem_warned = true;
                    zapi.logMessage(.Warning, "zexpr.Expr: executable memory allocation denied; falling back to scalar LUT");
                }
                planes[p] = .{ .lut_unshared = lut };
            } else {
                const entry = cache_blk: {
                    var exec: plane_cache.CompiledPlane = .{ .lut = lut };
                    errdefer exec.deinit(allocator, rt_ptr);
                    break :cache_blk try global_plane_cache.putOrGet(allocator, plane_keys[p], exec);
                };
                planes[p] = .{ .cached = entry };
            }
            continue;
        }

        const prop_bindings = try common.buildPropBindings(allocator, &prog);
        var owns_props = true;
        errdefer if (owns_props) common.freePropBindings(allocator, prop_bindings);

        // Try JIT when enabled and supported for this plane.
        var compiled_kernel: ?aj.CompiledKernel = null;
        var jit_execmem_denied = false;
        if (rt_ptr) |runtime| {
            if (codegen.compilePlane(allocator, runtime, cpu, &prog, &info, .{
                .width = pw,
                .height = ph,
                .src_fmts = src_fmts,
                .dst_fmt = dst_fmt,
            })) |res| {
                compiled_kernel = res.kernel;
            } else |err| switch (err) {
                error.ExecMemDenied => {
                    jit_execmem_denied = true;
                    if (!execmem_warned) {
                        execmem_warned = true;
                        const msg = if (!force_scalar and info.fallback_lut_kind != .none)
                            "zexpr.Expr: executable memory allocation denied; falling back to scalar LUT"
                        else
                            "zexpr.Expr: executable memory allocation denied; falling back to scalar interpreter";
                        zapi.logMessage(.Warning, msg);
                    }
                },
                error.UnsupportedFeature => {},
                error.OutOfMemory => return error.OutOfMemory,
                error.AbiMismatch, error.CompileFailed => {
                    return common.fail(map_out, filter_name, "JIT compilation failed on plane {d}: {s}", .{ p, @errorName(err) });
                },
            }
        }

        if (compiled_kernel) |kernel| {
            prog.deinit(allocator);
            owns_props = false;
            const entry = cache_blk: {
                var exec: plane_cache.CompiledPlane = .{ .jit = .{ .kernel = kernel, .props = prop_bindings } };
                errdefer exec.deinit(allocator, rt_ptr);
                break :cache_blk try global_plane_cache.putOrGet(allocator, plane_keys[p], exec);
            };
            planes[p] = .{ .cached = entry };
        } else if (!force_scalar and (rt_ptr == null or jit_execmem_denied) and info.fallback_lut_kind != .none) {
            // M1: when executable memory is unavailable and the plane can be served by a scalar table,
            // build the table (via scalar oracle, no JIT) instead of dropping to the per-pixel interpreter.
            info.lut_kind = info.fallback_lut_kind;
            info.lut_clips = info.fallback_lut_clips;
            const lut = try lut_mod.Lut.build(
                allocator,
                null,
                cpu,
                profile,
                &prog,
                &info,
                src_fmts,
                dst_fmt,
                pw,
                ph,
                null,
            );
            prog.deinit(allocator);
            common.freePropBindings(allocator, prop_bindings);
            owns_props = false;
            planes[p] = .{ .lut_unshared = lut };
        } else {
            owns_props = false;
            planes[p] = .{ .scalar = .{ .prog = prog, .props = prop_bindings } };
        }
    }

    const deps = try allocator.alloc(vs.FilterDependency, nodes.len);
    defer allocator.free(deps);

    const data = try allocator.create(ExprData);
    owns_nodes = false;
    owns_src_fmts = false;
    owns_clip_nf = false;
    owns_planes = false;
    data.* = .{
        .nodes = nodes,
        .clip_num_frames = clip_num_frames,
        .src_fmts = src_fmts,
        .out_vi = out_vi,
        .dst_fmt = dst_fmt,
        .profile = profile,
        .planes = planes,
        .num_planes = num_planes,
    };

    for (nodes, clip_num_frames, 0..) |node, nf, i| {
        deps[i] = .{
            .source = node,
            .requestPattern = if (nf >= out_vi.numFrames) .StrictSpatial else .FrameReuseLastOnly,
        };
    }

    zapi.createVideoFilter(out, filter_name, &data.out_vi, exprGetFrame, exprFree, .Parallel, deps, data);
}
