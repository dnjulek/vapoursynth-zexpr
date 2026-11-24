//! `zexpr.Expr` VapourSynth filter adapter.

const std = @import("std");
const vapoursynth = @import("vapoursynth");
const ast = @import("../expr/ast.zig");
const builtins = @import("../expr/builtins.zig");
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

const PlaneCacheType = plane_cache.PlaneCache(plane_cache.default_capacity);
var global_plane_cache: PlaneCacheType = .{};

pub fn registerTeardownHook(cb: aj.DsoTeardownFn) void {
    aj.setDsoTeardown(cb);
}

pub fn shutdownCache() void {
    const drained = global_plane_cache.shutdown(allocator);
    common.resetCoreAlignmentCache();
    if (drained) {
        aj.emitShutdownMarkerIfEnabled();
    }
}

const SinglePlaneExec = plane_cache.SinglePlaneExec(PlaneCacheType);
const RangeVariants = plane_cache.RangeVariants(PlaneCacheType, vs.VideoFormat);

const PlaneExec = union(enum) {
    single: SinglePlaneExec,
    range_variants: *RangeVariants,

    fn deinit(self: *PlaneExec) void {
        switch (self.*) {
            .single => |*s| s.deinit(allocator, &global_plane_cache),
            .range_variants => |rv| {
                rv.deinit(allocator, &global_plane_cache);
                allocator.destroy(rv);
            },
        }
        self.* = .{ .single = .copy };
    }
};

const StaticData = struct {
    src_fmts: []ast.SampleFormat,
    dst_fmt: ast.SampleFormat,
    planes: [max_planes]PlaneExec,
    num_planes: usize,

    fn deinit(self: *StaticData) void {
        for (self.planes[0..self.num_planes]) |*pe| {
            pe.deinit();
        }
        allocator.free(self.src_fmts);
    }
};

/// Phase C dynamic mode: serves variable-format clips, variable-resolution clips, and
/// constant-format expressions depending on `> 2` distinct integer clips' `_Range`.
const DynamicData = struct {
    plane_exprs: [max_planes][]u8,
    plane_builtins: [max_planes][]builtins.BuiltinSpec,
    has_format_override: bool,
    override_fmt: vs.VideoFormat,
    opt_int: bool,
    default_boundary: ast.BoundaryMode,
    cpu: cpu_mod.CpuInfo,
    force_scalar: bool,
    execmem_warned: std.atomic.Value(bool),

    fn deinit(self: *DynamicData) void {
        for (0..max_planes) |p| {
            if (self.plane_exprs[p].len > 0) allocator.free(self.plane_exprs[p]);
            if (self.plane_builtins[p].len > 0) allocator.free(self.plane_builtins[p]);
        }
    }
};

const ExprModeData = union(enum) {
    static: StaticData,
    dynamic: DynamicData,
};

const ExprData = struct {
    nodes: []*vs.Node,
    clip_num_frames: []i32,
    out_vi: vs.VideoInfo,
    profile: interp.TargetProfile,
    mode: ExprModeData,

    fn deinit(self: *ExprData, zapi: ZAPI) void {
        switch (self.mode) {
            .static => |*s| s.deinit(),
            .dynamic => |*dyn| dyn.deinit(),
        }
        for (self.nodes) |n| zapi.freeNode(n);
        allocator.free(self.nodes);
        allocator.free(self.clip_num_frames);
        allocator.destroy(self);
    }
};

fn setFilterErrorFmt(zapi: ZAPI, comptime fmt: []const u8, args: anytype) void {
    var buf: [512]u8 = undefined;
    var w: std.Io.Writer = .fixed(buf[0 .. buf.len - 1]);
    w.print(filter_name ++ ": " ++ fmt, args) catch {};
    buf[w.end] = 0;
    zapi.setFilterError(buf[0..w.end :0]);
}

fn executeScalarProg(
    zapi: ZAPI,
    prog: *const ast.Program,
    props: []const common.PropBinding,
    dst_frame: *vs.Frame,
    src_frames: []const ?*const vs.Frame,
    src_ptrs: [][*]const u8,
    src_strides: []isize,
    src_fmts: []const ast.SampleFormat,
    dst_fmt: ast.SampleFormat,
    profile: interp.TargetProfile,
    plane_idx: usize,
    frame_n: c_int,
) bool {
    const n_clips = src_frames.len;
    const dst_ptr = zapi.vsapi.getWritePtr.?(dst_frame, @intCast(plane_idx)).?;
    const dst_stride: isize = zapi.vsapi.getStride.?(dst_frame, @intCast(plane_idx));
    const pw: usize = @intCast(zapi.vsapi.getFrameWidth.?(dst_frame, @intCast(plane_idx)));
    const ph: usize = @intCast(zapi.vsapi.getFrameHeight.?(dst_frame, @intCast(plane_idx)));
    for (0..n_clips) |ci| {
        src_ptrs[ci] = zapi.vsapi.getReadPtr.?(src_frames[ci], @intCast(plane_idx)).?;
        src_strides[ci] = zapi.vsapi.getStride.?(src_frames[ci], @intCast(plane_idx));
    }

    var prop_stack: [64]f32 = undefined;
    const prop_heap: ?[]f32 = if (props.len > prop_stack.len)
        allocator.alloc(f32, props.len) catch {
            zapi.setFilterError("Expr: out of memory");
            return false;
        }
    else
        null;
    defer if (prop_heap) |ph_buf| allocator.free(ph_buf);
    const prop_vals: []f32 = if (prop_heap) |ph_buf| ph_buf else prop_stack[0..props.len];

    for (props, 0..) |pb, pi| {
        const fprops = zapi.vsapi.getFramePropertiesRO.?(src_frames[pb.clip]);
        prop_vals[pi] = common.readFrameProp(zapi, fprops, pb.name_z.ptr, std.math.nan(f32));
    }

    interp.evalPlane(allocator, prog, profile, .{
        .dst = dst_ptr,
        .dst_stride = dst_stride,
        .dst_fmt = dst_fmt,
        .srcs = src_ptrs,
        .src_strides = src_strides,
        .src_fmts = src_fmts,
        .width = pw,
        .height = ph,
        .frame_n = @intCast(frame_n),
        .props = prop_vals,
    }) catch {
        zapi.setFilterError("Expr: out of memory");
        return false;
    };
    return true;
}

fn executeCachedEntry(
    entry: *const plane_cache.Entry,
    zapi: ZAPI,
    dst_frame: *vs.Frame,
    src_frames: []const ?*const vs.Frame,
    src_ptrs: [][*]const u8,
    src_strides: []isize,
    src_fmts: []const ast.SampleFormat,
    dst_fmt: ast.SampleFormat,
    profile: interp.TargetProfile,
    plane_idx: usize,
    frame_n: c_int,
) bool {
    const n_clips = src_frames.len;
    switch (entry.exec) {
        .lut => |*l| {
            const dst_ptr = zapi.vsapi.getWritePtr.?(dst_frame, @intCast(plane_idx)).?;
            const dst_stride: isize = zapi.vsapi.getStride.?(dst_frame, @intCast(plane_idx));
            const pw: usize = @intCast(zapi.vsapi.getFrameWidth.?(dst_frame, @intCast(plane_idx)));
            const ph: usize = @intCast(zapi.vsapi.getFrameHeight.?(dst_frame, @intCast(plane_idx)));
            for (0..n_clips) |ci| {
                src_ptrs[ci] = zapi.vsapi.getReadPtr.?(src_frames[ci], @intCast(plane_idx)).?;
                src_strides[ci] = zapi.vsapi.getStride.?(src_frames[ci], @intCast(plane_idx));
            }
            l.apply(dst_ptr, dst_stride, src_ptrs, src_strides, pw, ph);
            return true;
        },
        .jit => |j| {
            const dst_ptr = zapi.vsapi.getWritePtr.?(dst_frame, @intCast(plane_idx)).?;
            const dst_stride: isize = zapi.vsapi.getStride.?(dst_frame, @intCast(plane_idx));
            for (0..n_clips) |ci| {
                src_ptrs[ci] = zapi.vsapi.getReadPtr.?(src_frames[ci], @intCast(plane_idx)).?;
                src_strides[ci] = zapi.vsapi.getStride.?(src_frames[ci], @intCast(plane_idx));
            }
            var prop_stack: [64]f32 = undefined;
            const prop_heap: ?[]f32 = if (j.props.len > prop_stack.len)
                allocator.alloc(f32, j.props.len) catch {
                    zapi.setFilterError("Expr: out of memory");
                    return false;
                }
            else
                null;
            defer if (prop_heap) |ph_buf| allocator.free(ph_buf);
            const prop_vals: []f32 = if (prop_heap) |ph_buf| ph_buf else prop_stack[0..j.props.len];

            for (j.props, 0..) |pb, pi| {
                const fprops = zapi.vsapi.getFramePropertiesRO.?(src_frames[pb.clip]);
                prop_vals[pi] = common.readFrameProp(zapi, fprops, pb.name_z.ptr, std.math.nan(f32));
            }

            j.kernel.func(
                dst_ptr,
                dst_stride,
                src_ptrs.ptr,
                src_strides.ptr,
                @intCast(frame_n),
                if (prop_vals.len > 0) prop_vals.ptr else null,
            );
            return true;
        },
        .scalar => |*s| {
            return executeScalarProg(
                zapi,
                &s.prog,
                s.props,
                dst_frame,
                src_frames,
                src_ptrs,
                src_strides,
                src_fmts,
                dst_fmt,
                profile,
                plane_idx,
                frame_n,
            );
        },
    }
}

fn compileDynamicPlane(
    zapi: ZAPI,
    dyn: *DynamicData,
    profile: interp.TargetProfile,
    plane_idx: usize,
    expr_str: []const u8,
    num_clips: u16,
    src_fmts: []const ast.SampleFormat,
    dst_fmt: ast.SampleFormat,
    plane_key: plane_cache.Key,
    builtin_ctx: ?*const builtins.ResolveContext,
) (std.mem.Allocator.Error || error{CompileFailed})!*plane_cache.Entry {
    var prog = parser.parse(allocator, expr_str, .{
        .mode = .expr,
        .num_clips = num_clips,
        .default_boundary = dyn.default_boundary,
        .opt_int = dyn.opt_int,
        .builtin_ctx = builtin_ctx,
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => unreachable, // Syntax, clip bounds, and >16-bit rules already validated.
    };
    errdefer prog.deinit(allocator);

    try lower_mod.lower(allocator, &prog, profile);
    try optimize_mod.optimize(allocator, &prog, profile, .{
        .enable_value_changing = true,
        .enable_fma = profile.use_fma,
    });

    const pw = plane_key.width;
    const ph = plane_key.height;
    var leased_rt = false;
    const rt_ptr: ?*aj.Runtime = if (dyn.cpu.isa != .scalar) switch (global_plane_cache.acquireRuntime()) {
        .lease => |l| blk: {
            leased_rt = true;
            break :blk l.rt;
        },
        .unavailable => null,
        .closed => {
            setFilterErrorFmt(zapi, "plugin is shutting down", .{});
            return error.CompileFailed;
        },
    } else null;
    defer if (leased_rt) global_plane_cache.releaseRuntime();

    const has_fast_u8_lut = rt_ptr != null and aj.Builder.cpuSupportsFastLutU8(dyn.cpu);
    var info = try analysis_mod.analyze(allocator, &prog, src_fmts, dst_fmt, @intCast(pw), @intCast(ph), .{
        .has_fast_u8_lut = has_fast_u8_lut,
        .f32_lanes = dyn.cpu.f32_lanes,
    });
    defer info.deinit(allocator);

    if (info.lut_kind != .none) {
        var lut_execmem_denied = false;
        const lut = try lut_mod.Lut.build(
            allocator,
            rt_ptr,
            dyn.cpu,
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
        if (lut_execmem_denied and !dyn.execmem_warned.swap(true, .acq_rel)) {
            zapi.logMessage(.Warning, "zexpr.Expr: executable memory allocation denied; falling back to scalar LUT");
        }
        var store_key = plane_key;
        if (lut.simd_kernel == null) {
            store_key.width = 0;
            store_key.height = 0;
        }
        var exec: plane_cache.CompiledPlane = .{ .lut = lut };
        errdefer exec.deinit(allocator, rt_ptr);
        return global_plane_cache.putOrGet(allocator, store_key, exec) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.CacheClosed => {
                setFilterErrorFmt(zapi, "plugin is shutting down", .{});
                return error.CompileFailed;
            },
        };
    }

    const prop_bindings = try common.buildPropBindings(allocator, &prog);
    var owns_props = true;
    errdefer if (owns_props) common.freePropBindings(allocator, prop_bindings);

    var compiled_kernel: ?aj.CompiledKernel = null;
    var jit_execmem_denied = false;
    if (rt_ptr) |runtime| {
        if (codegen.compilePlane(allocator, runtime, dyn.cpu, &prog, &info, .{
            .width = pw,
            .height = ph,
            .src_fmts = src_fmts,
            .dst_fmt = dst_fmt,
        })) |res| {
            compiled_kernel = res.kernel;
        } else |err| switch (err) {
            error.ExecMemDenied => {
                jit_execmem_denied = true;
                if (!dyn.execmem_warned.swap(true, .acq_rel)) {
                    const msg = if (info.fallback_lut_kind != .none)
                        "zexpr.Expr: executable memory allocation denied; falling back to scalar LUT"
                    else
                        "zexpr.Expr: executable memory allocation denied; falling back to scalar interpreter";
                    zapi.logMessage(.Warning, msg);
                }
            },
            error.UnsupportedFeature => {},
            error.OutOfMemory => return error.OutOfMemory,
            error.AbiMismatch, error.CompileFailed => {
                setFilterErrorFmt(zapi, "JIT compilation failed on plane {d}: {s}", .{ plane_idx, @errorName(err) });
                return error.CompileFailed;
            },
        }
    }

    if (compiled_kernel) |kernel| {
        prog.deinit(allocator);
        owns_props = false;
        var exec: plane_cache.CompiledPlane = .{ .jit = .{ .kernel = kernel, .props = prop_bindings } };
        errdefer exec.deinit(allocator, rt_ptr);
        return global_plane_cache.putOrGet(allocator, plane_key, exec) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.CacheClosed => {
                setFilterErrorFmt(zapi, "plugin is shutting down", .{});
                return error.CompileFailed;
            },
        };
    } else if ((rt_ptr == null or jit_execmem_denied) and info.fallback_lut_kind != .none) {
        info.lut_kind = info.fallback_lut_kind;
        info.lut_clips = info.fallback_lut_clips;
        const lut = try lut_mod.Lut.build(
            allocator,
            null,
            dyn.cpu,
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
        var store_key = plane_key;
        store_key.width = 0;
        store_key.height = 0;
        var exec: plane_cache.CompiledPlane = .{ .lut = lut };
        errdefer exec.deinit(allocator, null);
        return global_plane_cache.putOrGet(allocator, store_key, exec) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.CacheClosed => {
                setFilterErrorFmt(zapi, "plugin is shutting down", .{});
                return error.CompileFailed;
            },
        };
    } else {
        // F3: Cache unsupported-JIT / ExecMemDenied scalar program under `plane_key` so repeated
        // frames (e.g. non-LUT `f16` on SSE2/v2) avoid per-frame re-parse/lower/optimize.
        owns_props = false;
        var exec: plane_cache.CompiledPlane = .{ .scalar = .{ .prog = prog, .props = prop_bindings } };
        // `prog` ownership transfers to `exec`; prevent outer `errdefer prog.deinit` from double-freeing on OOM.
        prog = .{};
        errdefer exec.deinit(allocator, null);
        return global_plane_cache.putOrGet(allocator, plane_key, exec) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.CacheClosed => {
                setFilterErrorFmt(zapi, "plugin is shutting down", .{});
                return error.CompileFailed;
            },
        };
    }
}

fn exprGetFrameDynamic(
    d: *ExprData,
    dyn: *DynamicData,
    zapi: ZAPI,
    core: ?*vs.Core,
    n: c_int,
    src_frames: []const ?*const vs.Frame,
) ?*const vs.Frame {
    const n_clips = src_frames.len;

    // 1. C3 invariant checks across all input frames (format support + dims/planes/subsampling match clip 0).
    var stack_vfs: [32]vs.VideoFormat = undefined;
    const heap_vfs: ?[]vs.VideoFormat = if (n_clips > stack_vfs.len)
        allocator.alloc(vs.VideoFormat, n_clips) catch {
            zapi.setFilterError("Expr: out of memory");
            return null;
        }
    else
        null;
    defer if (heap_vfs) |hv| allocator.free(hv);
    const frame_vfs: []vs.VideoFormat = if (heap_vfs) |hv| hv else stack_vfs[0..n_clips];

    var stack_fmts: [32]ast.SampleFormat = undefined;
    const heap_fmts: ?[]ast.SampleFormat = if (n_clips > stack_fmts.len)
        allocator.alloc(ast.SampleFormat, n_clips) catch {
            zapi.setFilterError("Expr: out of memory");
            return null;
        }
    else
        null;
    defer if (heap_fmts) |hf| allocator.free(hf);
    const frame_src_fmts: []ast.SampleFormat = if (heap_fmts) |hf| hf else stack_fmts[0..n_clips];

    var fw0: c_int = 0;
    var fh0: c_int = 0;
    var vf0: vs.VideoFormat = undefined;

    for (0..n_clips) |ci| {
        const vf = zapi.vsapi.getVideoFrameFormat.?(src_frames[ci]).*;
        if (vf.colorFamily == .Undefined) {
            setFilterErrorFmt(zapi, "clip {d} frame has undefined format", .{ci});
            return null;
        }
        frame_vfs[ci] = vf;
        frame_src_fmts[ci] = common.toSampleFormat(vf) orelse {
            setFilterErrorFmt(zapi, "clip {d} has unsupported sample format ({d}-bit)", .{ ci, vf.bitsPerSample });
            return null;
        };
        const fw = zapi.vsapi.getFrameWidth.?(src_frames[ci], 0);
        const fh = zapi.vsapi.getFrameHeight.?(src_frames[ci], 0);
        if (ci == 0) {
            fw0 = fw;
            fh0 = fh;
            vf0 = vf;
        } else if (fw != fw0 or fh != fh0 or
            vf.numPlanes != vf0.numPlanes or
            vf.subSamplingW != vf0.subSamplingW or
            vf.subSamplingH != vf0.subSamplingH)
        {
            setFilterErrorFmt(zapi, "clip {d} frame dimensions, plane count, or subsampling do not match clip 0", .{ci});
            return null;
        }
    }

    // 2. C3 format= override compatibility with clip 0's frame format.
    const dst_vf: vs.VideoFormat = if (dyn.has_format_override) blk_ov: {
        const ov = dyn.override_fmt;
        if (vf0.colorFamily != ov.colorFamily or
            vf0.numPlanes != ov.numPlanes or
            vf0.subSamplingW != ov.subSamplingW or
            vf0.subSamplingH != ov.subSamplingH)
        {
            zapi.setFilterError("Expr: input frame format does not match output format color family, plane count, or subsampling");
            return null;
        }
        break :blk_ov ov;
    } else vf0;

    // 3. C3 advertised out_vi invariants (prevents VapourSynth `logFatal` abort on frame mismatch).
    if (d.out_vi.format.colorFamily != .Undefined and !vsh.isSameVideoFormat(&d.out_vi.format, &dst_vf)) {
        zapi.setFilterError("Expr: frame format does not match clip video info format");
        return null;
    }
    if (d.out_vi.width != 0 and fw0 != d.out_vi.width) {
        zapi.setFilterError("Expr: frame width does not match clip video info width");
        return null;
    }
    if (d.out_vi.height != 0 and fh0 != d.out_vi.height) {
        zapi.setFilterError("Expr: frame height does not match clip video info height");
        return null;
    }
    if (!vsh.areValidDimensions(&dst_vf, fw0, fh0)) {
        zapi.setFilterError("Expr: frame dimensions are not compatible with output format subsampling");
        return null;
    }

    const frame_dst_fmt = common.toSampleFormat(dst_vf) orelse {
        setFilterErrorFmt(zapi, "unsupported output sample format ({d}-bit)", .{dst_vf.bitsPerSample});
        return null;
    };

    // 4. Allocate dst_frame (C3: only copy empty planes when sampleType & bitsPerSample match clip 0).
    const num_planes: usize = @intCast(dst_vf.numPlanes);
    const same_sample_as_src0 =
        dst_vf.sampleType == vf0.sampleType and
        dst_vf.bitsPerSample == vf0.bitsPerSample;
    var plane_srcs: [max_planes]?*const vs.Frame = .{ null, null, null };
    const plane_indices = [max_planes]c_int{ 0, 1, 2 };
    for (0..num_planes) |p| {
        if (dyn.plane_exprs[p].len == 0 and same_sample_as_src0) {
            plane_srcs[p] = src_frames[0];
        }
    }
    const dst_frame = zapi.vsapi.newVideoFrame2.?(
        &dst_vf,
        fw0,
        fh0,
        &plane_srcs,
        &plane_indices,
        src_frames[0],
        core,
    ) orelse {
        zapi.setFilterError("Expr: failed to allocate output frame");
        return null;
    };
    var ok = false;
    defer if (!ok) zapi.freeFrame(dst_frame);

    var stack_ptrs: [32][*]const u8 = undefined;
    var stack_strides: [32]isize = undefined;
    const heap_ptrs: ?[][*]const u8 = if (n_clips > stack_ptrs.len)
        allocator.alloc([*]const u8, n_clips) catch {
            zapi.setFilterError("Expr: out of memory");
            return null;
        }
    else
        null;
    defer if (heap_ptrs) |hp| allocator.free(hp);
    const heap_strides: ?[]isize = if (n_clips > stack_strides.len)
        allocator.alloc(isize, n_clips) catch {
            zapi.setFilterError("Expr: out of memory");
            return null;
        }
    else
        null;
    defer if (heap_strides) |hs| allocator.free(hs);
    const src_ptrs: [][*]const u8 = if (heap_ptrs) |hp| hp else stack_ptrs[0..n_clips];
    const src_strides: []isize = if (heap_strides) |hs| hs else stack_strides[0..n_clips];

    // Check if any active plane uses built-in constants.
    var any_builtins = false;
    for (0..num_planes) |p| {
        if (dyn.plane_exprs[p].len > 0 and dyn.plane_builtins[p].len > 0) {
            any_builtins = true;
            break;
        }
    }

    var stack_metas: [32]builtins.ClipMeta = undefined;
    const heap_metas: ?[]builtins.ClipMeta = if (any_builtins and n_clips > stack_metas.len)
        allocator.alloc(builtins.ClipMeta, n_clips) catch {
            zapi.setFilterError("Expr: out of memory");
            return null;
        }
    else
        null;
    defer if (heap_metas) |hm| allocator.free(hm);
    const src_metas: []builtins.ClipMeta = if (any_builtins)
        (if (heap_metas) |hm| hm else stack_metas[0..n_clips])
    else
        &.{};

    var stack_full: [32]bool = undefined;
    const heap_full: ?[]bool = if (any_builtins and n_clips > stack_full.len)
        allocator.alloc(bool, n_clips) catch {
            zapi.setFilterError("Expr: out of memory");
            return null;
        }
    else
        null;
    defer if (heap_full) |hf| allocator.free(hf);
    const clip_is_full: []bool = if (any_builtins)
        (if (heap_full) |hf| hf else stack_full[0..n_clips])
    else
        &.{};
    var dst_is_full = true;

    if (any_builtins) {
        for (0..n_clips) |ci| {
            clip_is_full[ci] = if (frame_vfs[ci].sampleType == .Float)
                true
            else
                common.readFrameIsFullRange(zapi, zapi.vsapi.getFramePropertiesRO.?(src_frames[ci]), frame_vfs[ci]);
        }
        // E2: Always resolve dst's `_Range` using `dst_vf`, even when clip 0 is float.
        dst_is_full = if (dst_vf.sampleType == .Float)
            true
        else
            common.readFrameIsFullRange(zapi, zapi.vsapi.getFramePropertiesRO.?(src_frames[0]), dst_vf);
    }

    for (0..num_planes) |p| {
        const expr_str = dyn.plane_exprs[p];
        if (expr_str.len == 0) continue;

        const specs = dyn.plane_builtins[p];
        var stack_rb: [32]builtins.BuiltinVal = undefined;
        const heap_rb: ?[]builtins.BuiltinVal = if (specs.len > stack_rb.len)
            allocator.alloc(builtins.BuiltinVal, specs.len) catch {
                zapi.setFilterError("Expr: out of memory");
                return null;
            }
        else
            null;
        defer if (heap_rb) |hr| allocator.free(hr);
        const resolved_vals: []builtins.BuiltinVal = if (specs.len > 0)
            (if (heap_rb) |hr| hr else stack_rb[0..specs.len])
        else
            &.{};

        var rctx_storage: builtins.ResolveContext = undefined;
        var rctx_ptr: ?*const builtins.ResolveContext = null;

        if (specs.len > 0) {
            for (0..n_clips) |ci| {
                src_metas[ci] = common.clipMetaFromVi(frame_vfs[ci], @intCast(p), clip_is_full[ci]) orelse {
                    setFilterErrorFmt(zapi, "plane {d}: unsupported clip {d} color family", .{ p, ci });
                    return null;
                };
            }
            const dst_meta = common.clipMetaFromVi(dst_vf, @intCast(p), dst_is_full) orelse {
                setFilterErrorFmt(zapi, "plane {d}: unsupported output color family", .{p});
                return null;
            };
            for (specs, 0..) |spec, si| {
                resolved_vals[si] = builtins.resolveOne(spec, src_metas, dst_meta) catch |err| switch (err) {
                    error.UnsupportedIntBitDepthConstant => {
                        // E3: Name both plane and constant when rejecting >16-bit integer built-ins per frame.
                        setFilterErrorFmt(
                            zapi,
                            "plane {d}: built-in constant '{s}' is unsupported on >16-bit integer clips",
                            .{ p, @tagName(spec.kind) },
                        );
                        return null;
                    },
                };
            }
            rctx_storage = .{
                .src_metas = src_metas,
                .dst_meta = dst_meta,
                .opt_int = dyn.opt_int,
            };
            rctx_ptr = &rctx_storage;
        }

        // F3: Level 0 (`force_scalar`) bypasses global_plane_cache, LUT, and JIT completely.
        if (dyn.force_scalar) {
            var prog = parser.parse(allocator, expr_str, .{
                .mode = .expr,
                .num_clips = @intCast(n_clips),
                .default_boundary = dyn.default_boundary,
                .opt_int = dyn.opt_int,
                .builtin_ctx = rctx_ptr,
            }) catch |err| switch (err) {
                error.OutOfMemory => {
                    zapi.setFilterError("Expr: out of memory");
                    return null;
                },
                else => unreachable,
            };
            defer prog.deinit(allocator);

            lower_mod.lower(allocator, &prog, d.profile) catch {
                zapi.setFilterError("Expr: out of memory");
                return null;
            };
            optimize_mod.optimize(allocator, &prog, d.profile, .{
                .enable_value_changing = true,
                .enable_fma = d.profile.use_fma,
            }) catch {
                zapi.setFilterError("Expr: out of memory");
                return null;
            };
            const prop_bindings = common.buildPropBindings(allocator, &prog) catch {
                zapi.setFilterError("Expr: out of memory");
                return null;
            };
            defer common.freePropBindings(allocator, prop_bindings);

            if (!executeScalarProg(
                zapi,
                &prog,
                prop_bindings,
                dst_frame,
                src_frames,
                src_ptrs,
                src_strides,
                frame_src_fmts,
                frame_dst_fmt,
                d.profile,
                p,
                n,
            )) return null;
            continue;
        }

        const pw: usize = @intCast(zapi.vsapi.getFrameWidth.?(dst_frame, @intCast(p)));
        const ph: usize = @intCast(zapi.vsapi.getFrameHeight.?(dst_frame, @intCast(p)));
        const plane_key: plane_cache.Key = .{
            .expr = expr_str,
            .src_fmts = frame_src_fmts,
            .dst_fmt = frame_dst_fmt,
            .width = pw,
            .height = ph,
            .opt_int = dyn.opt_int,
            .default_boundary = dyn.default_boundary,
            .cpu = dyn.cpu,
            .profile = d.profile,
            .resolved_builtins = resolved_vals,
        };
        const entry: *plane_cache.Entry = global_plane_cache.getOrDim(plane_key) orelse
            (compileDynamicPlane(
                zapi,
                dyn,
                d.profile,
                p,
                expr_str,
                @intCast(n_clips),
                frame_src_fmts,
                frame_dst_fmt,
                plane_key,
                rctx_ptr,
            ) catch |err| switch (err) {
                error.OutOfMemory => {
                    zapi.setFilterError("Expr: out of memory");
                    return null;
                },
                error.CompileFailed => return null,
            });
        defer global_plane_cache.release(allocator, entry);

        if (!executeCachedEntry(
            entry,
            zapi,
            dst_frame,
            src_frames,
            src_ptrs,
            src_strides,
            frame_src_fmts,
            frame_dst_fmt,
            d.profile,
            p,
            n,
        )) return null;
    }

    ok = true;
    return dst_frame;
}

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
    const d: *ExprData = @ptrCast(@alignCast(instance_data));
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

            switch (d.mode) {
                .dynamic => |*dyn| return exprGetFrameDynamic(d, dyn, zapi, core, n, src_frames),
                .static => |*st| {
                    // Build dst frame using newVideoFrame2: copy planes reference src_frames[0], others null.
                    var plane_srcs: [max_planes]?*const vs.Frame = .{ null, null, null };
                    const plane_indices = [max_planes]c_int{ 0, 1, 2 };
                    for (0..st.num_planes) |p| {
                        if (st.planes[p] == .single and st.planes[p].single == .copy) {
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
                    var ok = false;
                    defer if (!ok) zapi.freeFrame(dst_frame);

                    var stack_ptrs: [32][*]const u8 = undefined;
                    var stack_strides: [32]isize = undefined;
                    const heap_ptrs: ?[][*]const u8 = if (n_clips > stack_ptrs.len)
                        allocator.alloc([*]const u8, n_clips) catch {
                            zapi.setFilterError("Expr: out of memory");
                            return null;
                        }
                    else
                        null;
                    defer if (heap_ptrs) |hp| allocator.free(hp);
                    const heap_strides: ?[]isize = if (n_clips > stack_strides.len)
                        allocator.alloc(isize, n_clips) catch {
                            zapi.setFilterError("Expr: out of memory");
                            return null;
                        }
                    else
                        null;
                    defer if (heap_strides) |hs| allocator.free(hs);

                    const src_ptrs: [][*]const u8 = if (heap_ptrs) |hp| hp else stack_ptrs[0..n_clips];
                    const src_strides: []isize = if (heap_strides) |hs| hs else stack_strides[0..n_clips];

                    for (0..st.num_planes) |p| {
                        const exec: *const SinglePlaneExec = switch (st.planes[p]) {
                            .single => |*s| s,
                            .range_variants => |rv| blk_rv: {
                                var mask: usize = 0;
                                for (0..rv.clip_count) |bi| {
                                    const fprops = zapi.vsapi.getFramePropertiesRO.?(src_frames[rv.clip_ids[bi]]);
                                    if (common.readFrameIsFullRange(zapi, fprops, rv.clip_fmts[bi])) {
                                        mask |= (@as(usize, 1) << @intCast(bi));
                                    }
                                }
                                break :blk_rv &rv.variants[mask];
                            },
                        };
                        switch (exec.*) {
                            .copy, .undefined_plane => continue,
                            .cached => |entry| {
                                if (!executeCachedEntry(
                                    entry,
                                    zapi,
                                    dst_frame,
                                    src_frames,
                                    src_ptrs,
                                    src_strides,
                                    st.src_fmts,
                                    st.dst_fmt,
                                    d.profile,
                                    p,
                                    n,
                                )) return null;
                            },
                            .scalar => |s| {
                                if (!executeScalarProg(
                                    zapi,
                                    &s.prog,
                                    s.props,
                                    dst_frame,
                                    src_frames,
                                    src_ptrs,
                                    src_strides,
                                    st.src_fmts,
                                    st.dst_fmt,
                                    d.profile,
                                    p,
                                    n,
                                )) return null;
                            },
                        }
                    }

                    ok = true;
                    return dst_frame;
                },
            }
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

fn compileSinglePlane(
    zapi: ZAPI,
    map_out: ZAPI.ZMap(?*vs.Map),
    plane_idx: usize,
    expr_str: []const u8,
    num_clips: u16,
    default_boundary: ast.BoundaryMode,
    opt_int: bool,
    cpu: cpu_mod.CpuInfo,
    profile: interp.TargetProfile,
    src_fmts: []const ast.SampleFormat,
    dst_fmt: ast.SampleFormat,
    plane_key: plane_cache.Key,
    rt_ptr: ?*aj.Runtime,
    force_scalar: bool,
    execmem_warned: *bool,
    existing_prog: ?*ast.Program,
    builtin_ctx: ?*const builtins.ResolveContext,
) (common.ArgError || std.mem.Allocator.Error)!SinglePlaneExec {
    if (!force_scalar) {
        if (global_plane_cache.getOrDim(plane_key)) |entry| {
            return .{ .cached = entry };
        }
    }

    var prog: ast.Program = undefined;
    if (existing_prog != null and builtin_ctx == null) {
        prog = existing_prog.?.*;
        existing_prog.?.* = .{};
    } else {
        prog = parser.parse(allocator, expr_str, .{
            .mode = .expr,
            .num_clips = num_clips,
            .default_boundary = default_boundary,
            .opt_int = opt_int,
            .builtin_ctx = builtin_ctx,
        }) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => unreachable, // Phase 2 already validated syntax, clip bounds, and >16-bit rules.
        };
    }
    errdefer prog.deinit(allocator);

    try lower_mod.lower(allocator, &prog, profile);
    try optimize_mod.optimize(allocator, &prog, profile, .{
        .enable_value_changing = true,
        .enable_fma = profile.use_fma,
    });

    const pw = plane_key.width;
    const ph = plane_key.height;

    const has_fast_u8_lut = rt_ptr != null and aj.Builder.cpuSupportsFastLutU8(cpu);
    var info = try analysis_mod.analyze(allocator, &prog, src_fmts, dst_fmt, @intCast(pw), @intCast(ph), .{
        .has_fast_u8_lut = has_fast_u8_lut,
        .f32_lanes = cpu.f32_lanes,
    });
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
        if (lut_execmem_denied and !execmem_warned.*) {
            execmem_warned.* = true;
            zapi.logMessage(.Warning, "zexpr.Expr: executable memory allocation denied; falling back to scalar LUT");
        }
        // C9: Cache scalar LUTs even after ExecMemDenied (they require zero executable memory).
        var store_key = plane_key;
        if (lut.simd_kernel == null) {
            store_key.width = 0;
            store_key.height = 0;
        }
        const entry = cache_blk: {
            var exec: plane_cache.CompiledPlane = .{ .lut = lut };
            errdefer exec.deinit(allocator, rt_ptr);
            break :cache_blk global_plane_cache.putOrGet(allocator, store_key, exec) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                error.CacheClosed => return common.fail(map_out, filter_name, "plugin is shutting down", .{}),
            };
        };
        return .{ .cached = entry };
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
                if (!execmem_warned.*) {
                    execmem_warned.* = true;
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
                return common.fail(map_out, filter_name, "JIT compilation failed on plane {d}: {s}", .{ plane_idx, @errorName(err) });
            },
        }
    }

    if (compiled_kernel) |kernel| {
        prog.deinit(allocator);
        owns_props = false;
        const entry = cache_blk: {
            var exec: plane_cache.CompiledPlane = .{ .jit = .{ .kernel = kernel, .props = prop_bindings } };
            errdefer exec.deinit(allocator, rt_ptr);
            break :cache_blk global_plane_cache.putOrGet(allocator, plane_key, exec) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                error.CacheClosed => return common.fail(map_out, filter_name, "plugin is shutting down", .{}),
            };
        };
        return .{ .cached = entry };
    } else if (!force_scalar and (rt_ptr == null or jit_execmem_denied) and info.fallback_lut_kind != .none) {
        // M1 + C9: when executable memory is unavailable and the plane can be served by a scalar table,
        // build and cache the table (via scalar oracle, zero execmem) instead of dropping to per-pixel interp.
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
        var store_key = plane_key;
        store_key.width = 0;
        store_key.height = 0;
        const entry = cache_blk: {
            var exec: plane_cache.CompiledPlane = .{ .lut = lut };
            errdefer exec.deinit(allocator, null);
            break :cache_blk global_plane_cache.putOrGet(allocator, store_key, exec) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                error.CacheClosed => return common.fail(map_out, filter_name, "plugin is shutting down", .{}),
            };
        };
        return .{ .cached = entry };
    } else {
        const scalar_exec = try allocator.create(plane_cache.ScalarExec);
        scalar_exec.* = .{ .prog = prog, .props = prop_bindings };
        owns_props = false;
        return .{ .scalar = scalar_exec };
    }
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

    const clip_num_frames = try allocator.alloc(i32, nodes.len);
    var owns_clip_nf = true;
    errdefer if (owns_clip_nf) allocator.free(clip_num_frames);

    var any_var_fmt = false;
    var any_var_res = false;
    var ref_fmt: ?vs.VideoFormat = null;
    var ref_w: c_int = 0;
    var ref_h: c_int = 0;

    for (nodes, 0..) |node, i| {
        const vi = zapi.getVideoInfo(node);
        clip_num_frames[i] = vi.numFrames;

        if (vi.format.colorFamily == .Undefined) {
            any_var_fmt = true;
        } else {
            if (common.toSampleFormat(vi.format) == null) {
                return common.fail(map_out, filter_name, "clip {d} has unsupported sample format ({d}-bit)", .{ i, vi.format.bitsPerSample });
            }
            if (ref_fmt) |rf| {
                if (vi.format.numPlanes != rf.numPlanes or
                    vi.format.subSamplingW != rf.subSamplingW or
                    vi.format.subSamplingH != rf.subSamplingH)
                {
                    return common.fail(map_out, filter_name, "clip {d} plane count or subsampling do not match clip 0", .{i});
                }
            } else {
                ref_fmt = vi.format;
            }
        }

        if (vi.width == 0 or vi.height == 0) {
            any_var_res = true;
        } else if (ref_w == 0) {
            ref_w = vi.width;
            ref_h = vi.height;
        } else if (vi.width != ref_w or vi.height != ref_h) {
            return common.fail(map_out, filter_name, "clip {d} dimensions do not match clip 0", .{i});
        }
    }

    // Validate optional `format=` override (D14).
    var out_vi = vi0.*;
    // If clip 0 has variable resolution (`0x0`) while another input fixes the dimensions,
    // every frame must match those fixed dimensions anyway, so advertise them on `out_vi`.
    if ((out_vi.width == 0 or out_vi.height == 0) and ref_w != 0 and ref_h != 0) {
        out_vi.width = ref_w;
        out_vi.height = ref_h;
    }
    var has_format_override = false;
    var override_fmt: vs.VideoFormat = undefined;
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
        if (vi0.format.colorFamily != .Undefined) {
            if (new_fmt.colorFamily != vi0.format.colorFamily or
                new_fmt.numPlanes != vi0.format.numPlanes or
                new_fmt.subSamplingW != vi0.format.subSamplingW or
                new_fmt.subSamplingH != vi0.format.subSamplingH)
            {
                return common.fail(map_out, filter_name, "output format must match input color family, plane count, and subsampling", .{});
            }
        } else if (ref_fmt) |rf| {
            if (new_fmt.numPlanes != rf.numPlanes or
                new_fmt.subSamplingW != rf.subSamplingW or
                new_fmt.subSamplingH != rf.subSamplingH)
            {
                return common.fail(map_out, filter_name, "output format must match input plane count and subsampling", .{});
            }
        }
        out_vi.format = new_fmt;
        has_format_override = true;
        override_fmt = new_fmt;
    }

    // F4 + cross-clip geometry check: Validate known dimensions against known input subsampling AND
    // any `format=` override BEFORE transferring node ownership to createVideoFilter, because
    // VapourSynth's `isValidVideoInfo` rejects invalid VideoInfo without calling `exprFree` and
    // cross-clip fixed odd dims + fixed subsampled format (e.g. 17x9 varformat + YUV420P8 varsize)
    // can never produce a matching frame.
    if (out_vi.format.colorFamily != .Undefined) {
        if (common.toSampleFormat(out_vi.format) == null) {
            return common.fail(map_out, filter_name, "unsupported output sample format", .{});
        }
        if (!vsh.areValidDimensions(&out_vi.format, out_vi.width, out_vi.height)) {
            return common.fail(
                map_out,
                filter_name,
                "output dimensions ({d}x{d}) are not compatible with output format subsampling",
                .{ out_vi.width, out_vi.height },
            );
        }
    }
    if (ref_w != 0 or ref_h != 0) {
        if (ref_fmt) |rf| {
            if (!vsh.areValidDimensions(&rf, ref_w, ref_h)) {
                return common.fail(
                    map_out,
                    filter_name,
                    "input dimensions ({d}x{d}) are not compatible with clip format subsampling",
                    .{ ref_w, ref_h },
                );
            }
        }
        if (has_format_override and !vsh.areValidDimensions(&override_fmt, ref_w, ref_h)) {
            return common.fail(
                map_out,
                filter_name,
                "input dimensions ({d}x{d}) are not compatible with output format subsampling",
                .{ ref_w, ref_h },
            );
        }
    }

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
    // Axis 5: fixed-format input enforces `1 <= expr.len <= vi0.format.numPlanes`;
    // variable-format input (`vi0.format.colorFamily == .Undefined`, with or without `format=`)
    // accepts `1 <= expr.len <= 3` (all entries parsed and validated) and executes `0..dst_vf.numPlanes`.
    const expr_planes: usize = if (vi0.format.colorFamily != .Undefined)
        @intCast(vi0.format.numPlanes)
    else
        max_planes;
    const expr_count_raw = vsapi.?.mapNumElements.?(in, "expr");
    if (expr_count_raw <= 0 or expr_count_raw > @as(c_int, @intCast(expr_planes))) {
        return common.fail(map_out, filter_name, "expr must have between 1 and {d} elements", .{expr_planes});
    }
    const expr_count: usize = @intCast(expr_count_raw);

    var plane_exprs: [max_planes][]const u8 = .{ "", "", "" };
    for (0..expr_planes) |p| {
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

    // Oracle profile: on x86, ZEXPR_CPU_LEVEL=0 defaults to use_fma=false (matching SSE2/v2)
    // unless ZEXPR_ORACLE_FMA=1 (N4); on AArch64, NEON always fuses FMA so level 0 uses use_fma=true (AR4).
    const profile: interp.TargetProfile = if (env_override != null and env_override.? == 0)
        interp.TargetProfile.fromCpuWithOracleFma(cpu, cpu_mod.readOracleFmaOverride())
    else if (cpu.isa == .scalar)
        .{ .use_fma = cpu_mod.detectWithOverride(null).has_fma, .isa_family = cpu_mod.IsaFamily.host() }
    else
        interp.TargetProfile.fromCpu(cpu);

    const force_scalar = env_override != null and env_override.? == 0;
    var use_dynamic = any_var_fmt or any_var_res;

    if (!use_dynamic) {
        const num_planes: usize = expr_planes;
        const src_fmts = try allocator.alloc(ast.SampleFormat, nodes.len);
        var owns_src_fmts = true;
        errdefer if (owns_src_fmts) allocator.free(src_fmts);
        for (nodes, 0..) |node, i| {
            src_fmts[i] = common.toSampleFormat(zapi.getVideoInfo(node).format).?;
        }
        const dst_fmt = common.toSampleFormat(out_vi.format).?;

        var planes: [max_planes]PlaneExec = .{ .{ .single = .copy }, .{ .single = .copy }, .{ .single = .copy } };
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

        // Phase 1: check LRU cache for each non-empty plane with `resolved_builtins = &.{}` (V4: zero-parse
        // fast path for built-in-free FrameEval hits; built-in expressions always have non-empty resolved_builtins).
        for (0..num_planes) |p| {
            const expr_str = plane_exprs[p];
            if (expr_str.len == 0) {
                planes[p] = .{ .single = if (same_sample_as_src0) .copy else .undefined_plane };
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
                if (global_plane_cache.getOrDim(plane_keys[p])) |entry| {
                    planes[p] = .{ .single = .{ .cached = entry } };
                    continue;
                }
            }
            needs_compile[p] = true;
            any_miss = true;
        }

        // Phase 2: parse all missed planes before compiling any, so syntax / >16-bit errors
        // on plane 1/2 fail without wasting JIT compilation or evicting live cache entries.
        var parsed_progs: [max_planes]?ast.Program = .{ null, null, null };
        defer {
            for (&parsed_progs) |*mp| {
                if (mp.*) |*pr| pr.deinit(allocator);
            }
        }
        var resolved_builtins: [max_planes][]builtins.BuiltinVal = .{ &.{}, &.{}, &.{} };
        defer {
            for (resolved_builtins) |rb| {
                if (rb.len > 0) allocator.free(rb);
            }
        }
        const PlaneRangeClips = struct {
            ids: [2]ast.ClipId = .{ 0, 0 },
            count: u8 = 0,
        };
        var plane_range_clips: [max_planes]PlaneRangeClips = .{ .{}, .{}, .{} };

        var stack_metas: [32]builtins.ClipMeta = undefined;
        const heap_metas: ?[]builtins.ClipMeta = if (nodes.len > stack_metas.len)
            try allocator.alloc(builtins.ClipMeta, nodes.len)
        else
            null;
        defer if (heap_metas) |hm| allocator.free(hm);
        const src_metas: []builtins.ClipMeta = if (heap_metas) |hm| hm else stack_metas[0..nodes.len];

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

            const specs = parsed_progs[p].?.builtins.items;
            if (specs.len > 0) {
                // D7: Deduplicate integer `_Range`-dependent clip IDs (`x:` and `dst:` share clip 0's single bit).
                var rc: PlaneRangeClips = .{};
                for (specs) |spec| {
                    if (builtins.rangeClipId(spec, src_fmts, dst_fmt)) |cid| {
                        var seen = false;
                        for (rc.ids[0..rc.count]) |existing| {
                            if (existing == cid) {
                                seen = true;
                                break;
                            }
                        }
                        if (!seen) {
                            if (rc.count >= 2) {
                                // Phase C: k > 2 distinct integer `_Range` clips fall through to dynamic mode.
                                use_dynamic = true;
                            } else {
                                rc.ids[rc.count] = cid;
                                rc.count += 1;
                            }
                        }
                    }
                }
                plane_range_clips[p] = rc;

                for (nodes, 0..) |node, ci| {
                    const vf = zapi.getVideoInfo(node).format;
                    src_metas[ci] = common.clipMetaFromVi(vf, @intCast(p), true) orelse {
                        return common.fail(map_out, filter_name, "plane {d}: unsupported clip {d} color family", .{ p, ci });
                    };
                }
                const dst_meta = common.clipMetaFromVi(out_vi.format, @intCast(p), true) orelse {
                    return common.fail(map_out, filter_name, "plane {d}: unsupported output color family", .{p});
                };
                const rb = try allocator.alloc(builtins.BuiltinVal, specs.len);
                resolved_builtins[p] = rb;
                for (specs, 0..) |spec, si| {
                    rb[si] = builtins.resolveOne(spec, src_metas, dst_meta) catch |err| switch (err) {
                        error.UnsupportedIntBitDepthConstant => {
                            return common.fail(
                                map_out,
                                filter_name,
                                "plane {d}: built-in constant '{s}' is unsupported on >16-bit integer clips",
                                .{ p, @tagName(spec.kind) },
                            );
                        },
                    };
                }
                if (rc.count == 0) {
                    plane_keys[p].resolved_builtins = rb;
                }
            }
        }

        if (!use_dynamic) {
            // Phase 3: compile missed planes (or precompile 2^k variants for 1 <= k <= 2) against the shared JitRuntime.
            var leased_rt = false;
            const rt_ptr: ?*aj.Runtime = if (any_miss and cpu.isa != .scalar) switch (global_plane_cache.acquireRuntime()) {
                .lease => |l| blk: {
                    leased_rt = true;
                    break :blk l.rt;
                },
                .unavailable => null,
                .closed => return common.fail(map_out, filter_name, "plugin is shutting down", .{}),
            } else null;
            defer if (leased_rt) global_plane_cache.releaseRuntime();
            var execmem_warned = false;

            for (0..num_planes) |p| {
                if (!needs_compile[p]) continue;

                const rc = plane_range_clips[p];
                if (rc.count > 0) {
                    // C1 + F2 + E2: Precompile all `2^k` limited/full variants in Create (`0` locks / `0` compiles in getFrame).
                    const specs = parsed_progs[p].?.builtins.items;
                    const rv = try allocator.create(RangeVariants);
                    rv.* = .{ .clip_count = rc.count };
                    for (0..rc.count) |bi| {
                        const cid = rc.ids[bi];
                        rv.clip_ids[bi] = cid;
                        // E2: When clip 0 is float and dst is integer, clip 0's shared bit must read `_Range` using dst's integer format.
                        rv.clip_fmts[bi] = if (cid == 0 and src_fmts[0].is_float)
                            out_vi.format
                        else
                            zapi.getVideoInfo(nodes[cid]).format;
                    }
                    errdefer {
                        rv.deinit(allocator, &global_plane_cache);
                        allocator.destroy(rv);
                    }

                    const n_variants: usize = @as(usize, 1) << @intCast(rc.count);
                    for (0..n_variants) |mask| {
                        for (nodes, 0..) |node, ci| {
                            var is_full = true;
                            for (0..rc.count) |bi| {
                                if (rv.clip_ids[bi] == ci) {
                                    is_full = ((mask >> @intCast(bi)) & 1) != 0;
                                    break;
                                }
                            }
                            src_metas[ci] = common.clipMetaFromVi(zapi.getVideoInfo(node).format, @intCast(p), is_full).?;
                        }
                        var dst_full = true;
                        for (0..rc.count) |bi| {
                            if (rv.clip_ids[bi] == 0) {
                                dst_full = ((mask >> @intCast(bi)) & 1) != 0;
                                break;
                            }
                        }
                        const dst_meta = common.clipMetaFromVi(out_vi.format, @intCast(p), dst_full).?;

                        builtins.resolveAll(specs, src_metas, dst_meta, resolved_builtins[p]) catch unreachable;
                        var vkey = plane_keys[p];
                        vkey.resolved_builtins = resolved_builtins[p];

                        const rctx: builtins.ResolveContext = .{
                            .src_metas = src_metas,
                            .dst_meta = dst_meta,
                            .opt_int = opt_val == 1,
                        };
                        rv.variants[mask] = try compileSinglePlane(
                            zapi,
                            map_out,
                            p,
                            plane_exprs[p],
                            @intCast(nodes.len),
                            default_boundary,
                            opt_val == 1,
                            cpu,
                            profile,
                            src_fmts,
                            dst_fmt,
                            vkey,
                            rt_ptr,
                            force_scalar,
                            &execmem_warned,
                            null,
                            &rctx,
                        );
                        rv.init_count += 1;
                    }
                    planes[p] = .{ .range_variants = rv };
                    continue;
                }

                const has_builtins = parsed_progs[p].?.builtins.items.len > 0;
                var rctx_storage: builtins.ResolveContext = undefined;
                const rctx_ptr: ?*const builtins.ResolveContext = if (has_builtins) blk_ctx: {
                    for (nodes, 0..) |node, ci| {
                        src_metas[ci] = common.clipMetaFromVi(zapi.getVideoInfo(node).format, @intCast(p), true).?;
                    }
                    const dst_meta = common.clipMetaFromVi(out_vi.format, @intCast(p), true).?;
                    rctx_storage = .{
                        .src_metas = src_metas,
                        .dst_meta = dst_meta,
                        .opt_int = opt_val == 1,
                    };
                    break :blk_ctx &rctx_storage;
                } else null;

                const single = try compileSinglePlane(
                    zapi,
                    map_out,
                    p,
                    plane_exprs[p],
                    @intCast(nodes.len),
                    default_boundary,
                    opt_val == 1,
                    cpu,
                    profile,
                    src_fmts,
                    dst_fmt,
                    plane_keys[p],
                    rt_ptr,
                    force_scalar,
                    &execmem_warned,
                    if (has_builtins) null else &parsed_progs[p].?,
                    rctx_ptr,
                );
                planes[p] = .{ .single = single };
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
                .out_vi = out_vi,
                .profile = profile,
                .mode = .{ .static = .{
                    .src_fmts = src_fmts,
                    .dst_fmt = dst_fmt,
                    .planes = planes,
                    .num_planes = num_planes,
                } },
            };

            for (nodes, clip_num_frames, 0..) |node, nf, i| {
                deps[i] = .{
                    .source = node,
                    .requestPattern = if (nf >= out_vi.numFrames) .StrictSpatial else .FrameReuseLastOnly,
                };
            }

            zapi.createVideoFilter(out, filter_name, &data.out_vi, exprGetFrame, exprFree, .Parallel, deps, data);
            return;
        }

        // `k > 2` triggered dynamic mode: release any Phase 1 cached refs & static `src_fmts` before dynamic setup.
        for (planes[0..num_planes]) |*pe| pe.deinit();
        owns_planes = false;
        allocator.free(src_fmts);
        owns_src_fmts = false;
    }

    // Dynamic mode setup (variable format, variable resolution, or k > 2 distinct integer `_Range` clips).
    var dyn: DynamicData = .{
        .plane_exprs = .{ &.{}, &.{}, &.{} },
        .plane_builtins = .{ &.{}, &.{}, &.{} },
        .has_format_override = has_format_override,
        .override_fmt = override_fmt,
        .opt_int = opt_val == 1,
        .default_boundary = default_boundary,
        .cpu = cpu,
        .force_scalar = force_scalar,
        .execmem_warned = .init(false),
    };
    var owns_dyn = true;
    errdefer if (owns_dyn) dyn.deinit();

    for (0..expr_planes) |p| {
        const expr_str = plane_exprs[p];
        if (expr_str.len == 0) continue;

        var prog = parser.parse(allocator, expr_str, .{
            .mode = .expr,
            .num_clips = @intCast(nodes.len),
            .default_boundary = default_boundary,
            .opt_int = opt_val == 1,
        }) catch |err| {
            return common.fail(map_out, filter_name, "parse error on plane {d}: {s}", .{ p, @errorName(err) });
        };
        defer prog.deinit(allocator);

        // Reject statically knowable >16-bit integer built-in targets in Create.
        for (prog.builtins.items) |spec| {
            const target_vf: vs.VideoFormat = switch (spec.target) {
                .clip => |cid| zapi.getVideoInfo(nodes[cid]).format,
                .dst => out_vi.format,
            };
            if (target_vf.colorFamily != .Undefined) {
                if (common.colorFamilyFromVs(target_vf.colorFamily) == null) {
                    return common.fail(
                        map_out,
                        filter_name,
                        "plane {d}: unsupported color family for built-in constant '{s}'",
                        .{ p, @tagName(spec.kind) },
                    );
                }
                if (target_vf.sampleType == .Integer and target_vf.bitsPerSample > 16) {
                    return common.fail(
                        map_out,
                        filter_name,
                        "plane {d}: built-in constant '{s}' is unsupported on >16-bit integer clips",
                        .{ p, @tagName(spec.kind) },
                    );
                }
            }
        }

        dyn.plane_exprs[p] = try allocator.dupe(u8, expr_str);
        if (prog.builtins.items.len > 0) {
            dyn.plane_builtins[p] = try allocator.dupe(builtins.BuiltinSpec, prog.builtins.items);
        }
    }

    const deps = try allocator.alloc(vs.FilterDependency, nodes.len);
    defer allocator.free(deps);

    const data = try allocator.create(ExprData);
    owns_nodes = false;
    owns_clip_nf = false;
    owns_dyn = false;
    data.* = .{
        .nodes = nodes,
        .clip_num_frames = clip_num_frames,
        .out_vi = out_vi,
        .profile = profile,
        .mode = .{ .dynamic = dyn },
    };

    for (nodes, clip_num_frames, 0..) |node, nf, i| {
        deps[i] = .{
            .source = node,
            .requestPattern = if (nf >= out_vi.numFrames) .StrictSpatial else .FrameReuseLastOnly,
        };
    }

    zapi.createVideoFilter(out, filter_name, &data.out_vi, exprGetFrame, exprFree, .Parallel, deps, data);
}
