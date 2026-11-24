//! Typed Zig wrapper over the translated AsmJit C ABI (`zasmjit_c`).

const std = @import("std");
const c = @import("zasmjit_c");
const cpu_mod = @import("cpu.zig");

pub const CpuInfo = cpu_mod.CpuInfo;
pub const Isa = cpu_mod.Isa;

pub const ExprPlaneFn = *const fn (
    dst: [*]u8,
    dst_stride: isize,
    srcs: [*]const [*]const u8,
    src_strides: [*]const isize,
    frame_n: isize,
    props: ?[*]const f32,
) callconv(.c) void;

pub const Reg = struct {
    raw: c.zj_reg_t,

    pub fn isValid(self: Reg) bool {
        return self.raw.sig != 0;
    }
};

pub const Label = struct {
    raw: c.zj_label_t,
};

pub const Mem = struct {
    raw: c.zj_mem_t,

    pub fn ptr(base: Reg, disp: i32) Mem {
        return .{ .raw = .{
            .base = base.raw,
            .index = .{ .id = 0, .sig = 0 },
            .disp = disp,
            .shift = 0,
            .has_index = 0,
            .is_label = 0,
            .size = 0,
            .label_id = 0,
        } };
    }

    pub fn indexed(base: Reg, index: Reg, shift: u8, disp: i32) Mem {
        return .{ .raw = .{
            .base = base.raw,
            .index = index.raw,
            .disp = disp,
            .shift = shift,
            .has_index = 1,
            .is_label = 0,
            .size = 0,
            .label_id = 0,
        } };
    }

    pub fn withDisp(self: Mem, delta: i32) Mem {
        var copy = self;
        copy.raw.disp += delta;
        return copy;
    }
};

pub const Cond = enum(u32) {
    eq = c.ZJ_COND_EQ,
    ne = c.ZJ_COND_NE,
    slt = c.ZJ_COND_SLT,
    sle = c.ZJ_COND_SLE,
    sgt = c.ZJ_COND_SGT,
    sge = c.ZJ_COND_SGE,
    ult = c.ZJ_COND_ULT,
    ule = c.ZJ_COND_ULE,
    ugt = c.ZJ_COND_UGT,
    uge = c.ZJ_COND_UGE,
};

pub const GpOp2 = enum(u32) {
    neg = c.ZJ_GP_NEG,
    not = c.ZJ_GP_NOT,
    bswap = c.ZJ_GP_BSWAP,
    clz = c.ZJ_GP_CLZ,
    ctz = c.ZJ_GP_CTZ,
};

pub const GpOp3 = enum(u32) {
    add = c.ZJ_GP_ADD,
    sub = c.ZJ_GP_SUB,
    mul = c.ZJ_GP_MUL,
    @"and" = c.ZJ_GP_AND,
    @"or" = c.ZJ_GP_OR,
    xor = c.ZJ_GP_XOR,
    shl = c.ZJ_GP_SHL,
    shr = c.ZJ_GP_SHR,
    sar = c.ZJ_GP_SAR,
    smin = c.ZJ_GP_SMIN,
    smax = c.ZJ_GP_SMAX,
    umin = c.ZJ_GP_UMIN,
    umax = c.ZJ_GP_UMAX,
};

pub const VecOp2 = enum(u32) {
    mov = c.ZJ_V2_MOV,
    abs_f32 = c.ZJ_V2_ABS_F32,
    neg_f32 = c.ZJ_V2_NEG_F32,
    sqrt_f32 = c.ZJ_V2_SQRT_F32,
    trunc_f32 = c.ZJ_V2_TRUNC_F32,
    floor_f32 = c.ZJ_V2_FLOOR_F32,
    ceil_f32 = c.ZJ_V2_CEIL_F32,
    round_even_f32 = c.ZJ_V2_ROUND_EVEN_F32,
    cvt_i32_to_f32 = c.ZJ_V2_CVT_I32_TO_F32,
    cvt_round_f32_to_i32 = c.ZJ_V2_CVT_ROUND_F32_TO_I32,
    cvt_trunc_f32_to_i32 = c.ZJ_V2_CVT_TRUNC_F32_TO_I32,
    not_u32 = c.ZJ_V2_NOT_U32,
    abs_i16 = c.ZJ_V2_ABS_I16,
    abs_i32 = c.ZJ_V2_ABS_I32,
    neg_i16 = c.ZJ_V2_NEG_I16,
    neg_i32 = c.ZJ_V2_NEG_I32,
    cvt_u8_lo_to_u16 = c.ZJ_V2_CVT_U8_LO_TO_U16,
    cvt_u8_hi_to_u16 = c.ZJ_V2_CVT_U8_HI_TO_U16,
    cvt_u16_lo_to_u32 = c.ZJ_V2_CVT_U16_LO_TO_U32,
    cvt_u16_hi_to_u32 = c.ZJ_V2_CVT_U16_HI_TO_U32,
    cvt_u8_to_u32 = c.ZJ_V2_CVT_U8_TO_U32,
    cvt_i16_lo_to_i32 = c.ZJ_V2_CVT_I16_LO_TO_I32,
    cvt_i16_hi_to_i32 = c.ZJ_V2_CVT_I16_HI_TO_I32,
    broadcast_u8 = c.ZJ_V2_BROADCAST_U8,
    broadcast_u16 = c.ZJ_V2_BROADCAST_U16,
    broadcast_u32 = c.ZJ_V2_BROADCAST_U32,
    broadcast_u64 = c.ZJ_V2_BROADCAST_U64,
    rsqrt_f32 = c.ZJ_V2_RSQRT_F32,
};

pub const VecOp3 = enum(u32) {
    add_f32 = c.ZJ_V3_ADD_F32,
    sub_f32 = c.ZJ_V3_SUB_F32,
    mul_f32 = c.ZJ_V3_MUL_F32,
    div_f32 = c.ZJ_V3_DIV_F32,
    min_f32 = c.ZJ_V3_MIN_F32,
    max_f32 = c.ZJ_V3_MAX_F32,
    cmp_eq_f32 = c.ZJ_V3_CMP_EQ_F32,
    cmp_ne_f32 = c.ZJ_V3_CMP_NE_F32,
    cmp_lt_f32 = c.ZJ_V3_CMP_LT_F32,
    cmp_le_f32 = c.ZJ_V3_CMP_LE_F32,
    cmp_gt_f32 = c.ZJ_V3_CMP_GT_F32,
    cmp_ge_f32 = c.ZJ_V3_CMP_GE_F32,
    and_u32 = c.ZJ_V3_AND_U32,
    or_u32 = c.ZJ_V3_OR_U32,
    xor_u32 = c.ZJ_V3_XOR_U32,
    andn_u32 = c.ZJ_V3_ANDN_U32,
    bic_u32 = c.ZJ_V3_BIC_U32,
    add_i32 = c.ZJ_V3_ADD_I32,
    sub_i32 = c.ZJ_V3_SUB_I32,
    mul_i32 = c.ZJ_V3_MUL_I32,
    min_i32 = c.ZJ_V3_MIN_I32,
    max_i32 = c.ZJ_V3_MAX_I32,
    cmp_eq_i32 = c.ZJ_V3_CMP_EQ_I32,
    cmp_gt_i32 = c.ZJ_V3_CMP_GT_I32,
    add_i16 = c.ZJ_V3_ADD_I16,
    sub_i16 = c.ZJ_V3_SUB_I16,
    adds_i16 = c.ZJ_V3_ADDS_I16,
    subs_i16 = c.ZJ_V3_SUBS_I16,
    adds_u16 = c.ZJ_V3_ADDS_U16,
    subs_u16 = c.ZJ_V3_SUBS_U16,
    mul_i16 = c.ZJ_V3_MUL_I16,
    min_i16 = c.ZJ_V3_MIN_I16,
    max_i16 = c.ZJ_V3_MAX_I16,
    min_u16 = c.ZJ_V3_MIN_U16,
    max_u16 = c.ZJ_V3_MAX_U16,
    cmp_eq_i16 = c.ZJ_V3_CMP_EQ_I16,
    cmp_gt_i16 = c.ZJ_V3_CMP_GT_I16,
    packs_i32_i16 = c.ZJ_V3_PACKS_I32_I16,
    packs_i32_u16 = c.ZJ_V3_PACKS_I32_U16,
    packs_i16_u8 = c.ZJ_V3_PACKS_I16_U8,
    packs_i16_i8 = c.ZJ_V3_PACKS_I16_I8,
    unpack_lo_u8 = c.ZJ_V3_UNPACK_LO_U8,
    unpack_hi_u8 = c.ZJ_V3_UNPACK_HI_U8,
    unpack_lo_u16 = c.ZJ_V3_UNPACK_LO_U16,
    unpack_hi_u16 = c.ZJ_V3_UNPACK_HI_U16,
    unpack_lo_u32 = c.ZJ_V3_UNPACK_LO_U32,
    unpack_hi_u32 = c.ZJ_V3_UNPACK_HI_U32,
    permute_u32 = c.ZJ_V3_PERMUTE_U32,
    add_u8 = c.ZJ_V3_ADD_U8,
    sub_u8 = c.ZJ_V3_SUB_U8,
    adds_u8 = c.ZJ_V3_ADDS_U8,
    subs_u8 = c.ZJ_V3_SUBS_U8,
    min_u8 = c.ZJ_V3_MIN_U8,
    max_u8 = c.ZJ_V3_MAX_U8,
};

pub const VecOp4 = enum(u32) {
    madd_f32 = c.ZJ_V4_MADD_F32,
    msub_f32 = c.ZJ_V4_MSUB_F32,
    nmadd_f32 = c.ZJ_V4_NMADD_F32,
    nmsub_f32 = c.ZJ_V4_NMSUB_F32,
    blendv_u8 = c.ZJ_V4_BLENDV_U8,
};

pub const VecOp2i = enum(u32) {
    sll_u16 = c.ZJ_V2I_SLL_U16,
    srl_u16 = c.ZJ_V2I_SRL_U16,
    sra_i16 = c.ZJ_V2I_SRA_I16,
    sll_u32 = c.ZJ_V2I_SLL_U32,
    srl_u32 = c.ZJ_V2I_SRL_U32,
    sra_i32 = c.ZJ_V2I_SRA_I32,
    swizzle_u32x4 = c.ZJ_V2I_SWIZZLE_U32X4,
};

pub const VecLoad = enum(u32) {
    a_vec = c.ZJ_VLOAD_A_VEC,
    u_vec = c.ZJ_VLOAD_U_VEC,
    u32 = c.ZJ_VLOAD_U_32,
    u64 = c.ZJ_VLOAD_U_64,
    u128 = c.ZJ_VLOAD_U_128,
    u256 = c.ZJ_VLOAD_U_256,
    a128 = c.ZJ_VLOAD_A_128,
    cvt_u8_to_u16 = c.ZJ_VLOAD_CVT_U8_TO_U16,
    cvt_u8_to_u32 = c.ZJ_VLOAD_CVT_U8_TO_U32,
    cvt_u16_to_u32 = c.ZJ_VLOAD_CVT_U16_TO_U32,
    broadcast_u32 = c.ZJ_VLOAD_BROADCAST_U32,
    broadcast_u64 = c.ZJ_VLOAD_BROADCAST_U64,
};

pub const VecStore = enum(u32) {
    a_vec = c.ZJ_VSTORE_A_VEC,
    u_vec = c.ZJ_VSTORE_U_VEC,
    u32 = c.ZJ_VSTORE_U_32,
    u64 = c.ZJ_VSTORE_U_64,
    u128 = c.ZJ_VSTORE_U_128,
    u256 = c.ZJ_VSTORE_U_256,
    a128 = c.ZJ_VSTORE_A_128,
};

pub const JitError = error{
    AbiMismatch,
    OutOfMemory,
    ExecMemDenied,
    CompileFailed,
};

pub const CompiledKernel = struct {
    func: ExprPlaneFn,
    code_size: usize,
};

pub const Runtime = struct {
    raw: *c.zj_runtime,

    pub fn init() JitError!Runtime {
        if (c.zj_abi_version() != c.ZJ_ABI_VERSION) {
            return error.AbiMismatch;
        }
        const raw = c.zj_runtime_create() orelse return error.OutOfMemory;
        return .{ .raw = raw };
    }

    pub fn deinit(self: *Runtime) void {
        c.zj_runtime_destroy(self.raw);
        self.* = undefined;
    }

    pub fn release(self: *Runtime, func: ExprPlaneFn) void {
        c.zj_runtime_release(self.raw, @ptrFromInt(@intFromPtr(func)));
    }
};

pub const Builder = struct {
    raw: *c.zj_ctx,
    cpu: CpuInfo,

    pub fn init(rt: *Runtime, cpu: CpuInfo, capture_disasm: bool) JitError!Builder {
        const c_cpu = cpu.toC();
        const raw = c.zj_ctx_create(rt.raw, &c_cpu, @intFromBool(capture_disasm)) orelse return error.OutOfMemory;
        return .{ .raw = raw, .cpu = cpu };
    }

    pub fn deinit(self: *Builder) void {
        c.zj_ctx_destroy(self.raw);
        self.* = undefined;
    }

    pub fn beginFunc(self: *Builder) JitError!void {
        if (c.zj_ctx_begin_func(self.raw) != 0) return error.CompileFailed;
    }

    pub fn argGp(self: *Builder, index: u32) Reg {
        return .{ .raw = c.zj_ctx_arg_gp(self.raw, index) };
    }

    pub fn retVoid(self: *Builder) void {
        c.zj_ctx_ret_void(self.raw);
    }

    pub fn endFunc(self: *Builder) void {
        c.zj_ctx_end_func(self.raw);
    }

    pub fn finalize(self: *Builder) JitError!CompiledKernel {
        var fn_ptr: ?*anyopaque = null;
        var code_size: usize = 0;
        var execmem_denied: u8 = 0;
        const rc = c.zj_ctx_finalize(self.raw, &fn_ptr, &code_size, &execmem_denied);
        if (rc != 0) {
            if (execmem_denied != 0) return error.ExecMemDenied;
            return error.CompileFailed;
        }
        const ptr = fn_ptr orelse return error.CompileFailed;
        return .{
            .func = @ptrCast(@alignCast(ptr)),
            .code_size = code_size,
        };
    }

    pub fn disasm(self: *const Builder) []const u8 {
        var ptr: [*c]const u8 = null;
        var len: usize = 0;
        c.zj_ctx_get_disasm(self.raw, &ptr, &len);
        if (ptr == null or len == 0) return "";
        return ptr[0..len];
    }

    pub fn lastError(self: *const Builder) u32 {
        return c.zj_ctx_last_error(self.raw);
    }

    pub fn newGp64(self: *Builder) Reg {
        return .{ .raw = c.zj_new_gp64(self.raw) };
    }

    pub fn newGp32(self: *Builder) Reg {
        return .{ .raw = c.zj_new_gp32(self.raw) };
    }

    pub fn newVec(self: *Builder) Reg {
        return .{ .raw = c.zj_new_vec(self.raw) };
    }

    pub fn newVec128(self: *Builder) Reg {
        return .{ .raw = c.zj_new_vec128(self.raw) };
    }

    pub fn newLabel(self: *Builder) Label {
        return .{ .raw = c.zj_new_label(self.raw) };
    }

    pub fn bind(self: *Builder, label: Label) void {
        c.zj_bind(self.raw, label.raw);
    }

    pub fn constF32Splat(self: *Builder, val: f32) Reg {
        return .{ .raw = c.zj_const_f32_splat(self.raw, val) };
    }

    pub fn constI32Splat(self: *Builder, val: i32) Reg {
        return .{ .raw = c.zj_const_i32_splat(self.raw, val) };
    }

    pub fn constBytes(self: *Builder, data: []const u8) Reg {
        return .{ .raw = c.zj_const_bytes(self.raw, data.ptr, @intCast(data.len)) };
    }

    pub fn vecZero(self: *Builder) Reg {
        return .{ .raw = c.zj_vec_zero(self.raw) };
    }

    pub fn gpMov(self: *Builder, dst: Reg, src: Reg) void {
        c.zj_gp_mov(self.raw, dst.raw, src.raw);
    }

    pub fn gpMovImm(self: *Builder, dst: Reg, imm: i64) void {
        c.zj_gp_mov_imm(self.raw, dst.raw, imm);
    }

    pub fn gpLoad(self: *Builder, dst: Reg, mem: Mem, bytes: u32, sign_extend: bool) void {
        c.zj_gp_load(self.raw, dst.raw, mem.raw, bytes, @intFromBool(sign_extend));
    }

    pub fn gpStore(self: *Builder, mem: Mem, src: Reg, bytes: u32) void {
        c.zj_gp_store(self.raw, mem.raw, src.raw, bytes);
    }

    pub fn gpOp2(self: *Builder, op: GpOp2, dst: Reg, src: Reg) void {
        c.zj_gp_op2(self.raw, @backingInt(op), dst.raw, src.raw);
    }

    pub fn gpOp3(self: *Builder, op: GpOp3, dst: Reg, a: Reg, b: Reg) void {
        c.zj_gp_op3(self.raw, @backingInt(op), dst.raw, a.raw, b.raw);
    }

    pub fn gpOp3Imm(self: *Builder, op: GpOp3, dst: Reg, a: Reg, imm: i64) void {
        c.zj_gp_op3_imm(self.raw, @backingInt(op), dst.raw, a.raw, imm);
    }

    pub fn gpCmov(self: *Builder, cond: Cond, dst: Reg, lhs: Reg, rhs: Reg, src: Reg) void {
        c.zj_gp_cmov(self.raw, @backingInt(cond), dst.raw, lhs.raw, rhs.raw, src.raw);
    }

    pub fn gpSelect(self: *Builder, cond: Cond, dst: Reg, lhs: Reg, rhs: Reg, true_val: Reg, false_val: Reg) void {
        c.zj_gp_select(self.raw, @backingInt(cond), dst.raw, lhs.raw, rhs.raw, true_val.raw, false_val.raw);
    }

    pub fn jmp(self: *Builder, target: Label) void {
        c.zj_jmp(self.raw, target.raw);
    }

    pub fn jcc(self: *Builder, cond: Cond, lhs: Reg, rhs: Reg, target: Label) void {
        c.zj_jcc(self.raw, @backingInt(cond), lhs.raw, rhs.raw, target.raw);
    }

    pub fn jccImm(self: *Builder, cond: Cond, lhs: Reg, rhs_imm: i64, target: Label) void {
        c.zj_jcc_imm(self.raw, @backingInt(cond), lhs.raw, rhs_imm, target.raw);
    }

    pub fn vecOp2(self: *Builder, op: VecOp2, dst: Reg, src: Reg) void {
        c.zj_vec_op2(self.raw, @backingInt(op), dst.raw, src.raw);
    }

    pub fn vecOp3(self: *Builder, op: VecOp3, dst: Reg, a: Reg, b: Reg) void {
        c.zj_vec_op3(self.raw, @backingInt(op), dst.raw, a.raw, b.raw);
    }

    pub fn vecOp4(self: *Builder, op: VecOp4, dst: Reg, a: Reg, b: Reg, c_reg: Reg) void {
        c.zj_vec_op4(self.raw, @backingInt(op), dst.raw, a.raw, b.raw, c_reg.raw);
    }

    pub fn vecOp2i(self: *Builder, op: VecOp2i, dst: Reg, src: Reg, imm: u32) void {
        c.zj_vec_op2i(self.raw, @backingInt(op), dst.raw, src.raw, imm);
    }

    pub fn vecLoad(self: *Builder, op: VecLoad, dst: Reg, mem: Mem) void {
        c.zj_vec_load(self.raw, @backingInt(op), dst.raw, mem.raw);
    }

    pub fn vecStore(self: *Builder, op: VecStore, mem: Mem, src: Reg) void {
        c.zj_vec_store(self.raw, @backingInt(op), mem.raw, src.raw);
    }

    pub fn vecFromGp32(self: *Builder, dst: Reg, gp32: Reg) void {
        c.zj_vec_from_gp32(self.raw, dst.raw, gp32.raw);
    }

    pub fn vecBroadcastGp32(self: *Builder, dst: Reg, gp32: Reg) void {
        c.zj_vec_broadcast_gp32(self.raw, dst.raw, gp32.raw);
    }

    pub fn gp32FromVec(self: *Builder, gp32: Reg, vec: Reg) void {
        c.zj_gp32_from_vec(self.raw, gp32.raw, vec.raw);
    }

    pub fn vecCvtF16ToF32(self: *Builder, dst: Reg, mem: Mem) void {
        c.zj_vec_cvt_f16_to_f32(self.raw, dst.raw, mem.raw);
    }

    pub fn vecCvtF16ToF32x2(self: *Builder, dst0: Reg, dst1: Reg, mem: Mem) void {
        c.zj_vec_cvt_f16_to_f32x2(self.raw, dst0.raw, dst1.raw, mem.raw);
    }

    pub fn vecCvtF32ToF16(self: *Builder, mem: Mem, src: Reg) void {
        c.zj_vec_cvt_f32_to_f16(self.raw, mem.raw, src.raw);
    }

    pub fn vecCvtF32ToF16x2(self: *Builder, mem: Mem, src0: Reg, src1: Reg) void {
        c.zj_vec_cvt_f32_to_f16x2(self.raw, mem.raw, src0.raw, src1.raw);
    }

    pub fn vecLoadUVecx2(self: *Builder, dst0: Reg, dst1: Reg, mem: Mem) void {
        c.zj_vec_load_u_vecx2(self.raw, dst0.raw, dst1.raw, mem.raw);
    }

    pub fn vecStoreUVecx2(self: *Builder, mem: Mem, src0: Reg, src1: Reg) void {
        c.zj_vec_store_u_vecx2(self.raw, mem.raw, src0.raw, src1.raw);
    }

    pub fn compileLut1dU8Neon(
        rt: *Runtime,
        table_256: *const [256]u8,
        clip_idx: u32,
        width: usize,
        height: usize,
    ) JitError!CompiledKernel {
        var fn_ptr: ?*anyopaque = null;
        var code_size: usize = 0;
        var execmem_denied: u8 = 0;
        const rc = c.zj_compile_lut1d_u8_neon(
            rt.raw,
            table_256,
            clip_idx,
            @intCast(width),
            @intCast(height),
            &fn_ptr,
            &code_size,
            &execmem_denied,
        );
        if (rc != 0) {
            if (execmem_denied != 0) return error.ExecMemDenied;
            return error.CompileFailed;
        }
        const ptr = fn_ptr orelse return error.CompileFailed;
        return .{
            .func = @ptrCast(@alignCast(ptr)),
            .code_size = code_size,
        };
    }

    pub fn vecStoreU8FromI32(self: *Builder, mem: Mem, src_i32: Reg) void {
        c.zj_vec_store_u8_from_i32(self.raw, mem.raw, src_i32.raw);
    }

    pub fn vecStoreU8FromI32x2(self: *Builder, mem: Mem, src0_i32: Reg, src1_i32: Reg) void {
        c.zj_vec_store_u8_from_i32x2(self.raw, mem.raw, src0_i32.raw, src1_i32.raw);
    }

    pub fn vecStoreU8FromI16(self: *Builder, mem: Mem, src_i16: Reg) void {
        c.zj_vec_store_u8_from_i16(self.raw, mem.raw, src_i16.raw);
    }

    pub fn vecStoreU8FromI16x2(self: *Builder, mem: Mem, src0_i16: Reg, src1_i16: Reg) void {
        c.zj_vec_store_u8_from_i16x2(self.raw, mem.raw, src0_i16.raw, src1_i16.raw);
    }

    pub fn vecStoreU16FromI32(self: *Builder, mem: Mem, src_i32: Reg) void {
        c.zj_vec_store_u16_from_i32(self.raw, mem.raw, src_i32.raw);
    }

    pub fn vecStoreU16FromI32x2(self: *Builder, mem: Mem, src0_i32: Reg, src1_i32: Reg) void {
        c.zj_vec_store_u16_from_i32x2(self.raw, mem.raw, src0_i32.raw, src1_i32.raw);
    }

    pub fn cpuSupportsFastLutU8(cpu: CpuInfo) bool {
        const c_cpu = cpu.toC();
        return c.zj_cpu_supports_fast_lut_u8(&c_cpu) != 0;
    }

    pub fn vecLutU8_256(self: *Builder, dst: Reg, idx: Reg, table_256: *const [256]u8) void {
        c.zj_vec_lut_u8_256(self.raw, dst.raw, idx.raw, table_256);
    }

    pub fn vecMulCF32(self: *Builder, dst: Reg, a: Reg, b_const: f32) void {
        c.zj_vec_mul_c_f32(self.raw, dst.raw, a.raw, b_const);
    }

    pub fn vecAddCF32(self: *Builder, dst: Reg, a: Reg, b_const: f32) void {
        c.zj_vec_add_c_f32(self.raw, dst.raw, a.raw, b_const);
    }

    pub fn vecSubCF32(self: *Builder, dst: Reg, a: Reg, b_const: f32) void {
        c.zj_vec_sub_c_f32(self.raw, dst.raw, a.raw, b_const);
    }

    pub fn vecMinCF32(self: *Builder, dst: Reg, a: Reg, b_const: f32) void {
        c.zj_vec_min_c_f32(self.raw, dst.raw, a.raw, b_const);
    }

    pub fn vecMaxCF32(self: *Builder, dst: Reg, a: Reg, b_const: f32) void {
        c.zj_vec_max_c_f32(self.raw, dst.raw, a.raw, b_const);
    }

    pub fn vecCmpCF32(self: *Builder, cmp_op: VecOp3, dst: Reg, a: Reg, b_const: f32) void {
        c.zj_vec_cmp_c_f32(self.raw, @backingInt(cmp_op), dst.raw, a.raw, b_const);
    }

    pub fn vecFmaACF32(self: *Builder, dst: Reg, a: Reg, b_const: f32, c_reg: Reg) void {
        c.zj_vec_fma_ac_f32(self.raw, dst.raw, a.raw, b_const, c_reg.raw);
    }

    pub fn vecBlendVF32(self: *Builder, dst: Reg, false_val: Reg, true_val: Reg, mask: Reg) void {
        c.zj_vec_blendv_f32(self.raw, dst.raw, false_val.raw, true_val.raw, mask.raw);
    }

    pub fn vecAndF32(self: *Builder, dst: Reg, a: Reg, b_reg: Reg) void {
        c.zj_vec_and_f32(self.raw, dst.raw, a.raw, b_reg.raw);
    }

    pub fn vecAndnF32(self: *Builder, dst: Reg, mask_not: Reg, b_reg: Reg) void {
        c.zj_vec_andn_f32(self.raw, dst.raw, mask_not.raw, b_reg.raw);
    }

    pub fn vecFmaBCF32(self: *Builder, dst: Reg, a: Reg, b_reg: Reg, c_const: f32) void {
        c.zj_vec_fma_bc_f32(self.raw, dst.raw, a.raw, b_reg.raw, c_const);
    }

    pub fn vecFmaCCF32(self: *Builder, dst: Reg, a: Reg, b_const: f32, c_const: f32) void {
        c.zj_vec_fma_cc_f32(self.raw, dst.raw, a.raw, b_const, c_const);
    }

    pub fn vecFnmaBCF32(self: *Builder, dst: Reg, a: Reg, b_const: f32, c_reg: Reg) void {
        c.zj_vec_fnma_bc_f32(self.raw, dst.raw, a.raw, b_const, c_reg.raw);
    }

    pub fn vecPermuteF32(self: *Builder, dst: Reg, src: Reg, lane_indices: []const u32) void {
        c.zj_vec_permute_f32(self.raw, dst.raw, src.raw, lane_indices.ptr, @intCast(lane_indices.len));
    }

    pub fn vecGatherF32(self: *Builder, dst: Reg, base: Reg, byte_offsets_i32: Reg, can_use_hw_gather: bool) void {
        c.zj_vec_gather_f32(self.raw, dst.raw, base.raw, byte_offsets_i32.raw, @intFromBool(can_use_hw_gather));
    }

    pub fn vecGatherF16(self: *Builder, dst_f32: Reg, base: Reg, byte_offsets_i32: Reg) void {
        c.zj_vec_gather_f16(self.raw, dst_f32.raw, base.raw, byte_offsets_i32.raw);
    }

    pub fn vecGatherU16(self: *Builder, dst_i32: Reg, base: Reg, byte_offsets_i32: Reg) void {
        c.zj_vec_gather_u16(self.raw, dst_i32.raw, base.raw, byte_offsets_i32.raw);
    }

    pub fn vecGatherU8(self: *Builder, dst_i32: Reg, base: Reg, byte_offsets_i32: Reg) void {
        c.zj_vec_gather_u8(self.raw, dst_i32.raw, base.raw, byte_offsets_i32.raw);
    }
};

// Native C++ ABI probe symbols (compiled in test builds only).
extern fn zj_probe_reg_layout(count: *usize) [*]const usize;
extern fn zj_probe_label_layout(count: *usize) [*]const usize;
extern fn zj_probe_mem_layout(count: *usize) [*]const usize;
extern fn zj_probe_cpu_info_layout(count: *usize) [*]const usize;
extern fn zj_probe_echo_reg(r: c.zj_reg_t) c.zj_reg_t;
extern fn zj_probe_echo_mem(m: c.zj_mem_t) c.zj_mem_t;

test "native C++ ABI layout matches translated zasmjit_c types" {
    try std.testing.expectEqual(c.ZJ_ABI_VERSION, c.zj_abi_version());

    {
        var n: usize = 0;
        const p = zj_probe_reg_layout(&n);
        try std.testing.expectEqual(4, n);
        try std.testing.expectEqual(p[0], @sizeOf(c.zj_reg_t));
        try std.testing.expectEqual(p[1], @alignOf(c.zj_reg_t));
        try std.testing.expectEqual(p[2], @offsetOf(c.zj_reg_t, "id"));
        try std.testing.expectEqual(p[3], @offsetOf(c.zj_reg_t, "sig"));
    }

    {
        var n: usize = 0;
        const p = zj_probe_label_layout(&n);
        try std.testing.expectEqual(3, n);
        try std.testing.expectEqual(p[0], @sizeOf(c.zj_label_t));
        try std.testing.expectEqual(p[1], @alignOf(c.zj_label_t));
        try std.testing.expectEqual(p[2], @offsetOf(c.zj_label_t, "id"));
    }

    {
        var n: usize = 0;
        const p = zj_probe_mem_layout(&n);
        try std.testing.expectEqual(10, n);
        try std.testing.expectEqual(p[0], @sizeOf(c.zj_mem_t));
        try std.testing.expectEqual(p[1], @alignOf(c.zj_mem_t));
        try std.testing.expectEqual(p[2], @offsetOf(c.zj_mem_t, "base"));
        try std.testing.expectEqual(p[3], @offsetOf(c.zj_mem_t, "index"));
        try std.testing.expectEqual(p[4], @offsetOf(c.zj_mem_t, "disp"));
        try std.testing.expectEqual(p[5], @offsetOf(c.zj_mem_t, "shift"));
        try std.testing.expectEqual(p[6], @offsetOf(c.zj_mem_t, "has_index"));
        try std.testing.expectEqual(p[7], @offsetOf(c.zj_mem_t, "is_label"));
        try std.testing.expectEqual(p[8], @offsetOf(c.zj_mem_t, "size"));
        try std.testing.expectEqual(p[9], @offsetOf(c.zj_mem_t, "label_id"));
    }

    {
        var n: usize = 0;
        const p = zj_probe_cpu_info_layout(&n);
        try std.testing.expectEqual(12, n);
        try std.testing.expectEqual(p[0], @sizeOf(c.zj_cpu_info_t));
        try std.testing.expectEqual(p[1], @alignOf(c.zj_cpu_info_t));
        try std.testing.expectEqual(p[2], @offsetOf(c.zj_cpu_info_t, "isa"));
        try std.testing.expectEqual(p[3], @offsetOf(c.zj_cpu_info_t, "vec_bytes"));
        try std.testing.expectEqual(p[4], @offsetOf(c.zj_cpu_info_t, "f32_lanes"));
        try std.testing.expectEqual(p[5], @offsetOf(c.zj_cpu_info_t, "has_ssse3"));
        try std.testing.expectEqual(p[6], @offsetOf(c.zj_cpu_info_t, "has_sse4_1"));
        try std.testing.expectEqual(p[7], @offsetOf(c.zj_cpu_info_t, "has_fma"));
        try std.testing.expectEqual(p[8], @offsetOf(c.zj_cpu_info_t, "has_f16c"));
        try std.testing.expectEqual(p[9], @offsetOf(c.zj_cpu_info_t, "has_avx2"));
        try std.testing.expectEqual(p[10], @offsetOf(c.zj_cpu_info_t, "has_avx512"));
        try std.testing.expectEqual(p[11], @offsetOf(c.zj_cpu_info_t, "prefer_hw_gather_f32"));
    }

    // By-value struct round trips across the C++ <-> Zig boundary.
    const r_in: c.zj_reg_t = .{ .id = 0x1234_5678, .sig = 0x9ABC_DEF0 };
    try std.testing.expectEqual(r_in, zj_probe_echo_reg(r_in));

    const m_in: c.zj_mem_t = .{
        .base = .{ .id = 7, .sig = 11 },
        .index = .{ .id = 13, .sig = 17 },
        .disp = -128,
        .shift = 2,
        .has_index = 1,
        .is_label = 0,
        .size = 16,
        .label_id = 42,
    };
    try std.testing.expectEqual(m_in, zj_probe_echo_mem(m_in));
}

fn runSmokeKernelForCpu(cpu: CpuInfo) !void {
    if (cpu.isa == .scalar) return;

    var rt = try Runtime.init();
    defer rt.deinit();

    var b = try Builder.init(&rt, cpu, true);
    defer b.deinit();

    try b.beginFunc();
    const dst_ptr = b.argGp(0);
    _ = b.argGp(1); // dst_stride
    const srcs_ptr = b.argGp(2);
    _ = b.argGp(3); // src_strides
    _ = b.argGp(4); // frame_n
    _ = b.argGp(5); // props

    const src0 = b.newGp64();
    b.gpLoad(src0, Mem.ptr(srcs_ptr, 0), 8, false);

    // Exercise custom splat constants, dedup, and base+index*(1<<2)+disp addressing
    // (which exercises AArch64 Mem legalisation in zasmjit.cpp).
    const c_scale = b.constF32Splat(2.5);
    const c_scale_dup = b.constF32Splat(2.5);
    try std.testing.expectEqual(c_scale.raw, c_scale_dup.raw);
    const c_bias = b.constF32Splat(-1.25);

    const idx = b.newGp64();
    b.gpMovImm(idx, 1);

    // Load vector starting at element index 2 via base + idx*(1<<2) + 4 bytes (total +8 bytes = +2 floats).
    const v = b.newVec();
    b.vecLoad(.u_vec, v, Mem.indexed(src0, idx, 2, 4));
    b.vecOp3(.mul_f32, v, v, c_scale);
    b.vecOp3(.add_f32, v, v, c_bias);
    b.vecStore(.u_vec, Mem.ptr(dst_ptr, 0), v);

    b.retVoid();
    b.endFunc();

    const kernel = try b.finalize();
    defer rt.release(kernel.func);

    try std.testing.expect(kernel.code_size > 0);
    try std.testing.expect(b.disasm().len > 0);

    var src_buf: [20]f32 align(64) = undefined;
    for (&src_buf, 0..) |*slot, i| {
        slot.* = @floatFromInt(i);
    }
    var dst_buf: [16]f32 align(64) = @splat(-999.0);

    const src_ptrs = [_][*]const u8{@ptrCast(&src_buf)};
    const src_strides = [_]isize{@sizeOf(@TypeOf(src_buf))};

    kernel.func(
        @ptrCast(&dst_buf),
        @sizeOf(@TypeOf(dst_buf)),
        &src_ptrs,
        &src_strides,
        0,
        null,
    );

    for (0..cpu.f32_lanes) |i| {
        const expected: f32 = src_buf[i + 2] * 2.5 - 1.25;
        try std.testing.expectEqual(expected, dst_buf[i]);
    }
}

test "JIT smoke kernel executes across supported ISA tiers" {
    switch (cpu_mod.IsaFamily.host()) {
        .x86 => {
            const host = cpu_mod.detectWithOverride(null);
            // Always test SSE2 baseline (level 1), plus v2/v3 if the host supports them.
            try runSmokeKernelForCpu(cpu_mod.detectWithOverride(1));
            if (@backingInt(host.isa) >= @backingInt(Isa.v2)) {
                try runSmokeKernelForCpu(cpu_mod.detectWithOverride(2));
            }
            if (@backingInt(host.isa) >= @backingInt(Isa.v3)) {
                try runSmokeKernelForCpu(cpu_mod.detectWithOverride(3));
            }
        },
        .aarch64 => {
            try runSmokeKernelForCpu(cpu_mod.detectWithOverride(null));
        },
    }
}
