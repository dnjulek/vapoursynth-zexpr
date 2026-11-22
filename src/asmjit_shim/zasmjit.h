#ifndef ZEXPR_ZASMJIT_H
#define ZEXPR_ZASMJIT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ZJ_ABI_VERSION 1u

typedef struct zj_runtime zj_runtime;
typedef struct zj_ctx zj_ctx;

typedef struct {
    uint32_t id;
    uint32_t sig;
} zj_reg_t;

typedef struct {
    uint32_t id;
} zj_label_t;

typedef struct {
    zj_reg_t base;
    zj_reg_t index;
    int32_t disp;
    uint8_t shift;     /* 0, 1, 2, 3 */
    uint8_t has_index; /* 0 or 1 */
    uint8_t is_label;  /* 0 or 1 */
    uint8_t size;      /* 0 = implicit, else 1/2/4/8/16/32/64 */
    uint32_t label_id;
} zj_mem_t;

/* Explicit ISA tiers (fixed-width, no C enum ABI ambiguity). */
typedef uint32_t zj_isa_t;
#define ZJ_ISA_SCALAR ((zj_isa_t)0u)
#define ZJ_ISA_SSE2   ((zj_isa_t)1u)
#define ZJ_ISA_V2     ((zj_isa_t)2u)
#define ZJ_ISA_V3     ((zj_isa_t)3u)
#define ZJ_ISA_V4     ((zj_isa_t)4u)
#define ZJ_ISA_NEON   ((zj_isa_t)5u)

typedef struct {
    zj_isa_t isa;
    uint32_t vec_bytes; /* 16, 32, or 64 (0 for scalar) */
    uint32_t f32_lanes; /* 4, 8, or 16 (0 for scalar) */
    uint8_t has_ssse3;
    uint8_t has_sse4_1;
    uint8_t has_fma;
    uint8_t has_f16c;
    uint8_t has_avx2;
    uint8_t has_avx512;
    uint8_t prefer_hw_gather_f32;
    uint8_t _pad[1];
} zj_cpu_info_t;

/* Kernel function signature: ZJ_SIG_EXPR_PLANE. */
typedef void (*zj_expr_plane_fn)(
    uint8_t* dst,
    intptr_t dst_stride,
    const uint8_t* const* srcs,
    const intptr_t* src_strides,
    intptr_t frame_n,
    const float* props
);

/* GP condition codes (fixed-width). */
typedef uint32_t zj_cond_t;
#define ZJ_COND_EQ  ((zj_cond_t)0u)
#define ZJ_COND_NE  ((zj_cond_t)1u)
#define ZJ_COND_SLT ((zj_cond_t)2u)
#define ZJ_COND_SLE ((zj_cond_t)3u)
#define ZJ_COND_SGT ((zj_cond_t)4u)
#define ZJ_COND_SGE ((zj_cond_t)5u)
#define ZJ_COND_ULT ((zj_cond_t)6u)
#define ZJ_COND_ULE ((zj_cond_t)7u)
#define ZJ_COND_UGT ((zj_cond_t)8u)
#define ZJ_COND_UGE ((zj_cond_t)9u)

/* GP unary ops. */
typedef uint32_t zj_gp_op2_t;
#define ZJ_GP_NEG   ((zj_gp_op2_t)0u)
#define ZJ_GP_NOT   ((zj_gp_op2_t)1u)
#define ZJ_GP_BSWAP ((zj_gp_op2_t)2u)
#define ZJ_GP_CLZ   ((zj_gp_op2_t)3u)
#define ZJ_GP_CTZ   ((zj_gp_op2_t)4u)

/* GP binary ops. */
typedef uint32_t zj_gp_op3_t;
#define ZJ_GP_ADD  ((zj_gp_op3_t)0u)
#define ZJ_GP_SUB  ((zj_gp_op3_t)1u)
#define ZJ_GP_MUL  ((zj_gp_op3_t)2u)
#define ZJ_GP_AND  ((zj_gp_op3_t)3u)
#define ZJ_GP_OR   ((zj_gp_op3_t)4u)
#define ZJ_GP_XOR  ((zj_gp_op3_t)5u)
#define ZJ_GP_SHL  ((zj_gp_op3_t)6u)
#define ZJ_GP_SHR  ((zj_gp_op3_t)7u)
#define ZJ_GP_SAR  ((zj_gp_op3_t)8u)
#define ZJ_GP_SMIN ((zj_gp_op3_t)9u)
#define ZJ_GP_SMAX ((zj_gp_op3_t)10u)
#define ZJ_GP_UMIN ((zj_gp_op3_t)11u)
#define ZJ_GP_UMAX ((zj_gp_op3_t)12u)

/* Vector unary ops (UniOpVV). */
typedef uint32_t zj_vec_op2_t;
#define ZJ_V2_MOV                  ((zj_vec_op2_t)0u)
#define ZJ_V2_ABS_F32              ((zj_vec_op2_t)1u)
#define ZJ_V2_NEG_F32              ((zj_vec_op2_t)2u)
#define ZJ_V2_SQRT_F32             ((zj_vec_op2_t)3u)
#define ZJ_V2_TRUNC_F32            ((zj_vec_op2_t)4u)
#define ZJ_V2_FLOOR_F32            ((zj_vec_op2_t)5u)
#define ZJ_V2_CEIL_F32             ((zj_vec_op2_t)6u)
#define ZJ_V2_ROUND_EVEN_F32       ((zj_vec_op2_t)7u)
#define ZJ_V2_CVT_I32_TO_F32       ((zj_vec_op2_t)8u)
#define ZJ_V2_CVT_ROUND_F32_TO_I32 ((zj_vec_op2_t)9u)
#define ZJ_V2_CVT_TRUNC_F32_TO_I32 ((zj_vec_op2_t)10u)
#define ZJ_V2_NOT_U32              ((zj_vec_op2_t)11u)
#define ZJ_V2_ABS_I16              ((zj_vec_op2_t)12u)
#define ZJ_V2_ABS_I32              ((zj_vec_op2_t)13u)
#define ZJ_V2_NEG_I16              ((zj_vec_op2_t)14u)
#define ZJ_V2_NEG_I32              ((zj_vec_op2_t)15u)
#define ZJ_V2_CVT_U8_LO_TO_U16     ((zj_vec_op2_t)16u)
#define ZJ_V2_CVT_U8_HI_TO_U16     ((zj_vec_op2_t)17u)
#define ZJ_V2_CVT_U16_LO_TO_U32    ((zj_vec_op2_t)18u)
#define ZJ_V2_CVT_U16_HI_TO_U32    ((zj_vec_op2_t)19u)
#define ZJ_V2_CVT_U8_TO_U32        ((zj_vec_op2_t)20u)
#define ZJ_V2_CVT_I16_LO_TO_I32    ((zj_vec_op2_t)21u)
#define ZJ_V2_CVT_I16_HI_TO_I32    ((zj_vec_op2_t)22u)
#define ZJ_V2_BROADCAST_U8         ((zj_vec_op2_t)23u)
#define ZJ_V2_BROADCAST_U16        ((zj_vec_op2_t)24u)
#define ZJ_V2_BROADCAST_U32        ((zj_vec_op2_t)25u)
#define ZJ_V2_BROADCAST_U64        ((zj_vec_op2_t)26u)
#define ZJ_V2_RSQRT_F32            ((zj_vec_op2_t)27u)

/* Vector binary ops (UniOpVVV). Note: kModF32 is intentionally omitted (AsmJit 1.23.0 bug). */
typedef uint32_t zj_vec_op3_t;
#define ZJ_V3_ADD_F32       ((zj_vec_op3_t)0u)
#define ZJ_V3_SUB_F32       ((zj_vec_op3_t)1u)
#define ZJ_V3_MUL_F32       ((zj_vec_op3_t)2u)
#define ZJ_V3_DIV_F32       ((zj_vec_op3_t)3u)
#define ZJ_V3_MIN_F32       ((zj_vec_op3_t)4u)
#define ZJ_V3_MAX_F32       ((zj_vec_op3_t)5u)
#define ZJ_V3_CMP_EQ_F32    ((zj_vec_op3_t)6u)
#define ZJ_V3_CMP_NE_F32    ((zj_vec_op3_t)7u)
#define ZJ_V3_CMP_LT_F32    ((zj_vec_op3_t)8u)
#define ZJ_V3_CMP_LE_F32    ((zj_vec_op3_t)9u)
#define ZJ_V3_CMP_GT_F32    ((zj_vec_op3_t)10u)
#define ZJ_V3_CMP_GE_F32    ((zj_vec_op3_t)11u)
#define ZJ_V3_AND_U32       ((zj_vec_op3_t)12u)
#define ZJ_V3_OR_U32        ((zj_vec_op3_t)13u)
#define ZJ_V3_XOR_U32       ((zj_vec_op3_t)14u)
#define ZJ_V3_ANDN_U32      ((zj_vec_op3_t)15u)
#define ZJ_V3_BIC_U32       ((zj_vec_op3_t)16u)
#define ZJ_V3_ADD_I32       ((zj_vec_op3_t)17u)
#define ZJ_V3_SUB_I32       ((zj_vec_op3_t)18u)
#define ZJ_V3_MUL_I32       ((zj_vec_op3_t)19u)
#define ZJ_V3_MIN_I32       ((zj_vec_op3_t)20u)
#define ZJ_V3_MAX_I32       ((zj_vec_op3_t)21u)
#define ZJ_V3_CMP_EQ_I32    ((zj_vec_op3_t)22u)
#define ZJ_V3_CMP_GT_I32    ((zj_vec_op3_t)23u)
#define ZJ_V3_ADD_I16       ((zj_vec_op3_t)24u)
#define ZJ_V3_SUB_I16       ((zj_vec_op3_t)25u)
#define ZJ_V3_ADDS_I16      ((zj_vec_op3_t)26u)
#define ZJ_V3_SUBS_I16      ((zj_vec_op3_t)27u)
#define ZJ_V3_ADDS_U16      ((zj_vec_op3_t)28u)
#define ZJ_V3_SUBS_U16      ((zj_vec_op3_t)29u)
#define ZJ_V3_MUL_I16       ((zj_vec_op3_t)30u)
#define ZJ_V3_MIN_I16       ((zj_vec_op3_t)31u)
#define ZJ_V3_MAX_I16       ((zj_vec_op3_t)32u)
#define ZJ_V3_MIN_U16       ((zj_vec_op3_t)33u)
#define ZJ_V3_MAX_U16       ((zj_vec_op3_t)34u)
#define ZJ_V3_CMP_EQ_I16    ((zj_vec_op3_t)35u)
#define ZJ_V3_CMP_GT_I16    ((zj_vec_op3_t)36u)
#define ZJ_V3_PACKS_I32_I16 ((zj_vec_op3_t)37u)
#define ZJ_V3_PACKS_I32_U16 ((zj_vec_op3_t)38u)
#define ZJ_V3_PACKS_I16_U8  ((zj_vec_op3_t)39u)
#define ZJ_V3_PACKS_I16_I8  ((zj_vec_op3_t)40u)
#define ZJ_V3_UNPACK_LO_U8  ((zj_vec_op3_t)41u)
#define ZJ_V3_UNPACK_HI_U8  ((zj_vec_op3_t)42u)
#define ZJ_V3_UNPACK_LO_U16 ((zj_vec_op3_t)43u)
#define ZJ_V3_UNPACK_HI_U16 ((zj_vec_op3_t)44u)
#define ZJ_V3_UNPACK_LO_U32 ((zj_vec_op3_t)45u)
#define ZJ_V3_UNPACK_HI_U32 ((zj_vec_op3_t)46u)
#define ZJ_V3_PERMUTE_U32   ((zj_vec_op3_t)47u)
#define ZJ_V3_ADD_U8        ((zj_vec_op3_t)48u)
#define ZJ_V3_SUB_U8        ((zj_vec_op3_t)49u)
#define ZJ_V3_ADDS_U8       ((zj_vec_op3_t)50u)
#define ZJ_V3_SUBS_U8       ((zj_vec_op3_t)51u)
#define ZJ_V3_MIN_U8        ((zj_vec_op3_t)52u)
#define ZJ_V3_MAX_U8        ((zj_vec_op3_t)53u)

/* Vector ternary ops (UniOpVVVV). */
typedef uint32_t zj_vec_op4_t;
#define ZJ_V4_MADD_F32  ((zj_vec_op4_t)0u)
#define ZJ_V4_MSUB_F32  ((zj_vec_op4_t)1u)
#define ZJ_V4_NMADD_F32 ((zj_vec_op4_t)2u)
#define ZJ_V4_NMSUB_F32 ((zj_vec_op4_t)3u)
#define ZJ_V4_BLENDV_U8 ((zj_vec_op4_t)4u)

/* Vector unary + immediate ops (UniOpVVI). */
typedef uint32_t zj_vec_op2i_t;
#define ZJ_V2I_SLL_U16       ((zj_vec_op2i_t)0u)
#define ZJ_V2I_SRL_U16       ((zj_vec_op2i_t)1u)
#define ZJ_V2I_SRA_I16       ((zj_vec_op2i_t)2u)
#define ZJ_V2I_SLL_U32       ((zj_vec_op2i_t)3u)
#define ZJ_V2I_SRL_U32       ((zj_vec_op2i_t)4u)
#define ZJ_V2I_SRA_I32       ((zj_vec_op2i_t)5u)
#define ZJ_V2I_SWIZZLE_U32X4 ((zj_vec_op2i_t)6u)

/* Vector memory load ops (UniOpVM). */
typedef uint32_t zj_vec_load_t;
#define ZJ_VLOAD_A_VEC         ((zj_vec_load_t)0u)
#define ZJ_VLOAD_U_VEC         ((zj_vec_load_t)1u)
#define ZJ_VLOAD_U_32          ((zj_vec_load_t)2u)
#define ZJ_VLOAD_U_64          ((zj_vec_load_t)3u)
#define ZJ_VLOAD_U_128         ((zj_vec_load_t)4u)
#define ZJ_VLOAD_U_256         ((zj_vec_load_t)5u)
#define ZJ_VLOAD_A_128         ((zj_vec_load_t)6u)
#define ZJ_VLOAD_CVT_U8_TO_U16 ((zj_vec_load_t)7u)
#define ZJ_VLOAD_CVT_U8_TO_U32 ((zj_vec_load_t)8u)
#define ZJ_VLOAD_CVT_U16_TO_U32 ((zj_vec_load_t)9u)
#define ZJ_VLOAD_BROADCAST_U32 ((zj_vec_load_t)10u)
#define ZJ_VLOAD_BROADCAST_U64 ((zj_vec_load_t)11u)

/* Vector memory store ops (UniOpMV). */
typedef uint32_t zj_vec_store_t;
#define ZJ_VSTORE_A_VEC ((zj_vec_store_t)0u)
#define ZJ_VSTORE_U_VEC ((zj_vec_store_t)1u)
#define ZJ_VSTORE_U_32  ((zj_vec_store_t)2u)
#define ZJ_VSTORE_U_64  ((zj_vec_store_t)3u)
#define ZJ_VSTORE_U_128 ((zj_vec_store_t)4u)
#define ZJ_VSTORE_U_256 ((zj_vec_store_t)5u)
#define ZJ_VSTORE_A_128 ((zj_vec_store_t)6u)

/* ABI version & Runtime / CPU detection. */
uint32_t zj_abi_version(void);
zj_runtime* zj_runtime_create(void);
void zj_runtime_destroy(zj_runtime* rt);
void zj_runtime_release(zj_runtime* rt, void* fn_ptr);
void zj_cpu_detect(zj_cpu_info_t* out, int32_t max_isa_override);

/* Compiler context & function lifecycle. */
zj_ctx* zj_ctx_create(zj_runtime* rt, const zj_cpu_info_t* cpu, uint8_t capture_disasm);
void zj_ctx_destroy(zj_ctx* ctx);
int32_t zj_ctx_begin_func(zj_ctx* ctx);
zj_reg_t zj_ctx_arg_gp(zj_ctx* ctx, uint32_t index);
void zj_ctx_ret_void(zj_ctx* ctx);
void zj_ctx_end_func(zj_ctx* ctx);
int32_t zj_ctx_finalize(zj_ctx* ctx, void** out_fn, size_t* out_size, uint8_t* out_execmem_denied);
void zj_ctx_get_disasm(const zj_ctx* ctx, const char** out_ptr, size_t* out_len);
uint32_t zj_ctx_last_error(const zj_ctx* ctx);

/* Virtual registers, labels & constants. */
zj_reg_t zj_new_gp64(zj_ctx* ctx);
zj_reg_t zj_new_gp32(zj_ctx* ctx);
zj_reg_t zj_new_vec(zj_ctx* ctx);
zj_reg_t zj_new_vec128(zj_ctx* ctx);
zj_label_t zj_new_label(zj_ctx* ctx);
void zj_bind(zj_ctx* ctx, zj_label_t label);
zj_reg_t zj_const_f32_splat(zj_ctx* ctx, float val);
zj_reg_t zj_const_i32_splat(zj_ctx* ctx, int32_t val);
zj_reg_t zj_const_bytes(zj_ctx* ctx, const uint8_t* data, uint32_t len);
zj_reg_t zj_vec_zero(zj_ctx* ctx);

/* GP ops & control flow. */
void zj_gp_mov(zj_ctx* ctx, zj_reg_t dst, zj_reg_t src);
void zj_gp_mov_imm(zj_ctx* ctx, zj_reg_t dst, int64_t imm);
void zj_gp_load(zj_ctx* ctx, zj_reg_t dst, zj_mem_t mem, uint32_t bytes, uint8_t sign_extend);
void zj_gp_store(zj_ctx* ctx, zj_mem_t mem, zj_reg_t src, uint32_t bytes);
void zj_gp_op2(zj_ctx* ctx, zj_gp_op2_t op, zj_reg_t dst, zj_reg_t src);
void zj_gp_op3(zj_ctx* ctx, zj_gp_op3_t op, zj_reg_t dst, zj_reg_t a, zj_reg_t b);
void zj_gp_op3_imm(zj_ctx* ctx, zj_gp_op3_t op, zj_reg_t dst, zj_reg_t a, int64_t imm);
void zj_gp_cmov(zj_ctx* ctx, zj_cond_t cond, zj_reg_t dst, zj_reg_t lhs, zj_reg_t rhs, zj_reg_t src);
void zj_gp_select(zj_ctx* ctx, zj_cond_t cond, zj_reg_t dst, zj_reg_t lhs, zj_reg_t rhs, zj_reg_t true_val, zj_reg_t false_val);
void zj_jmp(zj_ctx* ctx, zj_label_t target);
void zj_jcc(zj_ctx* ctx, zj_cond_t cond, zj_reg_t lhs, zj_reg_t rhs, zj_label_t target);
void zj_jcc_imm(zj_ctx* ctx, zj_cond_t cond, zj_reg_t lhs, int64_t rhs_imm, zj_label_t target);

/* Portable SIMD ops. */
void zj_vec_op2(zj_ctx* ctx, zj_vec_op2_t op, zj_reg_t dst, zj_reg_t src);
void zj_vec_op3(zj_ctx* ctx, zj_vec_op3_t op, zj_reg_t dst, zj_reg_t a, zj_reg_t b);
void zj_vec_op4(zj_ctx* ctx, zj_vec_op4_t op, zj_reg_t dst, zj_reg_t a, zj_reg_t b, zj_reg_t c);
void zj_vec_op2i(zj_ctx* ctx, zj_vec_op2i_t op, zj_reg_t dst, zj_reg_t src, uint32_t imm);
void zj_vec_load(zj_ctx* ctx, zj_vec_load_t op, zj_reg_t dst, zj_mem_t mem);
void zj_vec_store(zj_ctx* ctx, zj_vec_store_t op, zj_mem_t mem, zj_reg_t src);
void zj_vec_from_gp32(zj_ctx* ctx, zj_reg_t dst, zj_reg_t gp32);
void zj_vec_broadcast_gp32(zj_ctx* ctx, zj_reg_t dst, zj_reg_t gp32);
void zj_gp32_from_vec(zj_ctx* ctx, zj_reg_t gp32, zj_reg_t vec);

/* Minimal arch-specific escape hatches. */
void zj_vec_cvt_f16_to_f32(zj_ctx* ctx, zj_reg_t dst, zj_mem_t mem);
void zj_vec_cvt_f32_to_f16(zj_ctx* ctx, zj_mem_t mem, zj_reg_t src);
void zj_vec_store_u8_from_i32(zj_ctx* ctx, zj_mem_t mem, zj_reg_t src_i32);
void zj_vec_store_u8_from_i32x2(zj_ctx* ctx, zj_mem_t mem, zj_reg_t src0_i32, zj_reg_t src1_i32);
void zj_vec_store_u8_from_i16(zj_ctx* ctx, zj_mem_t mem, zj_reg_t src_i16);
void zj_vec_store_u8_from_i16x2(zj_ctx* ctx, zj_mem_t mem, zj_reg_t src0_i16, zj_reg_t src1_i16);
void zj_vec_store_u16_from_i32(zj_ctx* ctx, zj_mem_t mem, zj_reg_t src_i32);
void zj_vec_store_u16_from_i32x2(zj_ctx* ctx, zj_mem_t mem, zj_reg_t src0_i32, zj_reg_t src1_i32);
uint8_t zj_cpu_supports_fast_lut_u8(const zj_cpu_info_t* cpu);
void zj_vec_lut_u8_256(zj_ctx* ctx, zj_reg_t dst, zj_reg_t idx, const uint8_t* table_256);
void zj_vec_mul_c_f32(zj_ctx* ctx, zj_reg_t dst, zj_reg_t a, float b_const);
void zj_vec_add_c_f32(zj_ctx* ctx, zj_reg_t dst, zj_reg_t a, float b_const);
void zj_vec_sub_c_f32(zj_ctx* ctx, zj_reg_t dst, zj_reg_t a, float b_const);
void zj_vec_min_c_f32(zj_ctx* ctx, zj_reg_t dst, zj_reg_t a, float b_const);
void zj_vec_max_c_f32(zj_ctx* ctx, zj_reg_t dst, zj_reg_t a, float b_const);
void zj_vec_cmp_c_f32(zj_ctx* ctx, zj_vec_op3_t cmp_op, zj_reg_t dst, zj_reg_t a, float b_const);
void zj_vec_fma_ac_f32(zj_ctx* ctx, zj_reg_t dst, zj_reg_t a, float b_const, zj_reg_t c);
void zj_vec_blendv_f32(zj_ctx* ctx, zj_reg_t dst, zj_reg_t false_val, zj_reg_t true_val, zj_reg_t mask);
void zj_vec_and_f32(zj_ctx* ctx, zj_reg_t dst, zj_reg_t a, zj_reg_t b);
void zj_vec_andn_f32(zj_ctx* ctx, zj_reg_t dst, zj_reg_t mask_not, zj_reg_t b);
void zj_vec_fma_bc_f32(zj_ctx* ctx, zj_reg_t dst, zj_reg_t a, zj_reg_t b, float c_const);
void zj_vec_fma_cc_f32(zj_ctx* ctx, zj_reg_t dst, zj_reg_t a, float b_const, float c_const);
void zj_vec_fnma_bc_f32(zj_ctx* ctx, zj_reg_t dst, zj_reg_t a, float b_const, zj_reg_t c);
void zj_vec_permute_f32(zj_ctx* ctx, zj_reg_t dst, zj_reg_t src, const uint32_t* lane_indices, uint32_t lane_count);
void zj_vec_gather_f32(zj_ctx* ctx, zj_reg_t dst, zj_reg_t base, zj_reg_t byte_offsets_i32, uint8_t can_use_hw_gather);
void zj_vec_gather_f16(zj_ctx* ctx, zj_reg_t dst_f32, zj_reg_t base, zj_reg_t byte_offsets_i32);
void zj_vec_gather_u16(zj_ctx* ctx, zj_reg_t dst_i32, zj_reg_t base, zj_reg_t byte_offsets_i32);
void zj_vec_gather_u8(zj_ctx* ctx, zj_reg_t dst_i32, zj_reg_t base, zj_reg_t byte_offsets_i32);

#ifdef __cplusplus
}
#endif

#endif /* ZEXPR_ZASMJIT_H */
