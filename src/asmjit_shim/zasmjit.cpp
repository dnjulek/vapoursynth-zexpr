// Thin C++20 C-ABI shim wrapping AsmJit (ujit::UniCompiler + JitRuntime).
// Compiled with -fno-exceptions -fno-rtti -fvisibility=hidden (NO -fno-threadsafe-statics).

#include "zasmjit.h"

#include <asmjit/core.h>
#include <asmjit/ujit/ujit_base.h>
#include <asmjit/ujit/uni_compiler.h>
#include <asmjit/ujit/uni_condition.h>
#include <asmjit/ujit/uni_op.h>

#if defined(ASMJIT_UJIT_X86)
#include <asmjit/x86.h>
#elif defined(ASMJIT_UJIT_AARCH64)
#include <asmjit/a64.h>
#else
#error "zasmjit.cpp requires x86-64 or AArch64 host architecture"
#endif

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace aj = asmjit;
namespace ujit = asmjit::ujit;

namespace {

struct ShimErrorHandler final : public aj::ErrorHandler {
    aj::Error first_error = aj::Error::kOk;
    void handle_error(aj::Error err, const char* msg, aj::BaseEmitter* /*origin*/) override {
        if (first_error == aj::Error::kOk) {
            first_error = err;
            // Opt-in diagnostics: AsmJit's message is otherwise only reported as CompileFailed.
            if (std::getenv("ZEXPR_JIT_ERRORS") != nullptr) std::fprintf(stderr, "zexpr: AsmJit error %u: %s\n", unsigned(err), msg != nullptr ? msg : "(null)");
        }
    }
};

struct ConstEntry {
    uint8_t bytes[64];
    uint32_t len;
    uint32_t reg_id;
    uint32_t reg_sig;
};

static inline zj_reg_t pack_reg(const aj::Reg& r) noexcept {
    return zj_reg_t{r.id(), r.signature().bits()};
}

static inline ujit::Gp unpack_gp(zj_reg_t r) noexcept {
    return ujit::Gp(aj::OperandSignature{r.sig}, r.id);
}

static inline ujit::Vec unpack_vec(zj_reg_t r) noexcept {
    return ujit::Vec(aj::OperandSignature{r.sig}, r.id);
}

static inline aj::Label unpack_label(zj_label_t l) noexcept {
    return aj::Label(l.id);
}

static inline ujit::CondCode to_cond_code(zj_cond_t c) noexcept {
    switch (c) {
        case ZJ_COND_EQ:  return ujit::CondCode::kEqual;
        case ZJ_COND_SLT: return ujit::CondCode::kSignedLT;
        case ZJ_COND_SGE: return ujit::CondCode::kSignedGE;
        case ZJ_COND_ULT: return ujit::CondCode::kUnsignedLT;
        default:          return ujit::CondCode::kEqual;
    }
}

static aj::CpuFeatures build_cpu_features(const zj_cpu_info_t& cpu) noexcept {
    aj::CpuFeatures f;
#if defined(ASMJIT_UJIT_X86)
    const aj::CpuFeatures& host = aj::CpuInfo::host().features();
    f.add(aj::CpuFeatures::X86::kCMOV, aj::CpuFeatures::X86::kSSE, aj::CpuFeatures::X86::kSSE2);
    if (cpu.isa >= ZJ_ISA_V2) {
        if (host.x86().has_sse3())   f.add(aj::CpuFeatures::X86::kSSE3);
        if (host.x86().has_ssse3())  f.add(aj::CpuFeatures::X86::kSSSE3);
        if (host.x86().has_sse4_1()) f.add(aj::CpuFeatures::X86::kSSE4_1);
        if (host.x86().has_sse4_2()) f.add(aj::CpuFeatures::X86::kSSE4_2);
        if (host.x86().has_popcnt()) f.add(aj::CpuFeatures::X86::kPOPCNT);
    }
    if (cpu.isa >= ZJ_ISA_V3) {
        if (host.x86().has_avx())   f.add(aj::CpuFeatures::X86::kAVX);
        if (host.x86().has_avx2())  f.add(aj::CpuFeatures::X86::kAVX2);
        if (host.x86().has_fma())   f.add(aj::CpuFeatures::X86::kFMA);
        if (host.x86().has_f16c())  f.add(aj::CpuFeatures::X86::kF16C);
        if (host.x86().has_bmi())   f.add(aj::CpuFeatures::X86::kBMI);
        if (host.x86().has_bmi2())  f.add(aj::CpuFeatures::X86::kBMI2);
        if (host.x86().has_lzcnt()) f.add(aj::CpuFeatures::X86::kLZCNT);
        if (host.x86().has_movbe()) f.add(aj::CpuFeatures::X86::kMOVBE);
    }
    if (cpu.isa >= ZJ_ISA_V4) {
        if (host.x86().has_avx512_f())    f.add(aj::CpuFeatures::X86::kAVX512_F);
        if (host.x86().has_avx512_cd())   f.add(aj::CpuFeatures::X86::kAVX512_CD);
        if (host.x86().has_avx512_bw())   f.add(aj::CpuFeatures::X86::kAVX512_BW);
        if (host.x86().has_avx512_dq())   f.add(aj::CpuFeatures::X86::kAVX512_DQ);
        if (host.x86().has_avx512_vl())   f.add(aj::CpuFeatures::X86::kAVX512_VL);
        if (host.x86().has_avx512_vbmi()) f.add(aj::CpuFeatures::X86::kAVX512_VBMI);
    }
#elif defined(ASMJIT_UJIT_AARCH64)
    (void)cpu;
    const aj::CpuFeatures& host = aj::CpuInfo::host().features();
    f.add(aj::CpuFeatures::ARM::kARMv8a, aj::CpuFeatures::ARM::kFP, aj::CpuFeatures::ARM::kASIMD);
    if (host.arm().has_cssc()) f.add(aj::CpuFeatures::ARM::kCSSC);
#endif
    return f;
}

static aj::CpuHints build_cpu_hints(const zj_cpu_info_t& cpu) noexcept {
    aj::CpuHints h = aj::CpuInfo::host().hints();
#if defined(ASMJIT_UJIT_X86)
    if (cpu.isa < ZJ_ISA_V4) {
        const uint32_t mask =
            uint32_t(aj::CpuHints::kVecMaskedOps8) |
            uint32_t(aj::CpuHints::kVecMaskedOps16) |
            uint32_t(aj::CpuHints::kVecMaskedOps32) |
            uint32_t(aj::CpuHints::kVecMaskedOps64);
        h = static_cast<aj::CpuHints>(uint32_t(h) & ~mask);
    }
#else
    (void)cpu;
#endif
    return h;
}

} // namespace

struct zj_runtime {
    aj::JitRuntime rt;
};

struct zj_ctx {
    zj_runtime* rt;
    zj_cpu_info_t cpu;
    aj::CodeHolder code;
    aj::StringLogger logger;
    ShimErrorHandler err_handler;
    ujit::BackendCompiler cc;
    ujit::UniCompiler uc;
    aj::FuncNode* func_node = nullptr;
    std::vector<ConstEntry> const_cache;
    zj_reg_t zero_vec = {0u, 0u};
    bool has_zero_vec = false;
    bool capture_disasm = false;

    zj_ctx(zj_runtime* runtime, const zj_cpu_info_t& cpu_info, bool disasm) noexcept
        : rt(runtime),
          cpu(cpu_info),
          cc(&code),
          uc(&cc, build_cpu_features(cpu_info), build_cpu_hints(cpu_info)),
          capture_disasm(disasm) {
        const aj::CpuFeatures target_features = build_cpu_features(cpu_info);
        code.init(rt->rt.environment(), target_features);
        code.set_error_handler(&err_handler);
        if (capture_disasm) {
            logger.add_flags(aj::FormatFlags::kMachineCode | aj::FormatFlags::kRegCasts);
            code.set_logger(&logger);
        }
        code.attach(&cc);

        ujit::VecWidth vw = ujit::VecWidth::k128;
        if (cpu.vec_bytes == 64u) {
            vw = ujit::VecWidth::k512;
        } else if (cpu.vec_bytes == 32u) {
            vw = ujit::VecWidth::k256;
        }
        uc.init_vec_width(vw);
    }
};

namespace {

static inline bool ctx_ok(const zj_ctx* ctx) noexcept {
    return ctx != nullptr && ctx->err_handler.first_error == aj::Error::kOk;
}

static inline void set_shim_error(zj_ctx* ctx, aj::Error err, const char* msg) noexcept {
    if (ctx != nullptr) {
        ctx->err_handler.handle_error(err, msg, nullptr);
    }
}

#if defined(ASMJIT_UJIT_X86)
// Full-width view of a vector register for direct x86 emission: zmm, ymm or xmm by vec_bytes.
static inline aj::x86::Vec vw(const zj_ctx* ctx, const ujit::Vec& v) noexcept {
    return ctx->cpu.vec_bytes == 64u ? v.zmm() : (ctx->cpu.vec_bytes == 32u ? v.ymm() : v.xmm());
}
#define ZJ_X86_ID(name) aj::x86::Inst::kId##name
#else
#define ZJ_X86_ID(name) aj::InstId(0)
#endif

static inline zj_mem_t mem_add(zj_mem_t m, int32_t delta) noexcept {
    m.disp += delta;
    return m;
}

static inline ujit::Mem legalise_mem(zj_ctx* ctx, zj_mem_t m, uint32_t access_bytes) noexcept {
#if defined(ASMJIT_UJIT_X86)
    (void)ctx;
    const uint32_t sz = m.size != 0u ? uint32_t(m.size) : access_bytes;
    if (m.is_label) {
        return aj::x86::ptr(aj::Label(m.label_id), m.disp, sz);
    }
    const aj::x86::Gp base = unpack_gp(m.base);
    if (m.has_index) {
        const aj::x86::Gp index = unpack_gp(m.index);
        return aj::x86::ptr(base, index, m.shift, m.disp, sz);
    }
    return aj::x86::ptr(base, m.disp, sz);
#elif defined(ASMJIT_UJIT_AARCH64)
    if (m.is_label) {
        return aj::a64::ptr(aj::Label(m.label_id), m.disp);
    }
    const aj::a64::Gp base = unpack_gp(m.base);
    if (m.has_index) {
        const aj::a64::Gp index = unpack_gp(m.index);
        aj::a64::Gp addr = ctx->uc.new_gp64();
        if (m.shift == 0u) {
            ctx->cc.add(addr, base, index);
        } else {
            ctx->cc.add(addr, base, index, aj::a64::lsl(m.shift));
        }
        if (m.disp != 0) {
            ctx->uc.emit_3i(ujit::UniOpRRR::kAdd, addr, addr, aj::Imm(m.disp));
        }
        return aj::a64::ptr(addr);
    }
    if (m.disp == 0) {
        return aj::a64::ptr(base);
    }
    const uint32_t sz = m.size != 0u ? uint32_t(m.size) : access_bytes;
    const bool fits_unscaled = (m.disp >= -256 && m.disp <= 255);
    const bool fits_scaled =
        (sz > 0u) && (m.disp >= 0) && ((uint32_t(m.disp) % sz) == 0u) && ((uint32_t(m.disp) / sz) <= 4095u);
    if (fits_unscaled || fits_scaled) {
        return aj::a64::ptr(base, m.disp);
    }
    aj::a64::Gp addr = ctx->uc.new_gp64();
    ctx->uc.emit_3i(ujit::UniOpRRR::kAdd, addr, base, aj::Imm(m.disp));
    return aj::a64::ptr(addr);
#endif
}

// How a hoisted constant is loaded: the whole vector, or (x86 AVX2/AVX-512 splats) a broadcast of a
// 4-byte pool entry. Either way it is deduplicated by its vec_bytes-wide contents.
enum class ConstLoad { kVec, kBroadcastF32, kBroadcastI32 };

static zj_reg_t intern_const_vec(zj_ctx* ctx, const uint8_t* raw, uint32_t vec_bytes, ConstLoad load = ConstLoad::kVec) noexcept {
    if (!ctx_ok(ctx)) return zj_reg_t{0u, 0u};

    // Fast path for all-zero constant.
    bool all_zero = true;
    for (uint32_t i = 0; i < vec_bytes; ++i) {
        if (raw[i] != 0u) {
            all_zero = false;
            break;
        }
    }
    if (all_zero) {
        return zj_vec_zero(ctx);
    }

    for (const ConstEntry& e : ctx->const_cache) {
        if (e.len == vec_bytes && std::memcmp(e.bytes, raw, vec_bytes) == 0) {
            return zj_reg_t{e.reg_id, e.reg_sig};
        }
    }

    ujit::Vec v = ctx->uc.new_vec();
    ujit::Mem mem = ctx->cc.new_const(aj::ConstPoolScope::kLocal, raw, load == ConstLoad::kVec ? vec_bytes : 4u);
    {
        ujit::ScopedInjector inject(&ctx->cc, &ctx->uc._func_init_hook);
#if defined(ASMJIT_UJIT_X86)
        if (load == ConstLoad::kBroadcastF32) ctx->cc.vbroadcastss(vw(ctx, v), mem);
        else if (load == ConstLoad::kBroadcastI32) ctx->cc.vpbroadcastd(vw(ctx, v), mem);
        else
#endif
        ctx->uc.v_loadavec(v, mem);
    }

    ConstEntry entry{};
    std::memcpy(entry.bytes, raw, vec_bytes);
    entry.len = vec_bytes;
    entry.reg_id = v.id();
    entry.reg_sig = v.signature().bits();
    ctx->const_cache.push_back(entry);
    return pack_reg(v);
}

} // namespace

extern "C" {

uint32_t zj_abi_version(void) {
    return ZJ_ABI_VERSION;
}

zj_runtime* zj_runtime_create(void) {
    return new (std::nothrow) zj_runtime();
}

void zj_runtime_destroy(zj_runtime* rt) {
    delete rt;
}

void zj_runtime_release(zj_runtime* rt, void* fn_ptr) {
    if (rt != nullptr && fn_ptr != nullptr) {
        rt->rt.release(fn_ptr);
    }
}

static std::atomic<zj_teardown_fn> g_dso_teardown{nullptr};
static std::atomic<bool> g_emit_shutdown_marker{false};

void zj_set_dso_teardown(zj_teardown_fn fn) {
    if (fn != nullptr) {
        const char* env = std::getenv("ZEXPR_TEST_SHUTDOWN_MARKER");
        const bool enable = (env != nullptr && env[0] != '\0' && env[0] != '0');
        g_emit_shutdown_marker.store(enable, std::memory_order_release);
    }
    g_dso_teardown.store(fn, std::memory_order_release);
}

void zj_run_dso_teardown_now(void) {
    if (zj_teardown_fn fn = g_dso_teardown.exchange(nullptr, std::memory_order_acq_rel)) {
        fn();
    }
}

void zj_emit_shutdown_marker_if_enabled(void) {
    if (!g_emit_shutdown_marker.load(std::memory_order_acquire)) {
        return;
    }
    static constexpr char kMarker[] = "zexpr:shutdown:ok\n";
#if defined(_WIN32)
    HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
    if (h != nullptr && h != INVALID_HANDLE_VALUE) {
        DWORD written = 0;
        (void)WriteFile(h, kMarker, DWORD(sizeof(kMarker) - 1u), &written, nullptr);
    }
#else
    const ssize_t ignored = write(STDERR_FILENO, kMarker, sizeof(kMarker) - 1u);
    (void)ignored;
#endif
}

#if !defined(_WIN32)
// Linux ELF emits this into .fini_array; macOS Mach-O lowers it to
// __DATA,__mod_init_func calling ___cxa_atexit(..., &___dso_handle) for dyld4.
__attribute__((destructor)) static void zj_dso_destructor(void) {
    zj_run_dso_teardown_now();
}
#endif

void zj_cpu_detect(zj_cpu_info_t* out, int32_t max_isa_override) {
    if (out == nullptr) return;
    std::memset(out, 0, sizeof(*out));

    if (max_isa_override == 0) {
        out->isa = ZJ_ISA_SCALAR;
        return;
    }

    const aj::CpuInfo& host_cpu = aj::CpuInfo::host();
    const aj::CpuFeatures& f = host_cpu.features();

#if defined(ASMJIT_UJIT_X86)
    const bool has_v2 =
        f.x86().has_sse3() && f.x86().has_ssse3() && f.x86().has_sse4_1() &&
        f.x86().has_sse4_2() && f.x86().has_popcnt();
    const bool has_v3 =
        has_v2 && f.x86().has_avx() && f.x86().has_avx2() && f.x86().has_fma() &&
        f.x86().has_f16c() && f.x86().has_bmi() && f.x86().has_bmi2() &&
        f.x86().has_lzcnt() && f.x86().has_movbe();
    const bool has_v4 =
        has_v3 && f.x86().has_avx512_f() && f.x86().has_avx512_cd() &&
        f.x86().has_avx512_bw() && f.x86().has_avx512_dq() && f.x86().has_avx512_vl();

    // AVX-512 (v4) enabled by default on AVX-512 hosts after passing SDE -spr -mix/-ast & pytest gates.
    zj_isa_t target_isa = ZJ_ISA_SSE2;
    if (has_v4) {
        target_isa = ZJ_ISA_V4;
    } else if (has_v3) {
        target_isa = ZJ_ISA_V3;
    } else if (has_v2) {
        target_isa = ZJ_ISA_V2;
    }

    if (max_isa_override > 0) {
        const zj_isa_t req = static_cast<zj_isa_t>(max_isa_override);
        if (req >= ZJ_ISA_V4 && has_v4) {
            target_isa = ZJ_ISA_V4;
        } else if (req >= ZJ_ISA_V3 && has_v3) {
            target_isa = ZJ_ISA_V3;
        } else if (req >= ZJ_ISA_V2 && has_v2) {
            target_isa = ZJ_ISA_V2;
        } else {
            target_isa = ZJ_ISA_SSE2;
        }
    }

    out->isa = target_isa;
    if (target_isa == ZJ_ISA_V4) {
        out->vec_bytes = 64u;
        out->f32_lanes = 16u;
        out->has_ssse3 = 1u;
        out->has_sse4_1 = 1u;
        out->has_fma = 1u;
        out->has_f16c = 1u;
        out->has_avx2 = 1u;
        out->has_avx512 = 1u;
        // Real-silicon AVX-512 f32 gather split (overrideable via ZEXPR_HW_GATHER=0|1):
        // Intel (EMR/GNR) vpgatherdd zmm beats register gather by +63-65%, while AMD
        // (Zen 4/5) microcoded vpgatherdd zmm is 26-29% slower than 16-lane register gather.
        out->prefer_hw_gather_f32 = host_cpu.is_vendor("INTEL") ? 1u : 0u;
        // Split widening loads (see zj_vec_load): Granite Rapids silicon +3.3% on 8/16-bit
        // kernels, but Zen 4 (EPYC 9V74) -1% despite llvm-mca's -14%, so Intel only.
        out->prefer_alt_widen_load = host_cpu.is_vendor("INTEL") ? 1u : 0u;
    } else if (target_isa == ZJ_ISA_V3) {
        out->vec_bytes = 32u;
        out->f32_lanes = 8u;
        out->has_ssse3 = 1u;
        out->has_sse4_1 = 1u;
        out->has_fma = 1u;
        out->has_f16c = 1u;
        out->has_avx2 = 1u;
        out->has_avx512 = 0u;
        // Default false on AVX2: register gather beats Zen 3 vpgatherdd by ~35% and avoids
        // Intel GDS microcode penalties (overrideable via ZEXPR_HW_GATHER=1, P1).
        out->prefer_hw_gather_f32 = 0u;
    } else if (target_isa == ZJ_ISA_V2) {
        out->vec_bytes = 16u;
        out->f32_lanes = 4u;
        out->has_ssse3 = 1u;
        out->has_sse4_1 = 1u;
        out->has_fma = 0u;
        out->has_f16c = 0u;
        out->has_avx2 = 0u;
        out->has_avx512 = 0u;
    } else {
        out->vec_bytes = 16u;
        out->f32_lanes = 4u;
    }
#elif defined(ASMJIT_UJIT_AARCH64)
    (void)f;
    out->isa = ZJ_ISA_NEON;
    out->vec_bytes = 16u;
    out->f32_lanes = 4u;
    out->has_fma = 1u; // ARMv8.0 ASIMD has FMA (fmla/fmls) and fcvtl/fcvtn baseline
    // Real silicon: the zip + 2^23 bias u8/u16 -> f32 load is +6% geomean (up to +36%) on
    // Neoverse-N2, where 4S ucvtf is 2 uops on V0 only, but neutral on Apple M1, whose four FP
    // pipes all run ucvtf (and gamma/var_sq p16 lost 8-12% there). AsmJit names Apple "APPLE".
    out->prefer_alt_widen_load = host_cpu.is_vendor("APPLE") ? 0u : 1u;
#endif
}

zj_ctx* zj_ctx_create(zj_runtime* rt, const zj_cpu_info_t* cpu, uint8_t capture_disasm) {
    if (rt == nullptr || cpu == nullptr || cpu->isa == ZJ_ISA_SCALAR) {
        return nullptr;
    }
    return new (std::nothrow) zj_ctx(rt, *cpu, capture_disasm != 0u);
}

void zj_ctx_destroy(zj_ctx* ctx) {
    delete ctx;
}

static int32_t begin_with_sig(zj_ctx* ctx, const aj::FuncSignature& sig) noexcept {
    if (!ctx_ok(ctx)) return -1;
    ctx->func_node = ctx->cc.add_func(sig);
    if (ctx->func_node == nullptr) {
        return -1;
    }
    ctx->uc.hook_func();
    return 0;
}

int32_t zj_ctx_begin_func(zj_ctx* ctx) {
    // ZJ_SIG_EXPR_PLANE: (dst, dst_stride, srcs, src_strides, frame_n, props).
    return begin_with_sig(ctx, aj::FuncSignature::build<void, uint8_t*, intptr_t, const uint8_t* const*, const intptr_t*, intptr_t, const float*>());
}

int32_t zj_ctx_begin_lut_func(zj_ctx* ctx) {
    // ZJ_SIG_LUT_PLANE: (dst, dst_stride, srcs, src_strides, width, height).
    return begin_with_sig(ctx, aj::FuncSignature::build<void, uint8_t*, intptr_t, const uint8_t* const*, const intptr_t*, size_t, size_t>());
}

zj_reg_t zj_ctx_arg_gp(zj_ctx* ctx, uint32_t index) {
    if (!ctx_ok(ctx) || ctx->func_node == nullptr || index >= 6u) {
        return zj_reg_t{0u, 0u};
    }
    ujit::Gp reg = ctx->uc.new_gp64();
    ctx->func_node->set_arg(index, reg);
    return pack_reg(reg);
}

void zj_ctx_ret_void(zj_ctx* ctx) {
    if (!ctx_ok(ctx)) return;
    ctx->cc.ret();
}

void zj_ctx_end_func(zj_ctx* ctx) {
    if (!ctx_ok(ctx)) return;
    ctx->uc.unhook_func();
    ctx->cc.end_func();
}

// Adds finished code to the runtime and fills the outputs; flags denied executable memory.
static aj::Error add_to_runtime(zj_runtime* rt, aj::CodeHolder& code, void** out_fn, size_t* out_size, uint8_t* out_execmem_denied) noexcept {
    const size_t code_size = code.code_size();
    void* fn = nullptr;
    const aj::Error err = rt->rt.add(&fn, &code);
    if (err != aj::Error::kOk) {
        if (out_execmem_denied != nullptr &&
            (err == aj::Error::kInvalidState || err == aj::Error::kOutOfMemory)) {
            *out_execmem_denied = 1u;
        }
        return err;
    }
    if (out_fn != nullptr) *out_fn = fn;
    if (out_size != nullptr) *out_size = code_size;
    return aj::Error::kOk;
}

int32_t zj_ctx_finalize(zj_ctx* ctx, void** out_fn, size_t* out_size, uint8_t* out_execmem_denied) {
    if (out_fn != nullptr) *out_fn = nullptr;
    if (out_size != nullptr) *out_size = 0u;
    if (out_execmem_denied != nullptr) *out_execmem_denied = 0u;
    if (!ctx_ok(ctx)) return -1;

    aj::Error err = ctx->cc.finalize();
    if (err != aj::Error::kOk) {
        if (ctx->err_handler.first_error == aj::Error::kOk) {
            ctx->err_handler.first_error = err;
        }
        return -1;
    }

    err = add_to_runtime(ctx->rt, ctx->code, out_fn, out_size, out_execmem_denied);
    if (err != aj::Error::kOk) {
        if (ctx->err_handler.first_error == aj::Error::kOk) {
            ctx->err_handler.first_error = err;
        }
        return -1;
    }
    return 0;
}

void zj_ctx_get_disasm(const zj_ctx* ctx, const char** out_ptr, size_t* out_len) {
    if (out_ptr != nullptr) *out_ptr = "";
    if (out_len != nullptr) *out_len = 0u;
    if (ctx == nullptr || !ctx->capture_disasm) return;
    if (out_ptr != nullptr) *out_ptr = ctx->logger.data();
    if (out_len != nullptr) *out_len = ctx->logger.data_size();
}

zj_reg_t zj_new_gp64(zj_ctx* ctx) {
    if (!ctx_ok(ctx)) return zj_reg_t{0u, 0u};
    return pack_reg(ctx->uc.new_gp64());
}

zj_reg_t zj_new_vec(zj_ctx* ctx) {
    if (!ctx_ok(ctx)) return zj_reg_t{0u, 0u};
    return pack_reg(ctx->uc.new_vec());
}

zj_label_t zj_new_label(zj_ctx* ctx) {
    if (!ctx_ok(ctx)) return zj_label_t{0u};
    return zj_label_t{ctx->uc.new_label().id()};
}

void zj_bind(zj_ctx* ctx, zj_label_t label) {
    if (!ctx_ok(ctx)) return;
    ctx->uc.bind(unpack_label(label));
}

void zj_align_code(zj_ctx* ctx, uint32_t alignment) {
    if (!ctx_ok(ctx) || alignment <= 1u) return;
    ctx->cc.align(aj::AlignMode::kCode, alignment);
}

zj_reg_t zj_vec_zero(zj_ctx* ctx) {
    if (!ctx_ok(ctx)) return zj_reg_t{0u, 0u};
    if (ctx->has_zero_vec) return ctx->zero_vec;
    ujit::Vec v = ctx->uc.new_vec();
    {
        ujit::ScopedInjector inject(&ctx->cc, &ctx->uc._func_init_hook);
        ctx->uc.v_zero_i(v);
    }
    ctx->zero_vec = pack_reg(v);
    ctx->has_zero_vec = true;
    return ctx->zero_vec;
}

#if defined(ASMJIT_UJIT_X86)
static void emit_x86_ternlog_blend(zj_ctx* ctx, ujit::Vec d, ujit::Vec f, ujit::Vec t, ujit::Vec m) noexcept {
    // Bitwise lane blend `(m & t) | (~m & f)` in 1 EVEX vpternlogd (zero register copies when dst aliases m, f, or t).
    if (d.id() == m.id()) {
        ctx->cc.vpternlogd(d.zmm(), t.zmm(), f.zmm(), 0xCA);
    } else if (d.id() == f.id()) {
        ctx->cc.vpternlogd(d.zmm(), t.zmm(), m.zmm(), 0xD8);
    } else if (d.id() == t.id()) {
        ctx->cc.vpternlogd(d.zmm(), f.zmm(), m.zmm(), 0xE4);
    } else {
        ctx->cc.vmovdqa64(d.zmm(), m.zmm());
        ctx->cc.vpternlogd(d.zmm(), t.zmm(), f.zmm(), 0xCA);
    }
}
#endif

#if defined(ASMJIT_UJIT_AARCH64)
static void emit_a64_bsl_blend(zj_ctx* ctx, ujit::Vec d, ujit::Vec f, ujit::Vec t, ujit::Vec m) noexcept {
    if (d.id() == m.id()) {
        ctx->cc.bsl(d.b16(), t.b16(), f.b16());
    } else if (d.id() == f.id()) {
        ctx->cc.bit(d.b16(), t.b16(), m.b16());
    } else if (d.id() == t.id()) {
        ctx->cc.bif(d.b16(), f.b16(), m.b16());
    } else {
        ctx->cc.mov(d.b16(), m.b16());
        ctx->cc.bsl(d.b16(), t.b16(), f.b16());
    }
}
#endif

static zj_reg_t splat32(zj_ctx* ctx, const void* val4, bool is_f32) noexcept {
    if (!ctx_ok(ctx)) return zj_reg_t{0u, 0u};
    uint8_t buf[64]{};
    for (uint32_t i = 0; i < ctx->cpu.f32_lanes; ++i) {
        std::memcpy(buf + i * 4u, val4, 4u);
    }
    ConstLoad load = ConstLoad::kVec;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes >= 32u) load = is_f32 ? ConstLoad::kBroadcastF32 : ConstLoad::kBroadcastI32;
#else
    (void)is_f32;
#endif
    return intern_const_vec(ctx, buf, ctx->cpu.vec_bytes, load);
}

zj_reg_t zj_const_f32_splat(zj_ctx* ctx, float val) {
    return splat32(ctx, &val, true);
}

zj_reg_t zj_const_i32_splat(zj_ctx* ctx, int32_t val) {
    return splat32(ctx, &val, false);
}

zj_reg_t zj_const_bytes(zj_ctx* ctx, const uint8_t* data, uint32_t len) {
    if (!ctx_ok(ctx) || data == nullptr || len == 0u || len > 64u) return zj_reg_t{0u, 0u};
    uint8_t buf[64]{};
    const uint32_t vb = ctx->cpu.vec_bytes;
    if (len >= vb) {
        std::memcpy(buf, data, vb);
    } else {
        for (uint32_t off = 0; off < vb; off += len) {
            const uint32_t n = (vb - off) < len ? (vb - off) : len;
            std::memcpy(buf + off, data, n);
        }
    }
    return intern_const_vec(ctx, buf, vb);
}

void zj_gp_mov(zj_ctx* ctx, zj_reg_t dst, zj_reg_t src) {
    if (!ctx_ok(ctx)) return;
    ctx->uc.emit_mov(unpack_gp(dst), unpack_gp(src));
}

void zj_gp_mov_imm(zj_ctx* ctx, zj_reg_t dst, int64_t imm) {
    if (!ctx_ok(ctx)) return;
    ctx->uc.emit_mov(unpack_gp(dst), aj::Imm(imm));
}

void zj_gp_load(zj_ctx* ctx, zj_reg_t dst, zj_mem_t mem, uint32_t bytes, uint8_t sign_extend) {
    if (!ctx_ok(ctx)) return;
    ujit::UniOpRM op = ujit::UniOpRM::kLoadU64;
    switch (bytes) {
        case 1u: op = sign_extend ? ujit::UniOpRM::kLoadI8 : ujit::UniOpRM::kLoadU8; break;
        case 2u: op = sign_extend ? ujit::UniOpRM::kLoadI16 : ujit::UniOpRM::kLoadU16; break;
        case 4u: op = sign_extend ? ujit::UniOpRM::kLoadI32 : ujit::UniOpRM::kLoadU32; break;
        case 8u: op = ujit::UniOpRM::kLoadU64; break;
        default: set_shim_error(ctx, aj::Error::kInvalidArgument, "invalid gp_load width"); return;
    }
    ctx->uc.emit_rm(op, unpack_gp(dst), legalise_mem(ctx, mem, bytes));
}

static inline bool map_gp_op3(zj_gp_op3_t op, ujit::UniOpRRR* out) noexcept {
    switch (op) {
        case ZJ_GP_ADD:  *out = ujit::UniOpRRR::kAdd; return true;
        case ZJ_GP_SUB:  *out = ujit::UniOpRRR::kSub; return true;
        case ZJ_GP_MUL:  *out = ujit::UniOpRRR::kMul; return true;
        case ZJ_GP_SMIN: *out = ujit::UniOpRRR::kSMin; return true;
        case ZJ_GP_SMAX: *out = ujit::UniOpRRR::kSMax; return true;
        default: return false;
    }
}

void zj_gp_op3(zj_ctx* ctx, zj_gp_op3_t op, zj_reg_t dst, zj_reg_t a, zj_reg_t b) {
    if (!ctx_ok(ctx)) return;
    ujit::UniOpRRR uop;
    if (!map_gp_op3(op, &uop)) {
        set_shim_error(ctx, aj::Error::kInvalidArgument, "invalid gp_op3 opcode");
        return;
    }
    ctx->uc.emit_3i(uop, unpack_gp(dst), unpack_gp(a), unpack_gp(b));
}

void zj_gp_op3_imm(zj_ctx* ctx, zj_gp_op3_t op, zj_reg_t dst, zj_reg_t a, int64_t imm) {
    if (!ctx_ok(ctx)) return;
    ujit::UniOpRRR uop;
    if (!map_gp_op3(op, &uop)) {
        set_shim_error(ctx, aj::Error::kInvalidArgument, "invalid gp_op3_imm opcode");
        return;
    }
    ctx->uc.emit_3i(uop, unpack_gp(dst), unpack_gp(a), aj::Imm(imm));
}

void zj_gp_cmov(zj_ctx* ctx, zj_cond_t cond, zj_reg_t dst, zj_reg_t lhs, zj_reg_t rhs, zj_reg_t src) {
    if (!ctx_ok(ctx)) return;
    ujit::UniCondition c(ujit::UniOpCond::kCompare, to_cond_code(cond), unpack_gp(lhs), unpack_gp(rhs));
    ctx->uc.emit_cmov(unpack_gp(dst), unpack_gp(src), c);
}

void zj_jcc(zj_ctx* ctx, zj_cond_t cond, zj_reg_t lhs, zj_reg_t rhs, zj_label_t target) {
    if (!ctx_ok(ctx)) return;
    ujit::UniCondition c(ujit::UniOpCond::kCompare, to_cond_code(cond), unpack_gp(lhs), unpack_gp(rhs));
    ctx->uc.emit_j_if(unpack_label(target), c);
}

void zj_jcc_imm(zj_ctx* ctx, zj_cond_t cond, zj_reg_t lhs, int64_t rhs_imm, zj_label_t target) {
    if (!ctx_ok(ctx)) return;
    ujit::UniCondition c(ujit::UniOpCond::kCompare, to_cond_code(cond), unpack_gp(lhs), aj::Imm(rhs_imm));
    ctx->uc.emit_j_if(unpack_label(target), c);
}

void zj_vec_op2(zj_ctx* ctx, zj_vec_op2_t op, zj_reg_t dst, zj_reg_t src) {
    if (!ctx_ok(ctx)) return;
    ujit::UniOpVV uop;
    switch (op) {
        case ZJ_V2_MOV:                  uop = ujit::UniOpVV::kMov; break;
        case ZJ_V2_ABS_F32:              uop = ujit::UniOpVV::kAbsF32; break;
        case ZJ_V2_NEG_F32:              uop = ujit::UniOpVV::kNegF32; break;
        case ZJ_V2_SQRT_F32:             uop = ujit::UniOpVV::kSqrtF32; break;
        case ZJ_V2_TRUNC_F32:            uop = ujit::UniOpVV::kTruncF32; break;
        case ZJ_V2_FLOOR_F32:            uop = ujit::UniOpVV::kFloorF32; break;
        case ZJ_V2_CEIL_F32:             uop = ujit::UniOpVV::kCeilF32; break;
        case ZJ_V2_ROUND_EVEN_F32:       uop = ujit::UniOpVV::kRoundEvenF32; break;
        case ZJ_V2_CVT_I32_TO_F32:       uop = ujit::UniOpVV::kCvtI32ToF32; break;
        case ZJ_V2_CVT_ROUND_F32_TO_I32: uop = ujit::UniOpVV::kCvtRoundF32ToI32; break;
        case ZJ_V2_NOT_U32:              uop = ujit::UniOpVV::kNotU32; break;
        case ZJ_V2_ABS_I16:              uop = ujit::UniOpVV::kAbsI16; break;
        case ZJ_V2_ABS_I32:              uop = ujit::UniOpVV::kAbsI32; break;
        case ZJ_V2_NEG_I16: {
            ujit::Vec zero = unpack_vec(zj_vec_zero(ctx));
            ctx->uc.emit_3v(ujit::UniOpVVV::kSubU16, unpack_vec(dst), zero, unpack_vec(src));
            return;
        }
        case ZJ_V2_NEG_I32: {
            ujit::Vec zero = unpack_vec(zj_vec_zero(ctx));
            ctx->uc.emit_3v(ujit::UniOpVVV::kSubU32, unpack_vec(dst), zero, unpack_vec(src));
            return;
        }
        case ZJ_V2_CVT_U8_TO_U32:        uop = ujit::UniOpVV::kCvtU8ToU32; break;
        case ZJ_V2_CVT_I16_LO_TO_I32:    uop = ujit::UniOpVV::kCvtI16LoToI32; break;
        case ZJ_V2_CVT_I16_HI_TO_I32:    uop = ujit::UniOpVV::kCvtI16HiToI32; break;
        case ZJ_V2_BROADCAST_U32:        uop = ujit::UniOpVV::kBroadcastU32; break;
        case ZJ_V2_BROADCAST_U64:        uop = ujit::UniOpVV::kBroadcastU64; break;
        case ZJ_V2_RSQRT_F32: {
            // Refined hardware rsqrt(x) on clamped positive x (max_rel <= 2e-7 across all ISAs):
            // - x86 SSE2/v2: rsqrtps + 1 NR; AVX2: vrsqrtps + 1 NR; AVX-512: vrsqrt14ps + 1 NR.
            // - AArch64 NEON: frsqrte + 2x frsqrts (< 0.5 ULP).
            ujit::Vec s = unpack_vec(src);
            ujit::Vec d = unpack_vec(dst);
#if defined(ASMJIT_UJIT_X86)
            ujit::Vec y = ctx->uc.new_vec();
            if (ctx->cpu.vec_bytes == 64u) {
                ctx->cc.vrsqrt14ps(y.zmm(), s.zmm());
            } else if (ctx->cpu.vec_bytes == 32u) {
                ctx->cc.vrsqrtps(y.ymm(), s.ymm());
            } else {
                ctx->cc.rsqrtps(y.xmm(), s.xmm());
            }
            ujit::Vec xh = ctx->uc.new_vec();
            ujit::Vec yy = ctx->uc.new_vec();
            ujit::Vec c_half = unpack_vec(zj_const_f32_splat(ctx, 0.5f));
            ujit::Vec c_three_half = unpack_vec(zj_const_f32_splat(ctx, 1.5f));
            ctx->uc.emit_3v(ujit::UniOpVVV::kMulF32, xh, s, c_half);
            ctx->uc.emit_3v(ujit::UniOpVVV::kMulF32, yy, y, y);
            ctx->uc.emit_4v(ujit::UniOpVVVV::kNMAddF32, yy, xh, yy, c_three_half);
            ctx->uc.emit_3v(ujit::UniOpVVV::kMulF32, d, y, yy);
#elif defined(ASMJIT_UJIT_AARCH64)
            ujit::Vec y = ctx->uc.new_vec128();
            ujit::Vec t = ctx->uc.new_vec128();
            ctx->cc.frsqrte(y.s4(), s.s4());
            ctx->cc.fmul(t.s4(), y.s4(), y.s4());
            ctx->cc.frsqrts(t.s4(), s.s4(), t.s4());
            ctx->cc.fmul(y.s4(), y.s4(), t.s4());
            ctx->cc.fmul(t.s4(), y.s4(), y.s4());
            ctx->cc.frsqrts(t.s4(), s.s4(), t.s4());
            ctx->cc.fmul(d.s4(), y.s4(), t.s4());
#endif
            return;
        }
        default: set_shim_error(ctx, aj::Error::kInvalidArgument, "invalid vec_op2 opcode"); return;
    }
    ctx->uc.emit_2v(uop, unpack_vec(dst), unpack_vec(src));
}

void zj_vec_op3(zj_ctx* ctx, zj_vec_op3_t op, zj_reg_t dst, zj_reg_t a, zj_reg_t b) {
    if (!ctx_ok(ctx)) return;
    ujit::UniOpVVV uop;
    switch (op) {
        case ZJ_V3_ADD_F32:       uop = ujit::UniOpVVV::kAddF32; break;
        case ZJ_V3_SUB_F32:       uop = ujit::UniOpVVV::kSubF32; break;
        case ZJ_V3_MUL_F32:       uop = ujit::UniOpVVV::kMulF32; break;
        case ZJ_V3_DIV_F32:       uop = ujit::UniOpVVV::kDivF32; break;
        case ZJ_V3_MIN_F32:       uop = ujit::UniOpVVV::kMinF32; break;
        case ZJ_V3_MAX_F32:       uop = ujit::UniOpVVV::kMaxF32; break;
        case ZJ_V3_CMP_EQ_F32:    uop = ujit::UniOpVVV::kCmpEqF32; break;
        case ZJ_V3_CMP_LT_F32:    uop = ujit::UniOpVVV::kCmpLtF32; break;
        case ZJ_V3_CMP_LE_F32:    uop = ujit::UniOpVVV::kCmpLeF32; break;
        case ZJ_V3_CMP_GT_F32:    uop = ujit::UniOpVVV::kCmpGtF32; break;
        case ZJ_V3_CMP_GE_F32:    uop = ujit::UniOpVVV::kCmpGeF32; break;
        case ZJ_V3_AND_U32:       uop = ujit::UniOpVVV::kAndU32; break;
        case ZJ_V3_OR_U32:        uop = ujit::UniOpVVV::kOrU32; break;
        case ZJ_V3_XOR_U32:       uop = ujit::UniOpVVV::kXorU32; break;
        case ZJ_V3_ANDN_U32:      uop = ujit::UniOpVVV::kAndnU32; break;
        case ZJ_V3_ADD_I32:       uop = ujit::UniOpVVV::kAddU32; break;
        case ZJ_V3_SUB_I32:       uop = ujit::UniOpVVV::kSubU32; break;
        case ZJ_V3_MUL_I32:       uop = ujit::UniOpVVV::kMulU32; break;
        case ZJ_V3_MIN_I32:       uop = ujit::UniOpVVV::kMinI32; break;
        case ZJ_V3_MAX_I32:       uop = ujit::UniOpVVV::kMaxI32; break;
        case ZJ_V3_CMP_EQ_I32:    uop = ujit::UniOpVVV::kCmpEqU32; break;
        case ZJ_V3_CMP_GT_I32:    uop = ujit::UniOpVVV::kCmpGtI32; break;
        case ZJ_V3_ADD_I16:       uop = ujit::UniOpVVV::kAddU16; break;
        case ZJ_V3_SUB_I16:       uop = ujit::UniOpVVV::kSubU16; break;
        case ZJ_V3_ADDS_U16:      uop = ujit::UniOpVVV::kAddsU16; break;
        case ZJ_V3_SUBS_U16:      uop = ujit::UniOpVVV::kSubsU16; break;
        case ZJ_V3_MUL_I16:       uop = ujit::UniOpVVV::kMulU16; break;
        case ZJ_V3_MIN_I16:       uop = ujit::UniOpVVV::kMinI16; break;
        case ZJ_V3_MAX_I16:       uop = ujit::UniOpVVV::kMaxI16; break;
        case ZJ_V3_MIN_U16:       uop = ujit::UniOpVVV::kMinU16; break;
        case ZJ_V3_MAX_U16:       uop = ujit::UniOpVVV::kMaxU16; break;
        case ZJ_V3_CMP_EQ_I16:    uop = ujit::UniOpVVV::kCmpEqU16; break;
        case ZJ_V3_CMP_GT_I16:    uop = ujit::UniOpVVV::kCmpGtI16; break;
        case ZJ_V3_SUB_U8:        uop = ujit::UniOpVVV::kSubU8; break;
        case ZJ_V3_ADDS_U8:       uop = ujit::UniOpVVV::kAddsU8; break;
        case ZJ_V3_SUBS_U8:       uop = ujit::UniOpVVV::kSubsU8; break;
        case ZJ_V3_MIN_U8:        uop = ujit::UniOpVVV::kMinU8; break;
        case ZJ_V3_MAX_U8:        uop = ujit::UniOpVVV::kMaxU8; break;
        default: set_shim_error(ctx, aj::Error::kInvalidArgument, "invalid or unsupported vec_op3 opcode"); return;
    }
    ctx->uc.emit_3v(uop, unpack_vec(dst), unpack_vec(a), unpack_vec(b));
}

void zj_vec_op3_mem(zj_ctx* ctx, zj_vec_op3_t op, zj_reg_t dst, zj_reg_t a, zj_mem_t mem) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes == 32u || ctx->cpu.vec_bytes == 64u) {
        namespace Inst = aj::x86::Inst;
        const bool z = ctx->cpu.vec_bytes == 64u;
        aj::InstId id = Inst::kIdNone;
        switch (op) {
            case ZJ_V3_ADD_F32:  id = Inst::kIdVaddps; break;
            case ZJ_V3_SUB_F32:  id = Inst::kIdVsubps; break;
            case ZJ_V3_MUL_F32:  id = Inst::kIdVmulps; break;
            case ZJ_V3_MIN_F32:  id = Inst::kIdVminps; break;
            case ZJ_V3_MAX_F32:  id = Inst::kIdVmaxps; break;
            case ZJ_V3_MIN_U8:   id = Inst::kIdVpminub; break;
            case ZJ_V3_MAX_U8:   id = Inst::kIdVpmaxub; break;
            case ZJ_V3_ADDS_U8:  id = Inst::kIdVpaddusb; break;
            case ZJ_V3_SUBS_U8:  id = Inst::kIdVpsubusb; break;
            case ZJ_V3_MIN_U16:  id = Inst::kIdVpminuw; break;
            case ZJ_V3_MAX_U16:  id = Inst::kIdVpmaxuw; break;
            case ZJ_V3_ADDS_U16: id = Inst::kIdVpaddusw; break;
            case ZJ_V3_SUBS_U16: id = Inst::kIdVpsubusw; break;
            case ZJ_V3_AND_U32:  id = z ? Inst::kIdVpandd : Inst::kIdVpand; break;
            case ZJ_V3_OR_U32:   id = z ? Inst::kIdVpord : Inst::kIdVpor; break;
            case ZJ_V3_XOR_U32:  id = z ? Inst::kIdVpxord : Inst::kIdVpxor; break;
            default: break;
        }
        if (id != Inst::kIdNone) {
            ctx->cc.emit(id, vw(ctx, unpack_vec(dst)), vw(ctx, unpack_vec(a)), legalise_mem(ctx, mem, ctx->cpu.vec_bytes));
            return;
        }
    }
#endif
    ujit::Vec tmp = ctx->uc.new_vec();
    ctx->uc.v_loaduvec(tmp, legalise_mem(ctx, mem, ctx->cpu.vec_bytes));
    zj_vec_op3(ctx, op, dst, a, pack_reg(tmp));
}

void zj_vec_op4(zj_ctx* ctx, zj_vec_op4_t op, zj_reg_t dst, zj_reg_t a, zj_reg_t b, zj_reg_t c) {
    if (!ctx_ok(ctx)) return;
    // On non-FMA x86 (SSE2/v2), UniCompiler lowers kNMAddF32 as `(-a)*b + c` via sse_fsign_flip,
    // flipping the NaN sign bit of `a` (and adding an xorps). Emit `c - a * b` directly to match
    // interp.fnmaF32 bit-for-bit on NaN payloads/signs (review.11.md sin/cos/tan L1-L2 parity).
    if (op == ZJ_V4_NMADD_F32 && !ctx->cpu.has_fma) {
        ujit::Vec tmp = ctx->uc.new_vec();
        ctx->uc.emit_3v(ujit::UniOpVVV::kMulF32, tmp, unpack_vec(a), unpack_vec(b));
        ctx->uc.emit_3v(ujit::UniOpVVV::kSubF32, unpack_vec(dst), unpack_vec(c), tmp);
        return;
    }
#if defined(ASMJIT_UJIT_X86)
    if (op == ZJ_V4_BLENDV_U8 && ctx->cpu.vec_bytes == 64u) {
        emit_x86_ternlog_blend(ctx, unpack_vec(dst), unpack_vec(a), unpack_vec(b), unpack_vec(c));
        return;
    }
#elif defined(ASMJIT_UJIT_AARCH64)
    if (op == ZJ_V4_BLENDV_U8) {
        emit_a64_bsl_blend(ctx, unpack_vec(dst), unpack_vec(a), unpack_vec(b), unpack_vec(c));
        return;
    }
#endif
    ujit::UniOpVVVV uop;
    switch (op) {
        case ZJ_V4_MADD_F32:  uop = ujit::UniOpVVVV::kMAddF32; break;
        case ZJ_V4_MSUB_F32:  uop = ujit::UniOpVVVV::kMSubF32; break;
        case ZJ_V4_NMADD_F32: uop = ujit::UniOpVVVV::kNMAddF32; break;
        case ZJ_V4_NMSUB_F32: uop = ujit::UniOpVVVV::kNMSubF32; break;
        case ZJ_V4_BLENDV_U8: uop = ujit::UniOpVVVV::kBlendV_U8; break;
        default: set_shim_error(ctx, aj::Error::kInvalidArgument, "invalid vec_op4 opcode"); return;
    }
    ctx->uc.emit_4v(uop, unpack_vec(dst), unpack_vec(a), unpack_vec(b), unpack_vec(c));
}

void zj_vec_op2i(zj_ctx* ctx, zj_vec_op2i_t op, zj_reg_t dst, zj_reg_t src, uint32_t imm) {
    if (!ctx_ok(ctx)) return;
    ujit::UniOpVVI uop;
    switch (op) {
        case ZJ_V2I_SLL_U16:       uop = ujit::UniOpVVI::kSllU16; break;
        case ZJ_V2I_SRA_I16:       uop = ujit::UniOpVVI::kSraI16; break;
        case ZJ_V2I_SLL_U32:       uop = ujit::UniOpVVI::kSllU32; break;
        case ZJ_V2I_SRA_I32:       uop = ujit::UniOpVVI::kSraI32; break;
        default: set_shim_error(ctx, aj::Error::kInvalidArgument, "invalid vec_op2i opcode"); return;
    }
    ctx->uc.emit_2vi(uop, unpack_vec(dst), unpack_vec(src), imm);
}

void zj_vec_load(zj_ctx* ctx, zj_vec_load_t op, zj_reg_t dst, zj_mem_t mem) {
    if (!ctx_ok(ctx)) return;
    ujit::Vec d = unpack_vec(dst);
    switch (op) {
        case ZJ_VLOAD_U_VEC:
            ctx->uc.v_loaduvec(d, legalise_mem(ctx, mem, ctx->cpu.vec_bytes));
            return;
        case ZJ_VLOAD_U_128:
            ctx->uc.emit_vm(ujit::UniOpVM::kLoad128_U32, d, legalise_mem(ctx, mem, 16u), ujit::Alignment(1));
            return;
        case ZJ_VLOAD_U_256:
            ctx->uc.emit_vm(ujit::UniOpVM::kLoad256_U32, d, legalise_mem(ctx, mem, 32u), ujit::Alignment(1));
            return;
        case ZJ_VLOAD_CVT_U8_TO_U16:
#if defined(ASMJIT_UJIT_X86)
            if (ctx->cpu.vec_bytes == 64u && ctx->cpu.prefer_alt_widen_load != 0u) {
                ctx->cc.vmovdqu32(d.ymm(), legalise_mem(ctx, mem, 32u));
                ctx->cc.vpmovzxbw(d.zmm(), d.ymm());
                return;
            }
#endif
            ctx->uc.emit_vm(ujit::UniOpVM::kLoadCvtN_U8ToU16, d, legalise_mem(ctx, mem, ctx->cpu.vec_bytes / 2u), ujit::Alignment(1));
            return;
        case ZJ_VLOAD_CVT_U8_TO_U32:
#if defined(ASMJIT_UJIT_X86)
            // AVX-512 split widening loads (prefer_alt_widen_load, Intel only): load into the low part
            // of dst, then widen register to register. Granite Rapids silicon +3.3% on 8/16-bit kernels;
            // on Zen 4 silicon it lost ~1% although llvm-mca models the memory form as 1.5 vs 1.0
            // FP-pipe cycles there.
            if (ctx->cpu.vec_bytes == 64u && ctx->cpu.prefer_alt_widen_load != 0u) {
                ctx->cc.vmovdqu32(d.xmm(), legalise_mem(ctx, mem, 16u));
                ctx->cc.vpmovzxbd(d.zmm(), d.xmm());
                return;
            }
            if (!ctx->cpu.has_sse4_1) {
                // Work around upstream AsmJit 1.23.0 bug at uni_compiler_x86.cpp:5402
                // where kLoadCvtN_U8ToU32 emits VEX `vmovd` on pre-SSE4.1 CPUs.
                ujit::Mem m4 = legalise_mem(ctx, mem, 4u);
                m4.set_size(4);
                ctx->cc.movd(d.xmm(), m4);
                ctx->uc.emit_2v(ujit::UniOpVV::kCvtU8LoToU16, d, d);
                ctx->uc.emit_2v(ujit::UniOpVV::kCvtU16LoToU32, d, d);
                return;
            }
#elif defined(ASMJIT_UJIT_AARCH64)
            // V2: UniCompiler a64 kLoadCvtN_U8ToU32 loads dst.size()/2 = 8 bytes (OOB read and
            // unaligned scaled offset fault); load exactly 4 bytes and zero-extend to u32.
            ctx->uc.emit_vm(ujit::UniOpVM::kLoad32_U32, d, legalise_mem(ctx, mem, 4u), ujit::Alignment(1));
            ctx->uc.emit_2v(ujit::UniOpVV::kCvtU8ToU32, d, d);
            return;
#endif
            ctx->uc.emit_vm(ujit::UniOpVM::kLoadCvtN_U8ToU32, d, legalise_mem(ctx, mem, ctx->cpu.vec_bytes / 4u), ujit::Alignment(1));
            return;
        case ZJ_VLOAD_CVT_U16_TO_U32:
#if defined(ASMJIT_UJIT_X86)
            if (ctx->cpu.vec_bytes == 64u && ctx->cpu.prefer_alt_widen_load != 0u) {
                ctx->cc.vmovdqu32(d.ymm(), legalise_mem(ctx, mem, 32u));
                ctx->cc.vpmovzxwd(d.zmm(), d.ymm());
                return;
            }
#endif
            ctx->uc.emit_vm(ujit::UniOpVM::kLoadCvtN_U16ToU32, d, legalise_mem(ctx, mem, ctx->cpu.vec_bytes / 2u), ujit::Alignment(1));
            return;
        case ZJ_VLOAD_BROADCAST_U32:
            ctx->uc.v_broadcast_u32(d, legalise_mem(ctx, mem, 4u));
            return;
        case ZJ_VLOAD_BROADCAST_U64:
            ctx->uc.v_broadcast_u64(d, legalise_mem(ctx, mem, 8u));
            return;
        default:
            set_shim_error(ctx, aj::Error::kInvalidArgument, "invalid vec_load opcode");
            return;
    }
}

void zj_vec_store(zj_ctx* ctx, zj_vec_store_t op, zj_mem_t mem, zj_reg_t src) {
    if (!ctx_ok(ctx)) return;
    ujit::Vec s = unpack_vec(src);
    switch (op) {
        case ZJ_VSTORE_U_VEC:
            ctx->uc.v_storeuvec(legalise_mem(ctx, mem, ctx->cpu.vec_bytes), s);
            return;
        case ZJ_VSTORE_U_128:
            ctx->uc.emit_mv(ujit::UniOpMV::kStore128_U32, legalise_mem(ctx, mem, 16u), s, ujit::Alignment(1));
            return;
        case ZJ_VSTORE_U_256:
            ctx->uc.emit_mv(ujit::UniOpMV::kStore256_U32, legalise_mem(ctx, mem, 32u), s, ujit::Alignment(1));
            return;
        default:
            set_shim_error(ctx, aj::Error::kInvalidArgument, "invalid vec_store opcode");
            return;
    }
}

void zj_vec_broadcast_gp32(zj_ctx* ctx, zj_reg_t dst, zj_reg_t gp32) {
    if (!ctx_ok(ctx)) return;
    ujit::Vec v = unpack_vec(dst);
    ctx->uc.s_mov_u32(v, unpack_gp(gp32).r32());
    ctx->uc.emit_2v(ujit::UniOpVV::kBroadcastU32, v, v);
}

// ============================================================================
// Arch-specific escape hatches
// ============================================================================

void zj_vec_cvt_f16_to_f32(zj_ctx* ctx, zj_reg_t dst, zj_mem_t mem) {
    if (!ctx_ok(ctx)) return;
    ujit::Vec d = unpack_vec(dst);
    const uint32_t load_bytes = ctx->cpu.f32_lanes * 2u;
    ujit::Mem m = legalise_mem(ctx, mem, load_bytes);
#if defined(ASMJIT_UJIT_X86)
    if (!ctx->cpu.has_f16c) {
        set_shim_error(ctx, aj::Error::kFeatureNotEnabled, "f16 load requires F16C");
        return;
    }
    ctx->cc.vcvtph2ps(vw(ctx, d), m);
#elif defined(ASMJIT_UJIT_AARCH64)
    ujit::Vec tmp = ctx->uc.new_vec128();
    ctx->cc.ldr(tmp.d(), m);
    ctx->cc.fcvtl(d.s4(), tmp.h4());
#endif
}

void zj_vec_cvt_f16_to_f32x2(zj_ctx* ctx, zj_reg_t dst0, zj_reg_t dst1, zj_mem_t mem) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_AARCH64)
    ujit::Vec d0 = unpack_vec(dst0);
    ujit::Vec d1 = unpack_vec(dst1);
    ujit::Vec tmp = ctx->uc.new_vec128();
    ctx->cc.ldr(tmp.q(), legalise_mem(ctx, mem, 16u));
    ctx->cc.fcvtl(d0.s4(), tmp.h4());
    ctx->cc.fcvtl2(d1.s4(), tmp.h8());
#else
    zj_vec_cvt_f16_to_f32(ctx, dst0, mem);
    zj_vec_cvt_f16_to_f32(ctx, dst1, mem_add(mem, int32_t(ctx->cpu.f32_lanes * 2u)));
#endif
}

void zj_vec_cvt_f32_to_f16(zj_ctx* ctx, zj_mem_t mem, zj_reg_t src) {
    if (!ctx_ok(ctx)) return;
    ujit::Vec s = unpack_vec(src);
    const uint32_t store_bytes = ctx->cpu.f32_lanes * 2u;
    ujit::Mem m = legalise_mem(ctx, mem, store_bytes);
#if defined(ASMJIT_UJIT_X86)
    if (!ctx->cpu.has_f16c) {
        set_shim_error(ctx, aj::Error::kFeatureNotEnabled, "f16 store requires F16C");
        return;
    }
    // Rounding mode 4 = MXCSR (round to nearest even).
    ctx->cc.vcvtps2ph(m, vw(ctx, s), 4);
#elif defined(ASMJIT_UJIT_AARCH64)
    ujit::Vec tmp = ctx->uc.new_vec128();
    ctx->cc.fcvtn(tmp.h4(), s.s4());
    ctx->cc.str(tmp.d(), m);
#endif
}

void zj_vec_cvt_f32_to_f16x2(zj_ctx* ctx, zj_mem_t mem, zj_reg_t src0, zj_reg_t src1) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_AARCH64)
    ujit::Vec s0 = unpack_vec(src0);
    ujit::Vec s1 = unpack_vec(src1);
    ujit::Vec tmp = ctx->uc.new_vec128();
    ctx->cc.fcvtn(tmp.h4(), s0.s4());
    ctx->cc.fcvtn2(tmp.h8(), s1.s4());
    ctx->cc.str(tmp.q(), legalise_mem(ctx, mem, 16u));
#else
    zj_vec_cvt_f32_to_f16(ctx, mem, src0);
    zj_vec_cvt_f32_to_f16(ctx, mem_add(mem, int32_t(ctx->cpu.f32_lanes * 2u)), src1);
#endif
}

void zj_vec_load_u_vecx2(zj_ctx* ctx, zj_reg_t dst0, zj_reg_t dst1, zj_mem_t mem) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_AARCH64)
    if (!mem.is_label && !mem.has_index && (mem.disp % 16) == 0 && mem.disp >= -1024 && mem.disp <= 1008) {
        ctx->cc.ldp(unpack_vec(dst0).q(), unpack_vec(dst1).q(), aj::a64::ptr(unpack_gp(mem.base), mem.disp));
        return;
    }
#endif
    ctx->uc.v_loaduvec(unpack_vec(dst0), legalise_mem(ctx, mem, ctx->cpu.vec_bytes));
    ctx->uc.v_loaduvec(unpack_vec(dst1), legalise_mem(ctx, mem_add(mem, int32_t(ctx->cpu.vec_bytes)), ctx->cpu.vec_bytes));
}

void zj_vec_load_u8_to_u32x2(zj_ctx* ctx, zj_reg_t dst0, zj_reg_t dst1, zj_mem_t mem) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_AARCH64)
    ujit::Vec d0 = unpack_vec(dst0);
    ujit::Vec d1 = unpack_vec(dst1);
    ctx->cc.ldr(d0.d(), legalise_mem(ctx, mem, 8u));
    ctx->cc.ushll(d0.h8(), d0.b8(), 0);
    ctx->cc.ushll2(d1.s4(), d0.h8(), 0);
    ctx->cc.ushll(d0.s4(), d0.h4(), 0);
    return;
#endif
    zj_vec_load(ctx, ZJ_VLOAD_CVT_U8_TO_U32, dst0, mem);
    zj_vec_load(ctx, ZJ_VLOAD_CVT_U8_TO_U32, dst1, mem_add(mem, int32_t(ctx->cpu.f32_lanes)));
}

void zj_vec_load_u16_to_u32x2(zj_ctx* ctx, zj_reg_t dst0, zj_reg_t dst1, zj_mem_t mem) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_AARCH64)
    ujit::Vec d0 = unpack_vec(dst0);
    ujit::Vec d1 = unpack_vec(dst1);
    ctx->cc.ldr(d0.q(), legalise_mem(ctx, mem, 16u));
    ctx->cc.ushll2(d1.s4(), d0.h8(), 0);
    ctx->cc.ushll(d0.s4(), d0.h4(), 0);
    return;
#endif
    zj_vec_load(ctx, ZJ_VLOAD_CVT_U16_TO_U32, dst0, mem);
    zj_vec_load(ctx, ZJ_VLOAD_CVT_U16_TO_U32, dst1, mem_add(mem, int32_t(ctx->cpu.f32_lanes * 2u)));
}

void zj_vec_load_u8_to_u16x2(zj_ctx* ctx, zj_reg_t dst0, zj_reg_t dst1, zj_mem_t mem) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_AARCH64)
    ujit::Vec d0 = unpack_vec(dst0);
    ujit::Vec d1 = unpack_vec(dst1);
    ctx->cc.ldr(d0.q(), legalise_mem(ctx, mem, 16u));
    ctx->cc.ushll2(d1.h8(), d0.b16(), 0);
    ctx->cc.ushll(d0.h8(), d0.b8(), 0);
    return;
#endif
    zj_vec_load(ctx, ZJ_VLOAD_CVT_U8_TO_U16, dst0, mem);
    zj_vec_load(ctx, ZJ_VLOAD_CVT_U8_TO_U16, dst1, mem_add(mem, int32_t(ctx->cpu.vec_bytes / 2u)));
}

#if defined(ASMJIT_UJIT_AARCH64)
// Exact u8/u16 -> f32 without `ucvtf`: `zip` each u16 lane with a 0x4B00 high half, giving the
// f32 bit pattern of 2^23 + u (exact for u < 2^23), then subtract 2^23 (exact; u = 0 gives +0.0
// like `ucvtf`). Neoverse-N2 runs 4S `ucvtf` as 2 uops on V0 only and `ushll` on V1 only, while
// `zip1`/`zip2`/`fsub` issue on both V pipes; Apple M1 executes the same op count.
static ujit::Vec neon_f32_magic_hi(zj_ctx* ctx) noexcept {
    uint8_t raw[16];
    for (uint32_t i = 0; i < 16u; i += 2u) {
        raw[i] = 0x00u;
        raw[i + 1u] = 0x4Bu;
    }
    return unpack_vec(intern_const_vec(ctx, raw, 16u));
}

static ujit::Vec neon_f32_magic_bias(zj_ctx* ctx) noexcept {
    return unpack_vec(zj_const_f32_splat(ctx, 8388608.0f));
}
#endif

void zj_vec_load_uint_to_f32(zj_ctx* ctx, zj_reg_t dst, zj_mem_t mem, uint32_t bytes_per_sample) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_AARCH64)
    ujit::Vec d = unpack_vec(dst);
    ujit::Vec hi = neon_f32_magic_hi(ctx);
    if (bytes_per_sample == 1u) {
        ctx->cc.ldr(d.s(), legalise_mem(ctx, mem, 4u));
        ujit::Vec zero = unpack_vec(zj_vec_zero(ctx));
        ctx->cc.zip1(d.b16(), d.b16(), zero.b16());
    } else {
        ctx->cc.ldr(d.d(), legalise_mem(ctx, mem, 8u));
    }
    ctx->cc.zip1(d.h8(), d.h8(), hi.h8());
    ctx->cc.fsub(d.s4(), d.s4(), neon_f32_magic_bias(ctx).s4());
#else
    (void)dst; (void)mem; (void)bytes_per_sample;
    set_shim_error(ctx, aj::Error::kFeatureNotEnabled, "bias u8/u16 -> f32 load is NEON-only");
#endif
}

void zj_vec_load_uint_to_f32x2(zj_ctx* ctx, zj_reg_t dst0, zj_reg_t dst1, zj_mem_t mem, uint32_t bytes_per_sample) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_AARCH64)
    ujit::Vec d0 = unpack_vec(dst0);
    ujit::Vec d1 = unpack_vec(dst1);
    ujit::Vec hi = neon_f32_magic_hi(ctx);
    if (bytes_per_sample == 1u) {
        ctx->cc.ldr(d0.d(), legalise_mem(ctx, mem, 8u));
        ujit::Vec zero = unpack_vec(zj_vec_zero(ctx));
        ctx->cc.zip1(d0.b16(), d0.b16(), zero.b16());
    } else {
        ctx->cc.ldr(d0.q(), legalise_mem(ctx, mem, 16u));
    }
    ctx->cc.zip2(d1.h8(), d0.h8(), hi.h8());
    ctx->cc.zip1(d0.h8(), d0.h8(), hi.h8());
    ujit::Vec bias = neon_f32_magic_bias(ctx);
    ctx->cc.fsub(d0.s4(), d0.s4(), bias.s4());
    ctx->cc.fsub(d1.s4(), d1.s4(), bias.s4());
#else
    (void)dst0; (void)dst1; (void)mem; (void)bytes_per_sample;
    set_shim_error(ctx, aj::Error::kFeatureNotEnabled, "bias u8/u16 -> f32 load is NEON-only");
#endif
}

void zj_vec_widen_x2(zj_ctx* ctx, zj_widen_op_t op, zj_reg_t dst0, zj_reg_t dst1, zj_reg_t a0, zj_reg_t a1, zj_reg_t b) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_AARCH64)
    namespace Inst = aj::a64::Inst;
    const bool sub = (op & ZJ_WIDEN_SUB) != 0u;
    const bool wide = (op & ZJ_WIDEN_W) != 0u;
    ujit::Vec d0 = unpack_vec(dst0);
    ujit::Vec d1 = unpack_vec(dst1);
    ujit::Vec x0 = unpack_vec(a0);
    ujit::Vec x1 = wide ? unpack_vec(a1) : x0;
    ujit::Vec y = unpack_vec(b);
    const aj::InstId lo_id = wide ? (sub ? Inst::kIdUsubw_v : Inst::kIdUaddw_v) : (sub ? Inst::kIdUsubl_v : Inst::kIdUaddl_v);
    const aj::InstId hi_id = wide ? (sub ? Inst::kIdUsubw2_v : Inst::kIdUaddw2_v) : (sub ? Inst::kIdUsubl2_v : Inst::kIdUaddl2_v);
    if ((op & ZJ_WIDEN_U16) != 0u) {
        ctx->cc.emit(lo_id, d0.s4(), wide ? x0.s4() : x0.h4(), y.h4());
        ctx->cc.emit(hi_id, d1.s4(), wide ? x1.s4() : x1.h8(), y.h8());
    } else {
        ctx->cc.emit(lo_id, d0.h8(), wide ? x0.h8() : x0.b8(), y.b8());
        ctx->cc.emit(hi_id, d1.h8(), wide ? x1.h8() : x1.b16(), y.b16());
    }
#else
    (void)op; (void)dst0; (void)dst1; (void)a0; (void)a1; (void)b;
    set_shim_error(ctx, aj::Error::kFeatureNotEnabled, "NEON widening op on non-AArch64");
#endif
}

void zj_vec_store_u_vecx2(zj_ctx* ctx, zj_mem_t mem, zj_reg_t src0, zj_reg_t src1) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_AARCH64)
    if (!mem.is_label && !mem.has_index && (mem.disp % 16) == 0 && mem.disp >= -1024 && mem.disp <= 1008) {
        ctx->cc.stp(unpack_vec(src0).q(), unpack_vec(src1).q(), aj::a64::ptr(unpack_gp(mem.base), mem.disp));
        return;
    }
#endif
    ctx->uc.v_storeuvec(legalise_mem(ctx, mem, ctx->cpu.vec_bytes), unpack_vec(src0));
    ctx->uc.v_storeuvec(legalise_mem(ctx, mem_add(mem, int32_t(ctx->cpu.vec_bytes)), ctx->cpu.vec_bytes), unpack_vec(src1));
}

#if defined(ASMJIT_UJIT_X86)
static void emit_x86_zmm_pack_permq_store(zj_ctx* ctx, ujit::Mem m, ujit::Vec s0, ujit::Vec s1, bool i16_to_u8) noexcept {
    // vpackuswb/vpackusdw pack each 128-bit lane as [s0_lane, s1_lane] -> qwords [s0_0, s1_0, s0_1, s1_1, ...].
    // Permute qwords [0,2,4,6,1,3,5,7] and store 64B in one vmovdqu64 (3x faster than 2x vpmovus* [mem] on Zen 4/5).
    static constexpr uint64_t kPermIdx[8] = { 0u, 2u, 4u, 6u, 1u, 3u, 5u, 7u };
    // vpermt2q, not vpermq: AsmJit's operand info for vpermq merges its imm8 form (where operand 1
    // is r/m), so when RA spilled the hoisted index under pressure it folded the stack slot into the
    // register-only index slot (`vpermq zmm, [rsp+N], zmm` -> InvalidInstruction, e.g.
    // `x exp 0.5 * x log 0.25 * +` to u16). vpermt2q has a single form, so RA reloads a spilled
    // index instead. Indices < 8 select from the first table (t) only.
    ujit::Vec idx = unpack_vec(intern_const_vec(ctx, reinterpret_cast<const uint8_t*>(kPermIdx), 64u));
    ujit::Vec t = ctx->uc.new_vec();
    if (i16_to_u8) {
        ctx->cc.vpackuswb(t.zmm(), s0.zmm(), s1.zmm());
    } else {
        ctx->cc.vpackusdw(t.zmm(), s0.zmm(), s1.zmm());
    }
    ctx->cc.vpermt2q(t.zmm(), idx.zmm(), t.zmm());
    ctx->cc.vmovdqu64(m, t.zmm());
}
#endif

void zj_vec_store_u8_from_i32(zj_ctx* ctx, zj_mem_t mem, zj_reg_t src_i32) {
    if (!ctx_ok(ctx)) return;
    ujit::Vec s = unpack_vec(src_i32);
    const uint32_t store_bytes = ctx->cpu.f32_lanes;
    ujit::Mem m = legalise_mem(ctx, mem, store_bytes);
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes == 64u) {
        // Caller clamped src_i32 to [0, peak] before converting to i32.
        // Reg-dest vpmovusdb + EVEX vmovdqu32 avoids slow Zen 4/5 memory-dest narrowing store (-33% rtp).
        ujit::Vec t = ctx->uc.new_vec();
        ctx->cc.vpmovusdb(t.xmm(), s.zmm());
        ctx->cc.vmovdqu32(m, t.xmm());
    } else if (ctx->cpu.vec_bytes == 32u) {
        ujit::Vec t = ctx->uc.new_vec();
        ctx->cc.vpackssdw(t.ymm(), s.ymm(), s.ymm());
        ctx->cc.vpermq(t.ymm(), t.ymm(), 0xD8);
        ctx->cc.vpackuswb(t.ymm(), t.ymm(), t.ymm());
        ctx->cc.vmovq(m, t.xmm());
    } else {
        ujit::Vec t = ctx->uc.new_vec128();
        ctx->uc.emit_3v(ujit::UniOpVVV::kPacksI32_I16, t, s, s);
        ctx->uc.emit_3v(ujit::UniOpVVV::kPacksI16_U8, t, t, t);
        ctx->uc.emit_mv(ujit::UniOpMV::kStore32_U32, m, t, ujit::Alignment(1));
    }
#elif defined(ASMJIT_UJIT_AARCH64)
    // L3: narrow signed i32 -> signed i16 -> unsigned u8 directly (no dead mov d[1], d[0]).
    ujit::Vec t = ctx->uc.new_vec128();
    ctx->cc.sqxtn(t.h4(), s.s4());
    ctx->cc.sqxtun(t.b8(), t.h8());
    ctx->cc.str(t.s(), m);
#endif
}

void zj_vec_store_u8_from_i32x2(zj_ctx* ctx, zj_mem_t mem, zj_reg_t src0_i32, zj_reg_t src1_i32) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes == 64u) {
        zj_vec_store_u8_from_i32(ctx, mem, src0_i32);
        zj_vec_store_u8_from_i32(ctx, mem_add(mem, 16), src1_i32);
        return;
    }
#endif
    ujit::Vec s0 = unpack_vec(src0_i32);
    ujit::Vec s1 = unpack_vec(src1_i32);
    const uint32_t store_bytes = ctx->cpu.f32_lanes * 2u;
    ujit::Mem m = legalise_mem(ctx, mem, store_bytes);
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes == 32u) {
        ujit::Vec t = ctx->uc.new_vec();
        ctx->cc.vpackssdw(t.ymm(), s0.ymm(), s1.ymm());
        ctx->cc.vpermq(t.ymm(), t.ymm(), 0xD8);
        ctx->cc.vpackuswb(t.ymm(), t.ymm(), t.ymm());
        ctx->cc.vpermq(t.ymm(), t.ymm(), 0xD8);
        ctx->cc.vmovdqu(m, t.xmm());
    } else {
        ujit::Vec t = ctx->uc.new_vec128();
        ctx->uc.emit_3v(ujit::UniOpVVV::kPacksI32_I16, t, s0, s1);
        ctx->uc.emit_3v(ujit::UniOpVVV::kPacksI16_U8, t, t, t);
        ctx->uc.emit_mv(ujit::UniOpMV::kStore64_U32, m, t, ujit::Alignment(1));
    }
#elif defined(ASMJIT_UJIT_AARCH64)
    // L3: narrow signed i32 -> signed i16 -> unsigned u8 directly (no dead mov d[1], d[0]).
    ujit::Vec t = ctx->uc.new_vec128();
    ctx->cc.sqxtn(t.h4(), s0.s4());
    ctx->cc.sqxtn2(t.h8(), s1.s4());
    ctx->cc.sqxtun(t.b8(), t.h8());
    ctx->cc.str(t.d(), m);
#endif
}

void zj_vec_store_u8_from_i16(zj_ctx* ctx, zj_mem_t mem, zj_reg_t src_i16) {
    if (!ctx_ok(ctx)) return;
    ujit::Vec s = unpack_vec(src_i16);
    const uint32_t store_bytes = ctx->cpu.f32_lanes * 2u;
    ujit::Mem m = legalise_mem(ctx, mem, store_bytes);
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes == 64u) {
        ujit::Vec t = ctx->uc.new_vec();
        ctx->uc.emit_3v(ujit::UniOpVVV::kMaxI16, t, s, unpack_vec(zj_vec_zero(ctx)));
        ctx->cc.vpmovuswb(t.ymm(), t.zmm());
        ctx->cc.vmovdqu32(m, t.ymm());
    } else if (ctx->cpu.vec_bytes == 32u) {
        ujit::Vec t = ctx->uc.new_vec();
        ctx->cc.vpackuswb(t.ymm(), s.ymm(), s.ymm());
        ctx->cc.vpermq(t.ymm(), t.ymm(), 0xD8);
        ctx->cc.vmovdqu(m, t.xmm());
    } else {
        ujit::Vec t = ctx->uc.new_vec128();
        ctx->uc.emit_3v(ujit::UniOpVVV::kPacksI16_U8, t, s, s);
        ctx->uc.emit_mv(ujit::UniOpMV::kStore64_U32, m, t, ujit::Alignment(1));
    }
#elif defined(ASMJIT_UJIT_AARCH64)
    ujit::Vec t = ctx->uc.new_vec128();
    ctx->cc.sqxtun(t.b8(), s.h8());
    ctx->cc.str(t.d(), m);
#endif
}

void zj_vec_store_u8_from_i16x2(zj_ctx* ctx, zj_mem_t mem, zj_reg_t src0_i16, zj_reg_t src1_i16) {
    if (!ctx_ok(ctx)) return;
    ujit::Vec s0 = unpack_vec(src0_i16);
    ujit::Vec s1 = unpack_vec(src1_i16);
    const uint32_t store_bytes = ctx->cpu.f32_lanes * 4u;
    ujit::Mem m = legalise_mem(ctx, mem, store_bytes);
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes == 64u) {
        emit_x86_zmm_pack_permq_store(ctx, m, s0, s1, true);
        return;
    }
    if (ctx->cpu.vec_bytes == 32u) {
        ujit::Vec t = ctx->uc.new_vec();
        ctx->cc.vpackuswb(t.ymm(), s0.ymm(), s1.ymm());
        ctx->cc.vpermq(t.ymm(), t.ymm(), 0xD8);
        ctx->cc.vmovdqu(m, t.ymm());
        return;
    }
#endif
    ujit::Vec t = ctx->uc.new_vec128();
    ctx->uc.emit_3v(ujit::UniOpVVV::kPacksI16_U8, t, s0, s1);
    ctx->uc.v_storeuvec(m, t);
}

void zj_vec_store_u16_from_i32(zj_ctx* ctx, zj_mem_t mem, zj_reg_t src_i32) {
    if (!ctx_ok(ctx)) return;
    ujit::Vec s = unpack_vec(src_i32);
    const uint32_t store_bytes = ctx->cpu.f32_lanes * 2u;
    ujit::Mem m = legalise_mem(ctx, mem, store_bytes);
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes == 64u) {
        // Caller clamped src_i32 to [0, peak] before converting to i32.
        ujit::Vec t = ctx->uc.new_vec();
        ctx->cc.vpmovusdw(t.ymm(), s.zmm());
        ctx->cc.vmovdqu32(m, t.ymm());
    } else if (ctx->cpu.vec_bytes == 32u) {
        ujit::Vec t = ctx->uc.new_vec();
        ctx->cc.vpackusdw(t.ymm(), s.ymm(), s.ymm());
        ctx->cc.vpermq(t.ymm(), t.ymm(), 0xD8);
        ctx->cc.vmovdqu(m, t.xmm());
    } else {
        ujit::Vec t = ctx->uc.new_vec128();
        ctx->uc.emit_3v(ujit::UniOpVVV::kPacksI32_U16, t, s, s);
        ctx->uc.emit_mv(ujit::UniOpMV::kStore64_U32, m, t, ujit::Alignment(1));
    }
#elif defined(ASMJIT_UJIT_AARCH64)
    ujit::Vec t = ctx->uc.new_vec128();
    ctx->cc.sqxtun(t.h4(), s.s4());
    ctx->cc.str(t.d(), m);
#endif
}

void zj_vec_store_u16_from_i32x2(zj_ctx* ctx, zj_mem_t mem, zj_reg_t src0_i32, zj_reg_t src1_i32) {
    if (!ctx_ok(ctx)) return;
    ujit::Vec s0 = unpack_vec(src0_i32);
    ujit::Vec s1 = unpack_vec(src1_i32);
    const uint32_t store_bytes = ctx->cpu.f32_lanes * 4u;
    ujit::Mem m = legalise_mem(ctx, mem, store_bytes);
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes == 64u) {
        emit_x86_zmm_pack_permq_store(ctx, m, s0, s1, false);
        return;
    }
    if (ctx->cpu.vec_bytes == 32u) {
        ujit::Vec t = ctx->uc.new_vec();
        ctx->cc.vpackusdw(t.ymm(), s0.ymm(), s1.ymm());
        ctx->cc.vpermq(t.ymm(), t.ymm(), 0xD8);
        ctx->cc.vmovdqu(m, t.ymm());
        return;
    }
#endif
    ujit::Vec t = ctx->uc.new_vec128();
    ctx->uc.emit_3v(ujit::UniOpVVV::kPacksI32_U16, t, s0, s1);
    ctx->uc.v_storeuvec(m, t);
}

uint8_t zj_cpu_supports_fast_lut_u8(const zj_cpu_info_t* cpu) {
    if (cpu == nullptr || cpu->isa == ZJ_ISA_SCALAR) return 0u;
#if defined(ASMJIT_UJIT_X86)
    if (cpu->vec_bytes == 64u) {
        return aj::CpuInfo::host().features().x86().has_avx512_vbmi() ? 1u : 0u;
    }
    return cpu->has_sse4_1 ? 1u : 0u;
#elif defined(ASMJIT_UJIT_AARCH64)
    return 1u;
#else
    return 0u;
#endif
}

void zj_vec_lut_u8_256(zj_ctx* ctx, zj_reg_t dst_reg, zj_reg_t idx_reg, const uint8_t* table_256) {
    if (!ctx_ok(ctx) || table_256 == nullptr) return;
    ujit::Vec dst = unpack_vec(dst_reg);
    ujit::Vec idx = unpack_vec(idx_reg);
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes == 64u && aj::CpuInfo::host().features().x86().has_avx512_vbmi()) {
        // AVX-512 VBMI: 2x vpermi2b (128 bytes/lookup each) + 1 masked move = 64 pixels per vector (2x faster than 4x vpermb).
        ujit::Vec t0 = unpack_vec(intern_const_vec(ctx, table_256 + 0, 64u));
        ujit::Vec t1 = unpack_vec(intern_const_vec(ctx, table_256 + 64, 64u));
        ujit::Vec t2 = unpack_vec(intern_const_vec(ctx, table_256 + 128, 64u));
        ujit::Vec t3 = unpack_vec(intern_const_vec(ctx, table_256 + 192, 64u));
        uint8_t b128[64];
        std::memset(b128, 0x80, 64);
        ujit::Vec c128 = unpack_vec(intern_const_vec(ctx, b128, 64u));

        aj::x86::KReg k = ctx->cc.new_kq();
        ctx->cc.vptestmb(k, idx.zmm(), c128.zmm());

        ujit::Vec r0 = ctx->uc.new_vec();
        ujit::Vec r1 = ctx->uc.new_vec();
        ctx->cc.vmovdqa64(r0.zmm(), idx.zmm());
        ctx->cc.vpermi2b(r0.zmm(), t0.zmm(), t1.zmm());
        ctx->cc.vmovdqa64(r1.zmm(), idx.zmm());
        ctx->cc.vpermi2b(r1.zmm(), t2.zmm(), t3.zmm());
        ctx->cc.k(k).vmovdqu8(r0.zmm(), r1.zmm());
        ctx->uc.emit_2v(ujit::UniOpVV::kMov, dst, r0);
        return;
    }

    // AVX2 (32B) or SSE4.1 (16B): depth-first 4-level vpshufb + vpblendvb binary tree (0 spills!).
    // Note: AsmJit's ConstPool max single-entry size is 64 bytes, so intern each 16B/32B chunk individually.
    const uint32_t vb = ctx->cpu.vec_bytes;
    if (vb > 32u || !ctx->cpu.has_sse4_1) {
        set_shim_error(ctx, aj::Error::kFeatureNotEnabled, "vec_lut_u8_256 requires AVX-512 VBMI, AVX2, or SSE4.1");
        return;
    }
    ujit::Mem chunk_mems[16];
    for (uint32_t c = 0; c < 16u; ++c) {
        chunk_mems[c] = ctx->cc.new_const(aj::ConstPoolScope::kLocal, table_256 + c * 16u, 16u);
    }

    uint8_t m0f_raw[32];
    std::memset(m0f_raw, 0x0F, 32);
    ujit::Vec m0f = unpack_vec(intern_const_vec(ctx, m0f_raw, vb));

    ujit::Vec lo = ctx->uc.new_vec();
    ctx->uc.emit_3v(ujit::UniOpVVV::kAndU32, lo, idx, m0f);
    ujit::Vec b6 = ctx->uc.new_vec();
    ujit::Vec b5 = ctx->uc.new_vec();
    ujit::Vec b4 = ctx->uc.new_vec();
    ctx->uc.emit_3v(ujit::UniOpVVV::kAddU8, b6, idx, idx);
    ctx->uc.emit_3v(ujit::UniOpVVV::kAddU8, b5, b6, b6);
    ctx->uc.emit_3v(ujit::UniOpVVV::kAddU8, b4, b5, b5);

    auto load_shuf = [&](uint32_t c) -> ujit::Vec {
        ujit::Vec v = ctx->uc.new_vec();
        if (vb == 32u) {
            ctx->cc.vbroadcasti128(v.ymm(), chunk_mems[c]);
        } else {
            ctx->uc.v_loadavec(v, chunk_mems[c]);
        }
        ctx->uc.emit_3v(ujit::UniOpVVV::kSwizzlev_U8, v, v, lo);
        return v;
    };
    auto eval_pair = [&](uint32_t c0) -> ujit::Vec {
        ujit::Vec a = load_shuf(c0);
        ujit::Vec b = load_shuf(c0 + 1u);
        ctx->uc.emit_4v(ujit::UniOpVVVV::kBlendV_U8, a, a, b, b4);
        return a;
    };
    auto eval_quad = [&](uint32_t c0) -> ujit::Vec {
        ujit::Vec a = eval_pair(c0);
        ujit::Vec b = eval_pair(c0 + 2u);
        ctx->uc.emit_4v(ujit::UniOpVVVV::kBlendV_U8, a, a, b, b5);
        return a;
    };
    auto eval_oct = [&](uint32_t c0) -> ujit::Vec {
        ujit::Vec a = eval_quad(c0);
        ujit::Vec b = eval_quad(c0 + 4u);
        ctx->uc.emit_4v(ujit::UniOpVVVV::kBlendV_U8, a, a, b, b6);
        return a;
    };

    ujit::Vec lo_half = eval_oct(0u);
    ujit::Vec hi_half = eval_oct(8u);
    ctx->uc.emit_4v(ujit::UniOpVVVV::kBlendV_U8, dst, lo_half, hi_half, idx);
#elif defined(ASMJIT_UJIT_AARCH64)
    // AArch64 NEON: `tbl` (kSwizzlev_U8) returns 0 for any lane >= 16 unsigned, so
    // `OR_{c=0..15} tbl(chunk[c], idx - 16*c)` is exact with zero blends!
    // Intern each 16B chunk individually (ConstPool max entry size is 64B).
    ujit::Mem chunk_mems[16];
    for (uint32_t c = 0; c < 16u; ++c) {
        chunk_mems[c] = ctx->cc.new_const(aj::ConstPoolScope::kLocal, table_256 + c * 16u, 16u);
    }
    uint8_t step_raw[16];
    std::memset(step_raw, 16u, 16u);
    ujit::Vec v_step = unpack_vec(intern_const_vec(ctx, step_raw, 16u));

    ujit::Vec cur_idx = ctx->uc.new_vec();
    ctx->uc.emit_2v(ujit::UniOpVV::kMov, cur_idx, idx);

    ujit::Vec acc = ctx->uc.new_vec();
    ujit::Vec tmp = ctx->uc.new_vec();
    for (uint32_t c = 0; c < 16u; ++c) {
        ctx->uc.v_loadavec(tmp, chunk_mems[c]);
        ctx->uc.emit_3v(ujit::UniOpVVV::kSwizzlev_U8, tmp, tmp, cur_idx);
        if (c == 0u) {
            ctx->uc.emit_2v(ujit::UniOpVV::kMov, acc, tmp);
        } else {
            ctx->uc.emit_3v(ujit::UniOpVVV::kOrU32, acc, acc, tmp);
        }
        if (c + 1u < 16u) {
            ctx->uc.emit_3v(ujit::UniOpVVV::kSubU8, cur_idx, cur_idx, v_step);
        }
    }
    ctx->uc.emit_2v(ujit::UniOpVV::kMov, dst, acc);
#endif
}

static ujit::Mem make_f32_const_mem(zj_ctx* ctx, float val) noexcept {
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes == 64u) {
        return ctx->cc.new_const(aj::ConstPoolScope::kLocal, &val, 4u)
            .clone_broadcasted(aj::x86::Mem::Broadcast::k1To16);
    }
#endif
    uint8_t raw[64]{};
    const uint32_t lanes = ctx->cpu.f32_lanes;
    for (uint32_t i = 0; i < lanes; ++i) {
        std::memcpy(raw + i * 4u, &val, 4u);
    }
    return ctx->cc.new_const(aj::ConstPoolScope::kLocal, raw, ctx->cpu.vec_bytes);
}

// dst = a <op> splat(b_const). AVX2/AVX-512 fold the constant as a memory operand ({1to16} on AVX-512).
static void emit_op_c_f32(zj_ctx* ctx, ujit::UniOpVVV uop, aj::InstId x86_id, zj_reg_t dst_reg, zj_reg_t a_reg, float b_const) noexcept {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes >= 32u) {
        ctx->cc.emit(x86_id, vw(ctx, unpack_vec(dst_reg)), vw(ctx, unpack_vec(a_reg)), make_f32_const_mem(ctx, b_const));
        return;
    }
#else
    (void)x86_id;
#endif
    ctx->uc.emit_3v(uop, unpack_vec(dst_reg), unpack_vec(a_reg), unpack_vec(zj_const_f32_splat(ctx, b_const)));
}

void zj_vec_mul_c_f32(zj_ctx* ctx, zj_reg_t dst_reg, zj_reg_t a_reg, float b_const) {
    emit_op_c_f32(ctx, ujit::UniOpVVV::kMulF32, ZJ_X86_ID(Vmulps), dst_reg, a_reg, b_const);
}

void zj_vec_add_c_f32(zj_ctx* ctx, zj_reg_t dst_reg, zj_reg_t a_reg, float b_const) {
    emit_op_c_f32(ctx, ujit::UniOpVVV::kAddF32, ZJ_X86_ID(Vaddps), dst_reg, a_reg, b_const);
}

void zj_vec_sub_c_f32(zj_ctx* ctx, zj_reg_t dst_reg, zj_reg_t a_reg, float b_const) {
    emit_op_c_f32(ctx, ujit::UniOpVVV::kSubF32, ZJ_X86_ID(Vsubps), dst_reg, a_reg, b_const);
}

void zj_vec_min_c_f32(zj_ctx* ctx, zj_reg_t dst_reg, zj_reg_t a_reg, float b_const) {
    emit_op_c_f32(ctx, ujit::UniOpVVV::kMinF32, ZJ_X86_ID(Vminps), dst_reg, a_reg, b_const);
}

void zj_vec_max_c_f32(zj_ctx* ctx, zj_reg_t dst_reg, zj_reg_t a_reg, float b_const) {
    emit_op_c_f32(ctx, ujit::UniOpVVV::kMaxF32, ZJ_X86_ID(Vmaxps), dst_reg, a_reg, b_const);
}

void zj_vec_cmp_c_f32(zj_ctx* ctx, zj_vec_op3_t cmp_op, zj_reg_t dst_reg, zj_reg_t a_reg, float b_const) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes >= 32u) {
        uint8_t imm = 0u;
        bool valid = true;
        switch (cmp_op) {
            case ZJ_V3_CMP_EQ_F32: imm = 0u; break;   // _CMP_EQ_OQ
            case ZJ_V3_CMP_LT_F32: imm = 17u; break;  // _CMP_LT_OQ
            case ZJ_V3_CMP_LE_F32: imm = 18u; break;  // _CMP_LE_OQ
            case ZJ_V3_CMP_GE_F32: imm = 29u; break;  // _CMP_GE_OQ
            case ZJ_V3_CMP_GT_F32: imm = 30u; break;  // _CMP_GT_OQ
            default: valid = false; break;
        }
        if (valid) {
            ujit::Vec d = unpack_vec(dst_reg);
            ujit::Vec a = unpack_vec(a_reg);
            ujit::Mem m = make_f32_const_mem(ctx, b_const);
            if (ctx->cpu.vec_bytes == 64u) {
                aj::x86::KReg k = ctx->cc.new_kw();
                ctx->cc.vcmpps(k, a.zmm(), m, imm);
                ctx->cc.vpmovm2d(d.zmm(), k);
            } else {
                ctx->cc.vcmpps(d.ymm(), a.ymm(), m, imm);
            }
            return;
        }
    }
#endif
    zj_vec_op3(ctx, cmp_op, dst_reg, a_reg, zj_const_f32_splat(ctx, b_const));
}

#if defined(ASMJIT_UJIT_X86)
// dst = c + a * m (vfmadd231ps) or c - a * m (vfnmadd231ps). Guard dst == a && dst != c before copying c into acc.
static void emit_x86_fma231(zj_ctx* ctx, aj::InstId id, zj_reg_t dst_reg, zj_reg_t a_reg, const ujit::Mem& m, zj_reg_t c_reg) noexcept {
    ujit::Vec d = unpack_vec(dst_reg);
    ujit::Vec a = unpack_vec(a_reg);
    ujit::Vec c = unpack_vec(c_reg);
    ujit::Vec acc = (d.id() == a.id() && d.id() != c.id()) ? ctx->uc.new_vec() : d;
    if (acc.id() != c.id()) ctx->cc.vmovaps(vw(ctx, acc), vw(ctx, c));
    ctx->cc.emit(id, vw(ctx, acc), vw(ctx, a), m);
    if (acc.id() != d.id()) ctx->cc.vmovaps(vw(ctx, d), vw(ctx, acc));
}
#endif

void zj_vec_fma_ac_f32(zj_ctx* ctx, zj_reg_t dst_reg, zj_reg_t a_reg, float b_const, zj_reg_t c_reg) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.has_fma && ctx->cpu.vec_bytes >= 32u) {
        emit_x86_fma231(ctx, aj::x86::Inst::kIdVfmadd231ps, dst_reg, a_reg, make_f32_const_mem(ctx, b_const), c_reg);
        return;
    }
#endif
    ctx->uc.emit_4v(ujit::UniOpVVVV::kMAddF32, unpack_vec(dst_reg), unpack_vec(a_reg), unpack_vec(zj_const_f32_splat(ctx, b_const)), unpack_vec(c_reg));
}

void zj_vec_blendv_f32(zj_ctx* ctx, zj_reg_t dst_reg, zj_reg_t false_val, zj_reg_t true_val, zj_reg_t mask_reg) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes == 64u) {
        emit_x86_ternlog_blend(ctx, unpack_vec(dst_reg), unpack_vec(false_val), unpack_vec(true_val), unpack_vec(mask_reg));
        return;
    }
    if (ctx->cpu.vec_bytes == 32u) {
        ctx->cc.vblendvps(unpack_vec(dst_reg).ymm(), unpack_vec(false_val).ymm(), unpack_vec(true_val).ymm(), unpack_vec(mask_reg).ymm());
        return;
    }
#elif defined(ASMJIT_UJIT_AARCH64)
    emit_a64_bsl_blend(ctx, unpack_vec(dst_reg), unpack_vec(false_val), unpack_vec(true_val), unpack_vec(mask_reg));
    return;
#endif
    ctx->uc.emit_4v(ujit::UniOpVVVV::kBlendV_U8, unpack_vec(dst_reg), unpack_vec(false_val), unpack_vec(true_val), unpack_vec(mask_reg));
}

void zj_vec_and_f32(zj_ctx* ctx, zj_reg_t dst_reg, zj_reg_t a_reg, zj_reg_t b_reg) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes >= 32u) {
        ctx->cc.vandps(vw(ctx, unpack_vec(dst_reg)), vw(ctx, unpack_vec(a_reg)), vw(ctx, unpack_vec(b_reg)));
        return;
    }
#endif
    ctx->uc.emit_3v(ujit::UniOpVVV::kAndU32, unpack_vec(dst_reg), unpack_vec(a_reg), unpack_vec(b_reg));
}

void zj_vec_andn_f32(zj_ctx* ctx, zj_reg_t dst_reg, zj_reg_t mask_not, zj_reg_t b_reg) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes >= 32u) {
        ctx->cc.vandnps(vw(ctx, unpack_vec(dst_reg)), vw(ctx, unpack_vec(mask_not)), vw(ctx, unpack_vec(b_reg)));
        return;
    }
#endif
    ctx->uc.emit_3v(ujit::UniOpVVV::kAndnU32, unpack_vec(dst_reg), unpack_vec(mask_not), unpack_vec(b_reg));
}

void zj_vec_fma_bc_f32(zj_ctx* ctx, zj_reg_t dst_reg, zj_reg_t a_reg, zj_reg_t b_reg, float c_const) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.has_fma && ctx->cpu.vec_bytes >= 32u) {
        ujit::Vec d = unpack_vec(dst_reg);
        ujit::Vec a = unpack_vec(a_reg);
        ujit::Vec b = unpack_vec(b_reg);
        ujit::Mem mc = make_f32_const_mem(ctx, c_const);
        // L1: dst = a * b + [mc]. Guard dst == b (clobbering b before vfmadd213ps).
        ujit::Vec acc = d;
        ujit::Vec rhs = b;
        if (d.id() == b.id() && d.id() != a.id()) {
            // Commutative: a * b == b * a, so keep d (= b) as acc and use a as rhs!
            rhs = a;
        } else if (d.id() != a.id()) {
            ctx->cc.vmovaps(vw(ctx, acc), vw(ctx, a));
        }
        ctx->cc.vfmadd213ps(vw(ctx, acc), vw(ctx, rhs), mc);
        return;
    }
#endif
    ctx->uc.emit_4v(ujit::UniOpVVVV::kMAddF32, unpack_vec(dst_reg), unpack_vec(a_reg), unpack_vec(b_reg), unpack_vec(zj_const_f32_splat(ctx, c_const)));
}

void zj_vec_fma_cc_f32(zj_ctx* ctx, zj_reg_t dst_reg, zj_reg_t a_reg, float b_const, float c_const) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.has_fma && ctx->cpu.vec_bytes >= 32u) {
        ujit::Vec d = unpack_vec(dst_reg);
        ujit::Vec a = unpack_vec(a_reg);
        ujit::Mem mc = make_f32_const_mem(ctx, c_const);
        // L1: guard dst == a before loading [mb] into acc.
        ujit::Vec acc = (d.id() == a.id()) ? ctx->uc.new_vec() : d;
        if (ctx->cpu.vec_bytes == 64u) {
            // Hoist b_const once in prologue (4B const pool, 0-cycle vmovaps rename, avoids in-loop Zn4FP1 vbroadcastss).
            ujit::Vec bv = unpack_vec(zj_const_f32_splat(ctx, b_const));
            ctx->cc.vmovaps(acc.zmm(), bv.zmm());
            ctx->cc.vfmadd213ps(acc.zmm(), a.zmm(), mc);
            if (acc.id() != d.id()) ctx->cc.vmovaps(d.zmm(), acc.zmm());
        } else {
            ujit::Mem mb = make_f32_const_mem(ctx, b_const);
            ctx->uc.v_loadavec(acc, mb);
            ctx->cc.vfmadd213ps(acc.ymm(), a.ymm(), mc);
            if (acc.id() != d.id()) ctx->cc.vmovaps(d.ymm(), acc.ymm());
        }
        return;
    }
#endif
    ctx->uc.emit_4v(
        ujit::UniOpVVVV::kMAddF32,
        unpack_vec(dst_reg),
        unpack_vec(a_reg),
        unpack_vec(zj_const_f32_splat(ctx, b_const)),
        unpack_vec(zj_const_f32_splat(ctx, c_const))
    );
}

void zj_vec_fnma_bc_f32(zj_ctx* ctx, zj_reg_t dst_reg, zj_reg_t a_reg, float b_const, zj_reg_t c_reg) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.has_fma && ctx->cpu.vec_bytes >= 32u) {
        emit_x86_fma231(ctx, aj::x86::Inst::kIdVfnmadd231ps, dst_reg, a_reg, make_f32_const_mem(ctx, b_const), c_reg);
        return;
    }
#endif
    if (!ctx->cpu.has_fma) {
        ujit::Vec tmp = ctx->uc.new_vec();
        ctx->uc.emit_3v(ujit::UniOpVVV::kMulF32, tmp, unpack_vec(a_reg), unpack_vec(zj_const_f32_splat(ctx, b_const)));
        ctx->uc.emit_3v(ujit::UniOpVVV::kSubF32, unpack_vec(dst_reg), unpack_vec(c_reg), tmp);
        return;
    }
    ctx->uc.emit_4v(ujit::UniOpVVVV::kNMAddF32, unpack_vec(dst_reg), unpack_vec(a_reg), unpack_vec(zj_const_f32_splat(ctx, b_const)), unpack_vec(c_reg));
}

void zj_vec_fma_mem_f32(zj_ctx* ctx, zj_reg_t dst_reg, zj_reg_t a_reg, zj_mem_t mem, zj_reg_t c_reg) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.has_fma && ctx->cpu.vec_bytes >= 32u) {
        emit_x86_fma231(ctx, aj::x86::Inst::kIdVfmadd231ps, dst_reg, a_reg, legalise_mem(ctx, mem, ctx->cpu.vec_bytes), c_reg);
        return;
    }
#endif
    ujit::Vec tmp = ctx->uc.new_vec();
    ctx->uc.v_loaduvec(tmp, legalise_mem(ctx, mem, ctx->cpu.vec_bytes));
    ctx->uc.emit_4v(ujit::UniOpVVVV::kMAddF32, unpack_vec(dst_reg), unpack_vec(a_reg), tmp, unpack_vec(c_reg));
}

void zj_vec_fnma_mem_f32(zj_ctx* ctx, zj_reg_t dst_reg, zj_reg_t a_reg, zj_mem_t mem, zj_reg_t c_reg) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.has_fma && ctx->cpu.vec_bytes >= 32u) {
        emit_x86_fma231(ctx, aj::x86::Inst::kIdVfnmadd231ps, dst_reg, a_reg, legalise_mem(ctx, mem, ctx->cpu.vec_bytes), c_reg);
        return;
    }
#endif
    ujit::Vec tmp = ctx->uc.new_vec();
    ctx->uc.v_loaduvec(tmp, legalise_mem(ctx, mem, ctx->cpu.vec_bytes));
    zj_vec_op4(ctx, ZJ_V4_NMADD_F32, dst_reg, a_reg, pack_reg(tmp), c_reg);
}

void zj_vec_permute_f32(zj_ctx* ctx, zj_reg_t dst, zj_reg_t src, const uint32_t* lane_indices, uint32_t lane_count) {
    if (!ctx_ok(ctx)) return;
    if (lane_indices == nullptr || lane_count != ctx->cpu.f32_lanes) {
        set_shim_error(ctx, aj::Error::kInvalidArgument, "vec_permute_f32 lane_count mismatch");
        return;
    }
    ujit::Vec d = unpack_vec(dst);
    ujit::Vec s = unpack_vec(src);
    if (lane_count == 4u) {
        const uint32_t imm = ujit::swizzle(
            uint8_t(lane_indices[3] & 3u),
            uint8_t(lane_indices[2] & 3u),
            uint8_t(lane_indices[1] & 3u),
            uint8_t(lane_indices[0] & 3u)
        ).value;
        ctx->uc.emit_2vi(ujit::UniOpVVI::kSwizzleU32x4, d, s, imm);
        return;
    }
#if defined(ASMJIT_UJIT_X86)
    uint8_t idx_bytes[64]{};
    for (uint32_t i = 0; i < lane_count; ++i) {
        const uint32_t v = lane_indices[i];
        std::memcpy(idx_bytes + i * 4u, &v, 4u);
    }
    // Load shuffle indices locally instead of hoisting into _func_init_hook:
    // permute is only used on cold left/right edge peels, so hoisting it would
    // keep cold index vectors live across the hot interior loop and cause spills.
    ujit::Mem idx_mem = ctx->cc.new_const(aj::ConstPoolScope::kLocal, idx_bytes, ctx->cpu.vec_bytes);
    ujit::Vec idx = ctx->uc.new_vec();
    ctx->uc.v_loadavec(idx, idx_mem);
    if (lane_count == 8u) {
        ctx->cc.vpermps(d.ymm(), idx.ymm(), s.ymm());
    } else if (lane_count == 16u) {
        ctx->cc.vpermps(d.zmm(), idx.zmm(), s.zmm());
    }
#endif
}

static void emit_scalar_gather_impl(zj_ctx* ctx, zj_reg_t dst, zj_reg_t base_reg, zj_reg_t offsets_reg, uint32_t elem_bytes) noexcept {
    const uint32_t lanes = ctx->cpu.f32_lanes;
    const uint32_t vec_bytes = ctx->cpu.vec_bytes;
    ujit::Mem off_slot = ctx->cc.new_stack(vec_bytes, vec_bytes);
    ujit::Mem res_slot = ctx->cc.new_stack(vec_bytes, vec_bytes);
    ctx->uc.v_storeavec(off_slot, unpack_vec(offsets_reg));

    ujit::Gp base = unpack_gp(base_reg);
    ujit::Gp off = ctx->uc.new_gp64();
    ujit::Gp val = ctx->uc.new_gp32();

    for (uint32_t i = 0; i < lanes; ++i) {
        const int32_t slot_disp = int32_t(i * 4u);
        ujit::Mem off_m = off_slot;
        off_m.add_offset(slot_disp);
        // Zero-extended 32-bit offset into 64-bit register (callers clamp offsets >= 0).
        ctx->uc.emit_rm(ujit::UniOpRM::kLoadU32, off.r32(), off_m);

        zj_mem_t src_m{};
        src_m.base = pack_reg(base);
        src_m.index = pack_reg(off);
        src_m.has_index = 1u;
        src_m.size = uint8_t(elem_bytes);
        ujit::UniOpRM load_op = ujit::UniOpRM::kLoadU32;
        if (elem_bytes == 1u) load_op = ujit::UniOpRM::kLoadU8;
        else if (elem_bytes == 2u) load_op = ujit::UniOpRM::kLoadU16;
        ctx->uc.emit_rm(load_op, val, legalise_mem(ctx, src_m, elem_bytes));

        ujit::Mem res_m = res_slot;
        res_m.add_offset(slot_disp);
        ctx->uc.emit_mr(ujit::UniOpMR::kStoreU32, res_m, val);
    }

    ctx->uc.v_loadavec(unpack_vec(dst), res_slot);
}

#if defined(ASMJIT_UJIT_X86)
static void emit_x86_reg_gather_impl(zj_ctx* ctx, zj_reg_t dst, zj_reg_t base_reg, zj_reg_t offsets_reg, uint32_t elem_bytes, bool is_f16) noexcept {
    // Register-only 8-lane (AVX2) / 16-lane (AVX-512) gather: 64-bit extracts (vmovq/vpextrq), 0 stack spills, 0 STL stalls.
    const bool is_zmm = (ctx->cpu.vec_bytes == 64u);
    ujit::Gp base = unpack_gp(base_reg);
    ujit::Vec off = unpack_vec(offsets_reg);
    ujit::Vec d = unpack_vec(dst);

    ujit::Gp p01 = ctx->uc.new_gp64();
    ujit::Gp p23 = ctx->uc.new_gp64();
    ujit::Gp i0 = ctx->uc.new_gp64();
    ujit::Gp i2 = ctx->uc.new_gp64();
    ujit::Gp tmp = ctx->uc.new_gp32();

    // Offsets of 4 lanes -> i0, p01 (lane 1), i2, p23 (lane 3), each zero-extended from 32 bits.
    auto extract_offsets = [&](const ujit::Vec& off_x) {
        ctx->cc.vmovq(p01, off_x);
        ctx->cc.vpextrq(p23, off_x, 1);
        ctx->cc.mov(i0.r32(), p01.r32());
        ctx->cc.shr(p01, 32);
        ctx->cc.mov(i2.r32(), p23.r32());
        ctx->cc.shr(p23, 32);
    };
    auto emit_quarter = [&](const ujit::Vec& off_x, const ujit::Vec& out_x) {
        extract_offsets(off_x);
        if (elem_bytes == 4u) {
            ctx->cc.vmovss(out_x, aj::x86::dword_ptr(base, i0));
            ctx->cc.vinsertps(out_x, out_x, aj::x86::dword_ptr(base, p01), 0x10);
            ctx->cc.vinsertps(out_x, out_x, aj::x86::dword_ptr(base, i2), 0x20);
            ctx->cc.vinsertps(out_x, out_x, aj::x86::dword_ptr(base, p23), 0x30);
        } else if (elem_bytes == 2u) {
            ctx->cc.movzx(tmp, aj::x86::word_ptr(base, i0));
            ctx->cc.vmovd(out_x, tmp);
            ctx->cc.vpinsrw(out_x, out_x, aj::x86::word_ptr(base, p01), 1);
            ctx->cc.vpinsrw(out_x, out_x, aj::x86::word_ptr(base, i2), 2);
            ctx->cc.vpinsrw(out_x, out_x, aj::x86::word_ptr(base, p23), 3);
        } else {
            ctx->cc.movzx(tmp, aj::x86::byte_ptr(base, i0));
            ctx->cc.vmovd(out_x, tmp);
            ctx->cc.vpinsrb(out_x, out_x, aj::x86::byte_ptr(base, p01), 1);
            ctx->cc.vpinsrb(out_x, out_x, aj::x86::byte_ptr(base, i2), 2);
            ctx->cc.vpinsrb(out_x, out_x, aj::x86::byte_ptr(base, p23), 3);
        }
    };

    if (!is_zmm) {
        ujit::Vec hi_off = ctx->uc.new_vec();
        ctx->cc.vextracti128(hi_off.xmm(), off.ymm(), 1);
        ujit::Vec lo_v = ctx->uc.new_vec();
        ujit::Vec hi_v = ctx->uc.new_vec();
        emit_quarter(off.xmm(), lo_v.xmm());
        emit_quarter(hi_off.xmm(), hi_v.xmm());

        if (elem_bytes == 4u) {
            ctx->cc.vinsertf128(d.ymm(), lo_v.ymm(), hi_v.xmm(), 1);
        } else if (elem_bytes == 2u) {
            ctx->cc.vpunpcklqdq(lo_v.xmm(), lo_v.xmm(), hi_v.xmm());
            if (is_f16) ctx->cc.vcvtph2ps(d.ymm(), lo_v.xmm());
            else ctx->cc.vpmovzxwd(d.ymm(), lo_v.xmm());
        } else {
            ctx->cc.vpunpckldq(lo_v.xmm(), lo_v.xmm(), hi_v.xmm());
            ctx->cc.vpmovzxbd(d.ymm(), lo_v.xmm());
        }
        return;
    }

    // AVX-512 16-lane register gather: 4 quarters via EVEX vextracti32x4.
    // For u8/u16/f16, pack 4 lanes in GPRs before vmovd/vpinsrd or vmovq/vpinsrq
    // (cuts Zn4FP45 GPR->FP transfers 4x: 2.2x faster on Zen 4/5 with 0 stack spills).
    ujit::Vec cur_off = ctx->uc.new_vec();
    if (elem_bytes == 4u) {
        ujit::Vec q0 = ctx->uc.new_vec();
        ujit::Vec q1 = ctx->uc.new_vec();
        ujit::Vec q2 = ctx->uc.new_vec();
        ujit::Vec q3 = ctx->uc.new_vec();
        emit_quarter(off.xmm(), q0.xmm());
        ctx->cc.vextracti32x4(cur_off.xmm(), off.zmm(), 1);
        emit_quarter(cur_off.xmm(), q1.xmm());
        ctx->cc.vextracti32x4(cur_off.xmm(), off.zmm(), 2);
        emit_quarter(cur_off.xmm(), q2.xmm());
        ctx->cc.vextracti32x4(cur_off.xmm(), off.zmm(), 3);
        emit_quarter(cur_off.xmm(), q3.xmm());
        ctx->cc.vinsertf32x4(q0.ymm(), q0.ymm(), q1.xmm(), 1);
        ctx->cc.vinsertf32x4(q2.ymm(), q2.ymm(), q3.xmm(), 1);
        ctx->cc.vinsertf64x4(d.zmm(), q0.zmm(), q2.ymm(), 1);
        return;
    }

    ujit::Gp w0 = ctx->uc.new_gp64();
    ujit::Gp w1 = ctx->uc.new_gp64();
    auto extract_quarter_offsets = [&](uint32_t q) {
        if (q == 0u) {
            extract_offsets(off.xmm());
        } else {
            ctx->cc.vextracti32x4(cur_off.xmm(), off.zmm(), q);
            extract_offsets(cur_off.xmm());
        }
    };
    if (elem_bytes == 1u) {
        ujit::Vec packed = ctx->uc.new_vec();
        for (uint32_t q = 0; q < 4u; ++q) {
            extract_quarter_offsets(q);
            ctx->cc.movzx(w0.r32(), aj::x86::byte_ptr(base, i0));
            ctx->cc.movzx(w1.r32(), aj::x86::byte_ptr(base, p01));
            ctx->cc.shl(w1.r32(), 8);
            ctx->cc.or_(w0.r32(), w1.r32());
            ctx->cc.movzx(i0.r32(), aj::x86::byte_ptr(base, i2));
            ctx->cc.movzx(p01.r32(), aj::x86::byte_ptr(base, p23));
            ctx->cc.shl(p01.r32(), 8);
            ctx->cc.or_(i0.r32(), p01.r32());
            ctx->cc.shl(i0.r32(), 16);
            ctx->cc.or_(w0.r32(), i0.r32());
            if (q == 0u) ctx->cc.vmovd(packed.xmm(), w0.r32());
            else ctx->cc.vpinsrd(packed.xmm(), packed.xmm(), w0.r32(), q);
        }
        ctx->cc.vpmovzxbd(d.zmm(), packed.xmm());
    } else {
        ujit::Vec lo_x = ctx->uc.new_vec();
        ujit::Vec hi_x = ctx->uc.new_vec();
        for (uint32_t q = 0; q < 4u; ++q) {
            extract_quarter_offsets(q);
            ctx->cc.movzx(w0.r32(), aj::x86::word_ptr(base, i0));
            ctx->cc.movzx(w1.r32(), aj::x86::word_ptr(base, p01));
            ctx->cc.shl(w1.r32(), 16);
            ctx->cc.or_(w0.r32(), w1.r32());
            ctx->cc.movzx(i0.r32(), aj::x86::word_ptr(base, i2));
            ctx->cc.movzx(p01.r32(), aj::x86::word_ptr(base, p23));
            ctx->cc.shl(p01.r32(), 16);
            ctx->cc.or_(i0.r32(), p01.r32());
            ctx->cc.shl(i0, 32);
            ctx->cc.or_(w0, i0);
            if (q == 0u) ctx->cc.vmovq(lo_x.xmm(), w0);
            else if (q == 1u) ctx->cc.vpinsrq(lo_x.xmm(), lo_x.xmm(), w0, 1);
            else if (q == 2u) ctx->cc.vmovq(hi_x.xmm(), w0);
            else ctx->cc.vpinsrq(hi_x.xmm(), hi_x.xmm(), w0, 1);
        }
        ctx->cc.vinserti32x4(lo_x.ymm(), lo_x.ymm(), hi_x.xmm(), 1);
        if (is_f16) ctx->cc.vcvtph2ps(d.zmm(), lo_x.ymm());
        else ctx->cc.vpmovzxwd(d.zmm(), lo_x.ymm());
    }
}
#endif

#if defined(ASMJIT_UJIT_AARCH64)
// Register-only 4-lane NEON gather via umov + add + ldr(lane 0)/ld1(lanes 1..3).
// Loads directly from L1 into NEON lanes with zero stack spills and zero GP->SIMD ins transfers
// (+20%..+30% faster than ldr+ins on multi-tap gathers on Apple Silicon).
static void emit_a64_reg_gather_ld1(zj_ctx* ctx, zj_reg_t dst, zj_reg_t base_reg, zj_reg_t offsets_reg, uint32_t elem_bytes, bool is_f16) noexcept {
    ujit::Gp base = unpack_gp(base_reg);
    ujit::Vec off = unpack_vec(offsets_reg);
    ujit::Vec d = unpack_vec(dst);

    // Move the 4 u32 offsets out as two 64-bit pairs (`fmov` + `umov .d[1]`) instead of four
    // `umov .s[i]`: vector->GP transfers issue on the V pipes, which gather-heavy kernels saturate
    // on Neoverse, while the extended/shifted `add`s run on integer pipes. Each offset is still
    // zero-extended from 32 bits (`uxtw` / `lsr #32`), exactly like `umov w`.
    ujit::Gp lo = ctx->uc.new_gp64();
    ujit::Gp hi = ctx->uc.new_gp64();
    ujit::Gp a1 = ctx->uc.new_gp64();
    ujit::Gp a2 = ctx->uc.new_gp64();
    ujit::Gp a3 = ctx->uc.new_gp64();
    ctx->cc.fmov(lo, off.d());
    ctx->cc.umov(hi, off.d(1));
    ctx->cc.add(a1, base, lo, aj::a64::lsr(32));
    ctx->cc.add(a2, base, hi.r32(), aj::a64::uxtw(0));
    ctx->cc.add(a3, base, hi, aj::a64::lsr(32));
    const aj::a64::Mem m0 = aj::a64::ptr(base, lo.r32(), aj::a64::uxtw(0));

    if (elem_bytes == 4u) {
        ctx->cc.ldr(d.s(), m0);
        ctx->cc.ld1(d.s(1), aj::a64::ptr(a1));
        ctx->cc.ld1(d.s(2), aj::a64::ptr(a2));
        ctx->cc.ld1(d.s(3), aj::a64::ptr(a3));
    } else if (elem_bytes == 2u) {
        if (is_f16) {
            ujit::Vec tmp = ctx->uc.new_vec128();
            ctx->cc.ldr(tmp.h(), m0);
            ctx->cc.ld1(tmp.h(1), aj::a64::ptr(a1));
            ctx->cc.ld1(tmp.h(2), aj::a64::ptr(a2));
            ctx->cc.ld1(tmp.h(3), aj::a64::ptr(a3));
            ctx->cc.fcvtl(d.s4(), tmp.h4());
        } else {
            // AArch64 `ldr h` zeroes bits [16..127] of `d`, so loading directly into `h(2/4/6)`
            // yields zero-extended i32 lanes with zero `movi` and zero `ushll`.
            ctx->cc.ldr(d.h(), m0);
            ctx->cc.ld1(d.h(2), aj::a64::ptr(a1));
            ctx->cc.ld1(d.h(4), aj::a64::ptr(a2));
            ctx->cc.ld1(d.h(6), aj::a64::ptr(a3));
        }
    } else {
        // AArch64 `ldr b` zeroes bits [8..127] of `d`, so loading directly into `b(4/8/12)`
        // yields zero-extended i32 lanes with zero `movi` and zero `ushll`.
        ctx->cc.ldr(d.b(), m0);
        ctx->cc.ld1(d.b(4), aj::a64::ptr(a1));
        ctx->cc.ld1(d.b(8), aj::a64::ptr(a2));
        ctx->cc.ld1(d.b(12), aj::a64::ptr(a3));
    }
}
#endif

// Register gather of zero-extended u8/u16/u32 lanes (AVX2/AVX-512 extracts, NEON ld1), else via the stack.
static void emit_gather(zj_ctx* ctx, zj_reg_t dst, zj_reg_t base, zj_reg_t byte_offsets_i32, uint32_t elem_bytes) noexcept {
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes >= 32u) {
        emit_x86_reg_gather_impl(ctx, dst, base, byte_offsets_i32, elem_bytes, false);
        return;
    }
#elif defined(ASMJIT_UJIT_AARCH64)
    emit_a64_reg_gather_ld1(ctx, dst, base, byte_offsets_i32, elem_bytes, false);
    return;
#endif
    emit_scalar_gather_impl(ctx, dst, base, byte_offsets_i32, elem_bytes);
}

int32_t zj_compile_lut1d_u8_neon(
    zj_runtime* rt,
    const uint8_t* table_256,
    uint32_t clip_idx,
    void** out_fn,
    size_t* out_size,
    uint8_t* out_execmem_denied
) {
    if (out_fn != nullptr) *out_fn = nullptr;
    if (out_size != nullptr) *out_size = 0u;
    if (out_execmem_denied != nullptr) *out_execmem_denied = 0u;
#if defined(ASMJIT_UJIT_AARCH64)
    if (rt == nullptr || table_256 == nullptr) return -1;

    aj::CodeHolder code;
    if (code.init(rt->rt.environment(), rt->rt.cpu_features()) != aj::Error::kOk) return -1;
    ShimErrorHandler err_handler;
    code.set_error_handler(&err_handler);

    aj::a64::Assembler a(&code);
    using namespace aj::a64;

    const Vec q0 = Vec::make_q(0);
    const Vec q1 = Vec::make_q(1);
    const Vec q2 = Vec::make_q(2);
    const Vec q3 = Vec::make_q(3);
    const Vec q4 = Vec::make_q(4);

    // Args: x0=dst, x1=dst_stride, x2=srcs, x3=src_strides, x4=width, x5=height.
    // Uses only AAPCS64 caller-saved x0..x13, v0..v4, v16..v31.
    aj::Label l_done = a.new_label();
    a.cbz(x4, l_done);
    a.cbz(x5, l_done);

    const uint64_t clip_off = uint64_t(clip_idx) * 8u;
    if (clip_off <= 32760u) {
        a.ldr(x6, ptr(x2, int32_t(clip_off)));
        a.ldr(x7, ptr(x3, int32_t(clip_off)));
    } else {
        a.mov(x9, clip_off);
        a.ldr(x6, ptr(x2, x9));
        a.ldr(x7, ptr(x3, x9));
    }

    aj::Label l_table = a.new_label();
    a.adr(x9, l_table);
    for (uint32_t i = 0; i < 8u; ++i) {
        a.ldp(Vec::make_q(16u + i * 2u), Vec::make_q(17u + i * 2u), ptr(x9, int32_t(i * 32u)));
    }
    a.movi(q4.b16(), 64);

    // Preserve x12 (iters32 = width / 32) and x13 (rem = width & 31) across all rows.
    a.lsr(x12, x4, 5);
    a.and_(x13, x4, 31);
    a.mov(x8, x5);

    // Lookup in table group g (v[16 + 4g] .. v[19 + 4g]); the caller subtracts 64 from in_v between
    // groups. tbl zeroes out-of-range lanes, tbx keeps the previous groups' results.
    auto lookup = [&](uint32_t g, const Vec& out_v, const Vec& in_v) {
        const uint32_t t = 16u + g * 4u;
        const Vec t0 = Vec::make_q(t), t1 = Vec::make_q(t + 1u), t2 = Vec::make_q(t + 2u), t3 = Vec::make_q(t + 3u);
        if (g == 0u) {
            a.tbl(out_v.b16(), t0.b16(), t1.b16(), t2.b16(), t3.b16(), in_v.b16());
        } else {
            a.tbx(out_v.b16(), t0.b16(), t1.b16(), t2.b16(), t3.b16(), in_v.b16());
        }
    };
    auto emit_16 = [&](const Vec& out_v, const Vec& in_v) {
        for (uint32_t g = 0; g < 4u; ++g) {
            if (g != 0u) a.sub(in_v.b16(), in_v.b16(), q4.b16());
            lookup(g, out_v, in_v);
        }
    };

    aj::Label l_y = a.new_label();
    aj::Label l_tail = a.new_label();
    aj::Label l_next_y = a.new_label();
    a.bind(l_y);
    a.mov(x9, x0);
    a.mov(x10, x6);

    // 2x-unrolled 32B/iter loop using ldp/stp and interleaved 4-register tbl/tbx over v16..v31.
    a.cbz(x12, l_tail);
    a.mov(x11, x12);
    aj::Label l_x32 = a.new_label();
    a.bind(l_x32);
    a.ldp(q0, q1, ptr_post(x10, 32));
    for (uint32_t g = 0; g < 4u; ++g) {
        if (g != 0u) {
            a.sub(q0.b16(), q0.b16(), q4.b16());
            a.sub(q1.b16(), q1.b16(), q4.b16());
        }
        lookup(g, q2, q0);
        lookup(g, q3, q1);
    }
    a.stp(q2, q3, ptr_post(x9, 32));
    a.subs(x11, x11, 1);
    a.b_ne(l_x32);

    a.bind(l_tail);
    a.cbz(x13, l_next_y);
    a.ldr(q0, ptr_post(x10, 16));
    emit_16(q2, q0);
    a.str(q2, ptr_post(x9, 16));
    a.cmp(x13, 16);
    a.b_ls(l_next_y);
    a.ldr(q0, ptr(x10));
    emit_16(q2, q0);
    a.str(q2, ptr(x9));

    a.bind(l_next_y);
    a.add(x0, x0, x1);
    a.add(x6, x6, x7);
    a.subs(x8, x8, 1);
    a.b_ne(l_y);

    a.bind(l_done);
    a.ret(x30);

    a.align(aj::AlignMode::kData, 16);
    a.bind(l_table);
    a.embed(table_256, 256u);

    if (err_handler.first_error != aj::Error::kOk) return -1;
    return add_to_runtime(rt, code, out_fn, out_size, out_execmem_denied) == aj::Error::kOk ? 0 : -1;
#else
    (void)rt; (void)table_256; (void)clip_idx;
    return -1;
#endif
}

void zj_vec_gather_f32(zj_ctx* ctx, zj_reg_t dst, zj_reg_t base, zj_reg_t byte_offsets_i32, uint8_t can_use_hw_gather) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    const bool use_hw = (can_use_hw_gather != 0u) && (ctx->cpu.prefer_hw_gather_f32 != 0u);
    if (use_hw && ctx->cpu.vec_bytes == 64u) {
        aj::x86::KReg k = ctx->cc.new_kw();
        ctx->cc.kxnorw(k, k, k);
        ujit::Vec d = unpack_vec(dst);
        ujit::Vec off = unpack_vec(byte_offsets_i32);
        // L2: EVEX vpgatherdd forbids dst aliasing index register; use a fresh tmp if needed.
        if (d.id() == off.id()) {
            ujit::Vec tmp = ctx->uc.new_vec();
            ctx->cc.k(k).vpgatherdd(tmp.zmm(), aj::x86::ptr(unpack_gp(base), off.zmm(), 0));
            ctx->uc.emit_2v(ujit::UniOpVV::kMov, d, tmp);
        } else {
            ctx->cc.k(k).vpgatherdd(d.zmm(), aj::x86::ptr(unpack_gp(base), off.zmm(), 0));
        }
        return;
    }
    if (use_hw && ctx->cpu.vec_bytes == 32u) {
        ujit::Vec mask = ctx->uc.new_vec();
        ctx->cc.vpcmpeqd(mask.ymm(), mask.ymm(), mask.ymm());
        ujit::Vec d = unpack_vec(dst);
        ujit::Vec off = unpack_vec(byte_offsets_i32);
        // vpgatherdd forbids dst aliasing index or mask; use a fresh tmp if needed.
        if (d.id() == off.id()) {
            ujit::Vec tmp = ctx->uc.new_vec();
            ctx->cc.vpgatherdd(tmp.ymm(), aj::x86::ptr(unpack_gp(base), off.ymm(), 0), mask.ymm());
            ctx->uc.emit_2v(ujit::UniOpVV::kMov, d, tmp);
        } else {
            ctx->cc.vpgatherdd(d.ymm(), aj::x86::ptr(unpack_gp(base), off.ymm(), 0), mask.ymm());
        }
        return;
    }
#else
    (void)can_use_hw_gather;
#endif
    emit_gather(ctx, dst, base, byte_offsets_i32, 4u);
}

void zj_vec_gather_f16(zj_ctx* ctx, zj_reg_t dst_f32, zj_reg_t base_reg, zj_reg_t byte_offsets_i32) {
    if (!ctx_ok(ctx)) return;
#if defined(ASMJIT_UJIT_X86)
    if (ctx->cpu.vec_bytes >= 32u && ctx->cpu.has_f16c) {
        emit_x86_reg_gather_impl(ctx, dst_f32, base_reg, byte_offsets_i32, 2u, true);
        return;
    }
#elif defined(ASMJIT_UJIT_AARCH64)
    emit_a64_reg_gather_ld1(ctx, dst_f32, base_reg, byte_offsets_i32, 2u, true);
    return;
#endif
    set_shim_error(ctx, aj::Error::kFeatureNotEnabled, "f16 gather requires F16C");
}

void zj_vec_gather_u16(zj_ctx* ctx, zj_reg_t dst_i32, zj_reg_t base, zj_reg_t byte_offsets_i32) {
    if (ctx_ok(ctx)) emit_gather(ctx, dst_i32, base, byte_offsets_i32, 2u);
}

void zj_vec_gather_u8(zj_ctx* ctx, zj_reg_t dst_i32, zj_reg_t base, zj_reg_t byte_offsets_i32) {
    if (ctx_ok(ctx)) emit_gather(ctx, dst_i32, base, byte_offsets_i32, 1u);
}

} // extern "C"
