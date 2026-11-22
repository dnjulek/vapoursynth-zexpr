// Native C++ ABI probe compiled with the exact same header, defines, and flags
// as zasmjit.cpp. Tests compare these sizeof/alignof/offsetof values against
// Zig's translated @import("zasmjit_c") types.

#include <cstddef>
#include <cstdint>

#include "zasmjit.h"

extern "C" {

const size_t* zj_probe_reg_layout(size_t* count) {
    static const size_t layout[] = {
        sizeof(zj_reg_t),
        alignof(zj_reg_t),
        offsetof(zj_reg_t, id),
        offsetof(zj_reg_t, sig),
    };
    *count = sizeof(layout) / sizeof(layout[0]);
    return layout;
}

const size_t* zj_probe_label_layout(size_t* count) {
    static const size_t layout[] = {
        sizeof(zj_label_t),
        alignof(zj_label_t),
        offsetof(zj_label_t, id),
    };
    *count = sizeof(layout) / sizeof(layout[0]);
    return layout;
}

const size_t* zj_probe_mem_layout(size_t* count) {
    static const size_t layout[] = {
        sizeof(zj_mem_t),
        alignof(zj_mem_t),
        offsetof(zj_mem_t, base),
        offsetof(zj_mem_t, index),
        offsetof(zj_mem_t, disp),
        offsetof(zj_mem_t, shift),
        offsetof(zj_mem_t, has_index),
        offsetof(zj_mem_t, is_label),
        offsetof(zj_mem_t, size),
        offsetof(zj_mem_t, label_id),
    };
    *count = sizeof(layout) / sizeof(layout[0]);
    return layout;
}

const size_t* zj_probe_cpu_info_layout(size_t* count) {
    static const size_t layout[] = {
        sizeof(zj_cpu_info_t),
        alignof(zj_cpu_info_t),
        offsetof(zj_cpu_info_t, isa),
        offsetof(zj_cpu_info_t, vec_bytes),
        offsetof(zj_cpu_info_t, f32_lanes),
        offsetof(zj_cpu_info_t, has_ssse3),
        offsetof(zj_cpu_info_t, has_sse4_1),
        offsetof(zj_cpu_info_t, has_fma),
        offsetof(zj_cpu_info_t, has_f16c),
        offsetof(zj_cpu_info_t, has_avx2),
        offsetof(zj_cpu_info_t, has_avx512),
        offsetof(zj_cpu_info_t, prefer_hw_gather_f32),
    };
    *count = sizeof(layout) / sizeof(layout[0]);
    return layout;
}

zj_reg_t zj_probe_echo_reg(zj_reg_t r) { return r; }
zj_mem_t zj_probe_echo_mem(zj_mem_t m) { return m; }

} // extern "C"
