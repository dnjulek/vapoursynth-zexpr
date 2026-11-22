//! Host CPU ISA detection and `ZEXPR_CPU_LEVEL` per-architecture override parsing.

const std = @import("std");
const builtin = @import("builtin");
const c = @import("zasmjit_c");

pub const Isa = enum(u32) {
    scalar = c.ZJ_ISA_SCALAR,
    sse2 = c.ZJ_ISA_SSE2,
    v2 = c.ZJ_ISA_V2,
    v3 = c.ZJ_ISA_V3,
    v4 = c.ZJ_ISA_V4,
    neon = c.ZJ_ISA_NEON,
};

pub const IsaFamily = enum {
    x86,
    aarch64,

    pub fn host() IsaFamily {
        return switch (builtin.cpu.arch) {
            .x86_64, .x86 => .x86,
            .aarch64, .aarch64_be => .aarch64,
            else => @compileError("unsupported CPU architecture for zexpr"),
        };
    }
};

pub const CpuInfo = struct {
    isa: Isa,
    vec_bytes: u32,
    f32_lanes: u32,
    has_ssse3: bool,
    has_sse4_1: bool,
    has_fma: bool,
    has_f16c: bool,
    has_avx2: bool,
    has_avx512: bool,
    prefer_hw_gather_f32: bool,

    pub fn toC(self: CpuInfo) c.zj_cpu_info_t {
        return .{
            .isa = @backingInt(self.isa),
            .vec_bytes = self.vec_bytes,
            .f32_lanes = self.f32_lanes,
            .has_ssse3 = @intFromBool(self.has_ssse3),
            .has_sse4_1 = @intFromBool(self.has_sse4_1),
            .has_fma = @intFromBool(self.has_fma),
            .has_f16c = @intFromBool(self.has_f16c),
            .has_avx2 = @intFromBool(self.has_avx2),
            .has_avx512 = @intFromBool(self.has_avx512),
            .prefer_hw_gather_f32 = @intFromBool(self.prefer_hw_gather_f32),
            ._pad = .{0},
        };
    }

    pub fn fromC(raw: c.zj_cpu_info_t) CpuInfo {
        return .{
            .isa = std.enums.fromInt(Isa, raw.isa) orelse .scalar,
            .vec_bytes = raw.vec_bytes,
            .f32_lanes = raw.f32_lanes,
            .has_ssse3 = raw.has_ssse3 != 0,
            .has_sse4_1 = raw.has_sse4_1 != 0,
            .has_fma = raw.has_fma != 0,
            .has_f16c = raw.has_f16c != 0,
            .has_avx2 = raw.has_avx2 != 0,
            .has_avx512 = raw.has_avx512 != 0,
            .prefer_hw_gather_f32 = raw.prefer_hw_gather_f32 != 0,
        };
    }

    pub fn isaName(self: CpuInfo) []const u8 {
        return switch (self.isa) {
            .scalar => "scalar",
            .sse2 => "sse2",
            .v2 => "x86-64-v2",
            .v3 => "x86-64-v3",
            .v4 => "x86-64-v4",
            .neon => "neon",
        };
    }
};

/// Parse `ZEXPR_CPU_LEVEL` from the process environment.
/// Per-arch contract:
/// - x86-64: `0` = scalar, `1` = SSE2, `2` = v2, `3` = v3, `4` = v4.
/// - AArch64: `0` = scalar, `1` (or unset) = NEON.
/// Unset or unparseable values return `null` (use host default).
pub fn readEnvOverride() ?i32 {
    const raw = std.c.getenv("ZEXPR_CPU_LEVEL") orelse return null;
    const text = std.mem.span(raw);
    return parseEnvOverride(text, IsaFamily.host());
}

pub fn parseEnvOverride(text: []const u8, family: IsaFamily) ?i32 {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (trimmed.len == 0) return null;
    const val = std.fmt.parseInt(i32, trimmed, 10) catch return null;
    if (val < 0) return null;
    return switch (family) {
        .x86 => if (val > 4) 4 else val,
        .aarch64 => if (val == 0) 0 else 1,
    };
}

/// Parse optional `ZEXPR_HW_GATHER` override (`0` = scalar/register gather, `1` = hardware `vpgatherdd`, P1).
pub fn parseHwGatherOverride(text: []const u8) ?bool {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (std.mem.eql(u8, trimmed, "0")) return false;
    if (std.mem.eql(u8, trimmed, "1")) return true;
    return null;
}

pub fn readHwGatherOverride() ?bool {
    const raw = std.c.getenv("ZEXPR_HW_GATHER") orelse return null;
    return parseHwGatherOverride(std.mem.span(raw));
}

pub fn detectWithOverride(override: ?i32) CpuInfo {
    var raw: c.zj_cpu_info_t = undefined;
    c.zj_cpu_detect(&raw, override orelse -1);
    var info = CpuInfo.fromC(raw);
    if (info.has_avx2 or info.has_avx512) {
        if (readHwGatherOverride()) |hw| {
            info.prefer_hw_gather_f32 = hw;
        }
    }
    return info;
}

pub fn detect() CpuInfo {
    return detectWithOverride(readEnvOverride());
}

test "parseEnvOverride handles per-arch levels" {
    try std.testing.expectEqual(@as(?bool, null), parseHwGatherOverride(""));
    try std.testing.expectEqual(@as(?bool, false), parseHwGatherOverride(" 0 "));
    try std.testing.expectEqual(@as(?bool, true), parseHwGatherOverride("1"));
    try std.testing.expectEqual(@as(?bool, null), parseHwGatherOverride("2"));
    try std.testing.expectEqual(@as(?i32, null), parseEnvOverride("", .x86));
    try std.testing.expectEqual(@as(?i32, null), parseEnvOverride("bogus", .x86));
    try std.testing.expectEqual(@as(?i32, null), parseEnvOverride("-1", .x86));
    try std.testing.expectEqual(@as(?i32, 0), parseEnvOverride("0", .x86));
    try std.testing.expectEqual(@as(?i32, 1), parseEnvOverride("1", .x86));
    try std.testing.expectEqual(@as(?i32, 3), parseEnvOverride(" 3 ", .x86));
    try std.testing.expectEqual(@as(?i32, 4), parseEnvOverride("9", .x86));
    try std.testing.expectEqual(@as(?i32, 0), parseEnvOverride("0", .aarch64));
    try std.testing.expectEqual(@as(?i32, 1), parseEnvOverride("1", .aarch64));
    try std.testing.expectEqual(@as(?i32, 1), parseEnvOverride("3", .aarch64));
}

test "detectWithOverride scalar and host defaults" {
    const scalar = detectWithOverride(0);
    try std.testing.expectEqual(Isa.scalar, scalar.isa);
    try std.testing.expectEqual(0, scalar.vec_bytes);
    try std.testing.expectEqual(0, scalar.f32_lanes);

    const host = detectWithOverride(null);
    switch (IsaFamily.host()) {
        .x86 => {
            try std.testing.expect(host.isa != .scalar);
            try std.testing.expect(host.isa != .neon);
            try std.testing.expect(host.vec_bytes == 16 or host.vec_bytes == 32 or host.vec_bytes == 64);
            try std.testing.expectEqual(host.vec_bytes / 4, host.f32_lanes);
        },
        .aarch64 => {
            try std.testing.expectEqual(Isa.neon, host.isa);
            try std.testing.expectEqual(16, host.vec_bytes);
            try std.testing.expectEqual(4, host.f32_lanes);
            try std.testing.expect(host.has_fma);
        },
    }
}
