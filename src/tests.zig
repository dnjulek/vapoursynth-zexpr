//! Host-free unit test root for `zig build test` (no VapourSynth dependency).

comptime {
    _ = @import("jit/cpu.zig");
    _ = @import("jit/asmjit.zig");
    _ = @import("jit/pixel.zig");
    _ = @import("jit/math.zig");
    _ = @import("jit/codegen.zig");
    _ = @import("jit/lut.zig");
    _ = @import("jit/int_path.zig");
    _ = @import("expr/token.zig");
    _ = @import("expr/lexer.zig");
    _ = @import("expr/sortnet.zig");
    _ = @import("expr/parser.zig");
    _ = @import("expr/interp.zig");
    _ = @import("expr/lower.zig");
    _ = @import("expr/optimize.zig");
    _ = @import("expr/analysis.zig");
}
