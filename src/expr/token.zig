//! Token definitions and D1/Y15-compliant numeric literal & identifier parsers.

const std = @import("std");
const ast = @import("ast.zig");
const builtins = @import("builtins.zig");

pub const Literal = struct {
    /// Exact float32 value (used when `opt=0` or when literal is non-integer).
    f32_val: f32,
    /// Exact signed 32-bit integer value when literal is an integer fitting in i32
    /// (including hex literals in `[0x80000000, 0xFFFFFFFF]` wrapped as signed `i32` per Y15).
    i32_val: ?i32,
};

pub const TokenKind = union(enum) {
    literal: Literal,
    unary: ast.UnaryOp,
    binary: ast.BinaryOp,
    ternary_select,
    clip_op,
    lerp_op,
    hypot_op,
    linstep_op,
    smoothstep_op,
    difflimit_op,
    absdiff_op,
    minabs_op,
    maxabs_op,
    polyval: u32,
    clip_load: ast.ClipId,
    clip_rel: struct {
        clip: ast.ClipId,
        dx: i32,
        dy: i32,
        mode: ?ast.BoundaryMode,
    },
    clip_abs: struct {
        clip: ast.ClipId,
        mode: ?ast.BoundaryMode,
    },
    prop_load: struct {
        clip: ast.ClipId,
        name: []const u8,
    },
    var_store: []const u8,
    var_load: []const u8,
    dup: u32,
    swap: u32,
    drop: u32,
    sort: u32,
    argmin: u32,
    argmax: u32,
    argsort: u32,
    coord_x,
    coord_y,
    frame_n,
    plane_width,
    plane_height,
    pi,
    builtin_const: builtins.BuiltinSpec,
};

pub const Token = struct {
    kind: TokenKind,
    text: []const u8,
    offset: usize,
};

pub const TokenError = error{
    InvalidToken,
    InvalidNumber,
    InvalidClipRef,
    InvalidRelAccess,
    InvalidVarName,
    InvalidPropName,
    InvalidStackOp,
};

/// Classify a single whitespace-delimited lexeme into a `TokenKind`.
pub fn classify(text: []const u8) TokenError!TokenKind {
    if (text.len == 0) return error.InvalidToken;

    // 1. Numeric literals (D1 + Y15): starts with digit, '.', or sign followed by digit/'.'.
    if (isNumericStart(text)) {
        return .{ .literal = try parseNumber(text) };
    }

    // 2. Fixed operators & keywords.
    if (matchKeyword(text)) |kw| return kw;

    // 3. Variable store / load: `name!` / `name@`.
    const last = text[text.len - 1];
    if (last == '!' or last == '@') {
        const name = text[0 .. text.len - 1];
        if (!isValidVarName(name)) return error.InvalidVarName;
        return if (last == '!') .{ .var_store = name } else .{ .var_load = name };
    }

    // 4. Stack ops with numeric suffix: dupN, swapN, dropN, sortN, argminN, argmaxN, argsortN.
    if (try parseStackOp(text)) |op| return op;

    // 5. Clip property access: `<clip>.<PropName>`.
    if (std.mem.findScalar(u8, text, '.')) |dot| {
        const clip_part = text[0..dot];
        const prop_part = text[dot + 1 ..];
        const clip = parseClipPrefix(clip_part) orelse return error.InvalidPropName;
        if (!isValidPropName(prop_part)) return error.InvalidPropName;
        return .{ .prop_load = .{ .clip = clip, .name = prop_part } };
    }

    // 6. Clip spatial access: `<clip>[dx,dy][:c|:m]` or `<clip>[][:c|:m]`.
    if (std.mem.findScalar(u8, text, '[')) |lbrack| {
        return try parseBracketAccess(text, lbrack);
    }

    // 7. Scoped built-in constant: `<clip>:<const>` or `dst:<const>` (checked after `[` so `:c`/`:m` stay spatial).
    if (std.mem.findScalar(u8, text, ':')) |colon| {
        const left = text[0..colon];
        const right = text[colon + 1 ..];
        const target: builtins.BuiltinTarget = if (std.mem.eql(u8, left, "dst"))
            .dst
        else if (parseClipPrefix(left)) |clip|
            .{ .clip = clip }
        else
            return error.InvalidToken;
        const kind = builtins.matchKind(right) orelse return error.InvalidToken;
        return .{ .builtin_const = .{ .target = target, .kind = kind } };
    }

    // 8. Plain clip reference: `x`, `y`, `z`, `a`..`w`, `srcN`.
    if (parseClipPrefix(text)) |clip| {
        return .{ .clip_load = clip };
    }

    return error.InvalidToken;
}

fn isNumericStart(s: []const u8) bool {
    const c0 = s[0];
    if (std.ascii.isDigit(c0)) return true;
    if (c0 == '.') return s.len >= 2 and std.ascii.isDigit(s[1]);
    if (c0 == '-') {
        if (s.len < 2) return false;
        const c1 = s[1];
        if (std.ascii.isDigit(c1)) return true;
        if (c1 == '.') return s.len >= 3 and std.ascii.isDigit(s[2]);
    }
    return false;
}

/// D1 + Y15 numeric literal parser:
/// - Leading-zero decimals are base-10 (`023` = 23, `010` = 10), never octal.
/// - Hex integers (`0x...`) wrap `[0x80000000, 0xFFFFFFFF]` to signed `i32` (`0xFFFFFFFF` -> `-1`).
/// - Hex floats (`0x1.8p2`) follow C99 hex-float rules.
/// - Underscore separators, leading `+`, signed hex (`-0x...`), and trailing garbage are rejected.
pub fn parseNumber(text: []const u8) TokenError!Literal {
    if (text.len == 0) return error.InvalidNumber;
    if (std.mem.findScalar(u8, text, '_') != null) return error.InvalidNumber;

    var s = text;
    var negative = false;
    if (s[0] == '-') {
        negative = true;
        s = s[1..];
    } else if (s[0] == '+') {
        return error.InvalidNumber;
    }
    if (s.len == 0) return error.InvalidNumber;

    // Hex integer or hex float (`0x...` / `0X...`). Reject leading sign before `0x`.
    if (s.len >= 2 and s[0] == '0' and (s[1] == 'x' or s[1] == 'X')) {
        if (negative) return error.InvalidNumber;
        const has_hex_float_marker = std.mem.findAny(u8, s[2..], ".pP") != null;
        if (has_hex_float_marker) {
            if (std.mem.findAny(u8, s[2..], "pP") == null) return error.InvalidNumber;
            const f = std.fmt.parseFloat(f32, text) catch return error.InvalidNumber;
            return .{ .f32_val = f, .i32_val = null };
        }
        const digits = s[2..];
        if (digits.len == 0) return error.InvalidNumber;
        for (digits) |ch| {
            if (!std.ascii.isHex(ch)) return error.InvalidNumber;
        }
        const mag = std.fmt.parseInt(u64, digits, 16) catch return error.InvalidNumber;
        if (mag <= std.math.maxInt(u32)) {
            const raw_u32: u32 = @intCast(mag);
            const signed_i32: i32 = @bitCast(raw_u32);
            return .{ .f32_val = @floatFromInt(signed_i32), .i32_val = signed_i32 };
        }
        // Larger hex integer outside 32-bit range: represent as f32 (matches akarin fallback).
        const f_mag: f32 = @floatFromInt(mag);
        return .{ .f32_val = f_mag, .i32_val = null };
    }

    // Check if pure decimal integer (all ASCII digits, no '.', 'e', 'E').
    var all_digits = true;
    for (s) |ch| {
        if (!std.ascii.isDigit(ch)) {
            all_digits = false;
            break;
        }
    }
    if (all_digits) {
        // Explicit base 10 so leading zeroes are decimal (D1: `023` == 23).
        if (std.fmt.parseInt(i64, text, 10)) |val64| {
            if (val64 >= std.math.minInt(i32) and val64 <= std.math.maxInt(i32)) {
                const i32_v: i32 = @intCast(val64);
                // Preserve `-0.0` if user wrote `-0`!
                const f32_v: f32 = if (negative and i32_v == 0) -0.0 else @floatFromInt(i32_v);
                return .{ .f32_val = f32_v, .i32_val = i32_v };
            }
            const f32_v: f32 = @floatFromInt(val64);
            return .{ .f32_val = f32_v, .i32_val = null };
        } else |_| {
            // Overflowed i64 (e.g. huge integer literal): fall back to f32 parse.
            const f32_v = std.fmt.parseFloat(f32, text) catch return error.InvalidNumber;
            return .{ .f32_val = f32_v, .i32_val = null };
        }
    }

    // Decimal floating-point literal; reject hex/inf/nan tricks or trailing junk.
    const f32_v = std.fmt.parseFloat(f32, text) catch return error.InvalidNumber;
    return .{ .f32_val = f32_v, .i32_val = null };
}

fn matchKeyword(s: []const u8) ?TokenKind {
    // Single-character fast path
    if (s.len == 1) {
        return switch (s[0]) {
            '+' => .{ .binary = .add },
            '-' => .{ .binary = .sub },
            '*' => .{ .binary = .mul },
            '/' => .{ .binary = .div },
            '%' => .{ .binary = .mod },
            '<' => .{ .binary = .lt },
            '>' => .{ .binary = .gt },
            '=' => .{ .binary = .eq },
            '?' => .ternary_select,
            'N' => .frame_n,
            'X' => .coord_x,
            'Y' => .coord_y,
            else => null,
        };
    }

    const map = std.StaticStringMap(TokenKind).initComptime(.{
        .{ "<=", TokenKind{ .binary = .le } },
        .{ ">=", TokenKind{ .binary = .ge } },
        .{ "**", TokenKind{ .binary = .pow } },
        .{ "pow", TokenKind{ .binary = .pow } },
        .{ "min", TokenKind{ .binary = .min } },
        .{ "max", TokenKind{ .binary = .max } },
        .{ "atan2", TokenKind{ .binary = .atan2 } },
        .{ "copysign", TokenKind{ .binary = .copysign } },
        .{ "and", TokenKind{ .binary = .and_op } },
        .{ "or", TokenKind{ .binary = .or_op } },
        .{ "xor", TokenKind{ .binary = .xor_op } },
        .{ "not", TokenKind{ .unary = .not } },
        .{ "bitand", TokenKind{ .binary = .bitand } },
        .{ "bitor", TokenKind{ .binary = .bitor } },
        .{ "bitxor", TokenKind{ .binary = .bitxor } },
        .{ "bitnot", TokenKind{ .unary = .bitnot } },
        .{ "abs", TokenKind{ .unary = .abs } },
        .{ "neg", TokenKind{ .unary = .neg } },
        .{ "sqrt", TokenKind{ .unary = .sqrt } },
        .{ "rsqrt", TokenKind{ .unary = .rsqrt } },
        .{ "sgn", TokenKind{ .unary = .sgn } },
        .{ "trunc", TokenKind{ .unary = .trunc } },
        .{ "floor", TokenKind{ .unary = .floor } },
        .{ "ceil", TokenKind{ .unary = .ceil } },
        .{ "round", TokenKind{ .unary = .round } },
        .{ "exp", TokenKind{ .unary = .exp } },
        .{ "exp2", TokenKind{ .unary = .exp2 } },
        .{ "log", TokenKind{ .unary = .log } },
        .{ "log2", TokenKind{ .unary = .log2 } },
        .{ "sin", TokenKind{ .unary = .sin } },
        .{ "cos", TokenKind{ .unary = .cos } },
        .{ "tan", TokenKind{ .unary = .tan } },
        .{ "atan", TokenKind{ .unary = .atan } },
        .{ "asin", TokenKind{ .unary = .asin } },
        .{ "acos", TokenKind{ .unary = .acos } },
        .{ "clip", TokenKind.clip_op },
        .{ "clamp", TokenKind.clip_op },
        .{ "lerp", TokenKind.lerp_op },
        .{ "hypot", TokenKind.hypot_op },
        .{ "linstep", TokenKind.linstep_op },
        .{ "smoothstep", TokenKind.smoothstep_op },
        .{ "difflimit", TokenKind.difflimit_op },
        .{ "absdiff", TokenKind.absdiff_op },
        .{ "minabs", TokenKind.minabs_op },
        .{ "maxabs", TokenKind.maxabs_op },
        .{ "dup", TokenKind{ .dup = 0 } },
        .{ "swap", TokenKind{ .swap = 1 } },
        .{ "drop", TokenKind{ .drop = 1 } },
        .{ "pi", TokenKind.pi },
        .{ "width", TokenKind.plane_width },
        .{ "height", TokenKind.plane_height },
        .{ "range_min", TokenKind{ .builtin_const = .{ .target = .{ .clip = 0 }, .kind = .range_min } } },
        .{ "range_max", TokenKind{ .builtin_const = .{ .target = .{ .clip = 0 }, .kind = .range_max } } },
        .{ "range_size", TokenKind{ .builtin_const = .{ .target = .{ .clip = 0 }, .kind = .range_size } } },
        .{ "mask_max", TokenKind{ .builtin_const = .{ .target = .{ .clip = 0 }, .kind = .mask_max } } },
        .{ "neutral", TokenKind{ .builtin_const = .{ .target = .{ .clip = 0 }, .kind = .neutral } } },
        .{ "plane_min", TokenKind{ .builtin_const = .{ .target = .{ .clip = 0 }, .kind = .plane_min } } },
        .{ "plane_max", TokenKind{ .builtin_const = .{ .target = .{ .clip = 0 }, .kind = .plane_max } } },
    });
    return map.get(s);
}

fn parseStackOp(s: []const u8) TokenError!?TokenKind {
    if (std.mem.startsWith(u8, s, "dup") and s.len > 3) {
        if (parseExactUint(s[3..])) |n| return .{ .dup = n };
        return null;
    }
    if (std.mem.startsWith(u8, s, "swap") and s.len > 4) {
        if (parseExactUint(s[4..])) |n| {
            return .{ .swap = n };
        }
        return null;
    }
    if (std.mem.startsWith(u8, s, "drop") and s.len > 4) {
        if (parseExactUint(s[4..])) |n| return .{ .drop = n };
        return null;
    }
    if (std.mem.startsWith(u8, s, "sort") and s.len > 4) {
        if (parseExactUint(s[4..])) |n| {
            if (n == 0) return error.InvalidStackOp;
            return .{ .sort = n };
        }
        return null;
    }
    if (std.mem.startsWith(u8, s, "argmin") and s.len > 6) {
        if (parseExactUint(s[6..])) |n| {
            if (n == 0) return error.InvalidStackOp;
            return .{ .argmin = n };
        }
        return null;
    }
    if (std.mem.startsWith(u8, s, "argmax") and s.len > 6) {
        if (parseExactUint(s[6..])) |n| {
            if (n == 0) return error.InvalidStackOp;
            return .{ .argmax = n };
        }
        return null;
    }
    if (std.mem.startsWith(u8, s, "argsort") and s.len > 7) {
        if (parseExactUint(s[7..])) |n| {
            if (n == 0) return error.InvalidStackOp;
            return .{ .argsort = n };
        }
        return null;
    }
    if (std.mem.startsWith(u8, s, "polyval") and s.len > 7) {
        if (parseExactUint(s[7..])) |n| return .{ .polyval = n };
        return null;
    }
    if (std.mem.startsWith(u8, s, "polyeval") and s.len > 8) {
        if (parseExactUint(s[8..])) |n| return .{ .polyval = n };
        return null;
    }
    return null;
}

fn parseExactUint(s: []const u8) ?u32 {
    if (s.len == 0) return null;
    for (s) |ch| {
        if (!std.ascii.isDigit(ch)) return null;
    }
    return std.fmt.parseInt(u32, s, 10) catch null;
}

/// Maps `x` -> 0, `y` -> 1, `z` -> 2, `a`..`w` -> 3..25, `srcN` -> N.
pub fn parseClipPrefix(s: []const u8) ?ast.ClipId {
    if (s.len == 1) {
        const ch = s[0];
        if (ch == 'x') return 0;
        if (ch == 'y') return 1;
        if (ch == 'z') return 2;
        if (ch >= 'a' and ch <= 'w') return @intCast(3 + (ch - 'a'));
        return null;
    }
    if (std.mem.startsWith(u8, s, "src") and s.len > 3) {
        const digits = s[3..];
        for (digits) |ch| {
            if (!std.ascii.isDigit(ch)) return null;
        }
        return std.fmt.parseInt(ast.ClipId, digits, 10) catch null;
    }
    return null;
}

fn parseExactRelInt(s: []const u8) ?i32 {
    if (s.len == 0) return null;
    const digits = if (s[0] == '-') s[1..] else s;
    if (digits.len == 0) return null;
    for (digits) |ch| {
        if (!std.ascii.isDigit(ch)) return null;
    }
    return std.fmt.parseInt(i32, s, 10) catch null;
}

fn parseBracketAccess(text: []const u8, lbrack: usize) TokenError!TokenKind {
    const clip = parseClipPrefix(text[0..lbrack]) orelse return error.InvalidClipRef;
    const rbrack = std.mem.findScalarPos(u8, text, lbrack + 1, ']') orelse return error.InvalidRelAccess;

    const suffix = text[rbrack + 1 ..];
    const mode: ?ast.BoundaryMode = if (suffix.len == 0)
        null
    else if (std.mem.eql(u8, suffix, ":c"))
        .clamp
    else if (std.mem.eql(u8, suffix, ":m"))
        .mirror
    else
        return error.InvalidRelAccess;

    const inside = text[lbrack + 1 .. rbrack];
    if (inside.len == 0) {
        return .{ .clip_abs = .{ .clip = clip, .mode = mode } };
    }

    const comma = std.mem.findScalar(u8, inside, ',') orelse return error.InvalidRelAccess;
    const dx_str = inside[0..comma];
    const dy_str = inside[comma + 1 ..];
    if (dx_str.len == 0 or dy_str.len == 0) return error.InvalidRelAccess;
    if (std.mem.findScalar(u8, dy_str, ',') != null) return error.InvalidRelAccess;

    const dx = parseExactRelInt(dx_str) orelse return error.InvalidRelAccess;
    const dy = parseExactRelInt(dy_str) orelse return error.InvalidRelAccess;
    return .{ .clip_rel = .{ .clip = clip, .dx = dx, .dy = dy, .mode = mode } };
}

fn isValidVarName(name: []const u8) bool {
    if (name.len == 0) return false;
    if (!(std.ascii.isAlphabetic(name[0]) or name[0] == '_')) return false;
    for (name[1..]) |ch| {
        if (!(std.ascii.isAlphanumeric(ch) or ch == '_')) return false;
    }
    return true;
}

fn isValidPropName(name: []const u8) bool {
    if (name.len == 0) return false;
    if (!(std.ascii.isAlphabetic(name[0]) or name[0] == '_')) return false;
    for (name[1..]) |ch| {
        if (!(std.ascii.isAlphanumeric(ch) or ch == '_')) return false;
    }
    return true;
}

test "D1 and Y15 numeric literal parsing" {
    // Leading-zero decimals are base-10, not octal.
    const l23 = try parseNumber("023");
    try std.testing.expectEqual(@as(?i32, 23), l23.i32_val);
    try std.testing.expectEqual(@as(f32, 23.0), l23.f32_val);

    const l10 = try parseNumber("010");
    try std.testing.expectEqual(@as(?i32, 10), l10.i32_val);
    try std.testing.expectEqual(@as(f32, 10.0), l10.f32_val);

    // Negative zero preserves sign bit.
    const neg0 = try parseNumber("-0");
    try std.testing.expectEqual(@as(u32, 0x8000_0000), @as(u32, @bitCast(neg0.f32_val)));
    try std.testing.expectEqual(@as(?i32, 0), neg0.i32_val);

    // Hex integers including [0x80000000, 0xFFFFFFFF] signed i32 wrap (Y15).
    const hff = try parseNumber("0xFFFFFFFF");
    try std.testing.expectEqual(@as(?i32, -1), hff.i32_val);
    try std.testing.expectEqual(@as(f32, -1.0), hff.f32_val);

    const hmin = try parseNumber("0x80000000");
    try std.testing.expectEqual(@as(?i32, std.math.minInt(i32)), hmin.i32_val);

    const h1f = try parseNumber("0x1F");
    try std.testing.expectEqual(@as(?i32, 31), h1f.i32_val);

    // Hex floats (C99).
    const hf = try parseNumber("0x1.8p2");
    try std.testing.expectEqual(@as(?i32, null), hf.i32_val);
    try std.testing.expectEqual(@as(f32, 6.0), hf.f32_val);

    // Reject trailing garbage, underscores, leading '+', and signed hex.
    try std.testing.expectError(error.InvalidNumber, parseNumber("123abc"));
    try std.testing.expectError(error.InvalidNumber, parseNumber("0x1.5"));
    try std.testing.expectError(error.InvalidNumber, parseNumber("1_000"));
    try std.testing.expectError(error.InvalidNumber, parseNumber("1_0.5"));
    try std.testing.expectError(error.InvalidNumber, parseNumber("+1"));
    try std.testing.expectError(error.InvalidNumber, parseNumber("-0x10"));
    try std.testing.expectError(error.InvalidToken, classify("+1"));
    try std.testing.expectError(error.InvalidRelAccess, classify("x[+1,0]"));
    try std.testing.expectError(error.InvalidRelAccess, classify("x[1_0,0]"));
    try std.testing.expectError(error.InvalidToken, classify("nan"));
    try std.testing.expectError(error.InvalidToken, classify("inf"));
}

test "clip, spatial, prop, var, and stack token classification" {
    try std.testing.expectEqual(TokenKind{ .clip_load = 0 }, try classify("x"));
    try std.testing.expectEqual(TokenKind{ .clip_load = 1 }, try classify("y"));
    try std.testing.expectEqual(TokenKind{ .clip_load = 2 }, try classify("z"));
    try std.testing.expectEqual(TokenKind{ .clip_load = 3 }, try classify("a"));
    try std.testing.expectEqual(TokenKind{ .clip_load = 25 }, try classify("w"));
    try std.testing.expectEqual(TokenKind{ .clip_load = 42 }, try classify("src42"));

    const rel = try classify("x[-2,3]:m");
    try std.testing.expectEqual(TokenKind{ .clip_rel = .{ .clip = 0, .dx = -2, .dy = 3, .mode = .mirror } }, rel);

    const abs_tok = try classify("src10[]:c");
    try std.testing.expectEqual(TokenKind{ .clip_abs = .{ .clip = 10, .mode = .clamp } }, abs_tok);

    const prop = try classify("y._Matrix");
    try std.testing.expectEqual(@as(ast.ClipId, 1), prop.prop_load.clip);
    try std.testing.expectEqualStrings("_Matrix", prop.prop_load.name);

    const vs = try classify("tmp_1!");
    try std.testing.expectEqualStrings("tmp_1", vs.var_store);
    const vl = try classify("tmp_1@");
    try std.testing.expectEqualStrings("tmp_1", vl.var_load);

    try std.testing.expectEqual(TokenKind{ .dup = 0 }, try classify("dup"));
    try std.testing.expectEqual(TokenKind{ .dup = 3 }, try classify("dup3"));
    try std.testing.expectEqual(TokenKind{ .swap = 1 }, try classify("swap"));
    try std.testing.expectEqual(TokenKind{ .swap = 2 }, try classify("swap2"));
    try std.testing.expectEqual(TokenKind{ .drop = 1 }, try classify("drop"));
    try std.testing.expectEqual(TokenKind{ .drop = 0 }, try classify("drop0"));
    try std.testing.expectEqual(TokenKind{ .sort = 9 }, try classify("sort9"));
    try std.testing.expectEqual(TokenKind{ .swap = 0 }, try classify("swap0"));
    try std.testing.expectError(error.InvalidStackOp, classify("sort0"));

    try std.testing.expectEqual(TokenKind{ .unary = .neg }, try classify("neg"));
    try std.testing.expectEqual(TokenKind{ .unary = .ceil }, try classify("ceil"));
    try std.testing.expectEqual(TokenKind{ .unary = .atan }, try classify("atan"));
    try std.testing.expectEqual(TokenKind{ .unary = .asin }, try classify("asin"));
    try std.testing.expectEqual(TokenKind{ .unary = .acos }, try classify("acos"));
    try std.testing.expectEqual(TokenKind{ .unary = .exp2 }, try classify("exp2"));
    try std.testing.expectEqual(TokenKind{ .unary = .log2 }, try classify("log2"));
    try std.testing.expectEqual(TokenKind{ .unary = .rsqrt }, try classify("rsqrt"));
    try std.testing.expectEqual(TokenKind{ .binary = .copysign }, try classify("copysign"));
    try std.testing.expectEqual(TokenKind.lerp_op, try classify("lerp"));
    try std.testing.expectEqual(TokenKind.hypot_op, try classify("hypot"));
    try std.testing.expectEqual(TokenKind.linstep_op, try classify("linstep"));
    try std.testing.expectEqual(TokenKind.smoothstep_op, try classify("smoothstep"));
    try std.testing.expectEqual(TokenKind.difflimit_op, try classify("difflimit"));
    try std.testing.expectEqual(TokenKind.absdiff_op, try classify("absdiff"));
    try std.testing.expectEqual(TokenKind.minabs_op, try classify("minabs"));
    try std.testing.expectEqual(TokenKind.maxabs_op, try classify("maxabs"));
    try std.testing.expectEqual(TokenKind{ .polyval = 0 }, try classify("polyval0"));
    try std.testing.expectEqual(TokenKind{ .polyval = 3 }, try classify("polyval3"));
    try std.testing.expectEqual(TokenKind{ .polyval = 5 }, try classify("polyeval5"));
    try std.testing.expectError(error.InvalidToken, classify("polyval"));
    try std.testing.expectError(error.InvalidToken, classify("polyvalx"));
    try std.testing.expectError(error.InvalidToken, classify("polyeval-1"));
    try std.testing.expectError(error.InvalidToken, classify("polyval1_0"));

    // Built-in constants (bare, scoped clip, dst) and non-collision with vars/props/spatial.
    try std.testing.expectEqual(TokenKind{ .builtin_const = .{ .target = .{ .clip = 0 }, .kind = .neutral } }, try classify("neutral"));
    try std.testing.expectEqual(TokenKind{ .builtin_const = .{ .target = .{ .clip = 0 }, .kind = .range_min } }, try classify("range_min"));
    try std.testing.expectEqual(TokenKind{ .builtin_const = .{ .target = .{ .clip = 0 }, .kind = .range_max } }, try classify("range_max"));
    try std.testing.expectEqual(TokenKind{ .builtin_const = .{ .target = .{ .clip = 0 }, .kind = .range_size } }, try classify("range_size"));
    try std.testing.expectEqual(TokenKind{ .builtin_const = .{ .target = .{ .clip = 0 }, .kind = .mask_max } }, try classify("mask_max"));
    try std.testing.expectEqual(TokenKind{ .builtin_const = .{ .target = .{ .clip = 0 }, .kind = .plane_min } }, try classify("plane_min"));
    try std.testing.expectEqual(TokenKind{ .builtin_const = .{ .target = .{ .clip = 0 }, .kind = .plane_max } }, try classify("plane_max"));
    try std.testing.expectEqual(TokenKind{ .builtin_const = .{ .target = .{ .clip = 1 }, .kind = .mask_max } }, try classify("y:mask_max"));
    try std.testing.expectEqual(TokenKind{ .builtin_const = .{ .target = .{ .clip = 26 }, .kind = .range_max } }, try classify("src26:range_max"));
    try std.testing.expectEqual(TokenKind{ .builtin_const = .{ .target = .dst, .kind = .plane_max } }, try classify("dst:plane_max"));
    try std.testing.expectEqualStrings("neutral", (try classify("neutral!")).var_store);
    try std.testing.expectEqualStrings("neutral", (try classify("neutral@")).var_load);
    try std.testing.expectEqual(TokenKind{ .clip_load = 7 }, try classify("e"));
    try std.testing.expectError(error.InvalidToken, classify("range_half"));
    try std.testing.expectError(error.InvalidToken, classify("x:range_half"));
    try std.testing.expectError(error.InvalidToken, classify("x:bits"));
    try std.testing.expectError(error.InvalidToken, classify("x:width"));
    try std.testing.expectError(error.InvalidToken, classify("foo:neutral"));
    try std.testing.expectError(error.InvalidToken, classify(":neutral"));
    try std.testing.expectError(error.InvalidToken, classify("dst:"));
    try std.testing.expectError(error.InvalidToken, classify("x:neutral:m"));
}
