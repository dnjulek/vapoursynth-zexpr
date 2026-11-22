//! Zero-copy single-pass whitespace lexer for RPN expression strings.

const std = @import("std");
const token = @import("token.zig");

pub const Lexer = struct {
    source: []const u8,
    pos: usize = 0,

    pub fn init(source: []const u8) Lexer {
        return .{ .source = source };
    }

    pub fn next(self: *Lexer) token.TokenError!?token.Token {
        while (self.pos < self.source.len and isWhitespace(self.source[self.pos])) {
            self.pos += 1;
        }
        if (self.pos >= self.source.len) return null;

        const start = self.pos;
        while (self.pos < self.source.len and !isWhitespace(self.source[self.pos])) {
            self.pos += 1;
        }
        const slice = self.source[start..self.pos];
        return .{
            .kind = try token.classify(slice),
            .text = slice,
            .offset = start,
        };
    }

    fn isWhitespace(ch: u8) bool {
        return ch == ' ' or ch == '\t' or ch == '\n' or ch == '\r' or ch == '\x0b' or ch == '\x0c';
    }
};

test "lexer splits tokens and tracks offsets" {
    var lex = Lexer.init("  x  023\t+\n y[-1,0]:m * ");
    const t0 = (try lex.next()).?;
    try std.testing.expectEqual(@as(usize, 2), t0.offset);
    try std.testing.expectEqual(token.TokenKind{ .clip_load = 0 }, t0.kind);

    const t1 = (try lex.next()).?;
    try std.testing.expectEqual(@as(?i32, 23), t1.kind.literal.i32_val);

    const t2 = (try lex.next()).?;
    try std.testing.expectEqual(token.TokenKind{ .binary = .add }, t2.kind);

    const t3 = (try lex.next()).?;
    try std.testing.expectEqual(token.TokenKind{ .clip_rel = .{ .clip = 1, .dx = -1, .dy = 0, .mode = .mirror } }, t3.kind);

    const t4 = (try lex.next()).?;
    try std.testing.expectEqual(token.TokenKind{ .binary = .mul }, t4.kind);

    try std.testing.expectEqual(@as(?token.Token, null), try lex.next());
}
