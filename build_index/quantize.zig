//! Build-time int16 quantizer for the IVF index.
//!
//! Maps each f32 vector dimension into the int16 range using a fixed integer
//! `SCALE` factor (10_000) and rounds to the nearest integer. The result is
//! padded from 14 to 16 dims so the binary index can be loaded as
//! `[]align(64) const i16` and consumed by AVX2 kernels (16 × i16 = 32 bytes
//! per vector, two vectors per ymm cache line).
//!
//! Saturation clamps to [INT16_MIN, INT16_MAX] before the cast so that any
//! out-of-range input (which should not happen for the dataset, but we defend
//! anyway) becomes a representable value rather than wrapping silently.

const std = @import("std");

/// Multiplicative factor applied before rounding. Must match `fmt.SCALE` in
/// src/index/format.zig — duplicated here because build_index/ is a separate
/// module and Zig 0.13 disallows escaping a module's root via `..` imports.
pub const SCALE: i32 = 10000;

/// Quantize a batch of 14-dim f32 vectors into 16-dim padded i16 vectors.
/// The two trailing lanes (indices 14 and 15) are zeroed so they contribute
/// nothing to the squared-distance kernel.
pub fn quantizeBatch(out: [][16]i16, vectors: [][14]f32) void {
    for (vectors, 0..) |v, i| {
        inline for (0..14) |j| {
            const x: f32 = v[j] * @as(f32, @floatFromInt(SCALE));
            const c: f32 = @max(-32768.0, @min(32767.0, x));
            out[i][j] = @intFromFloat(@round(c));
        }
        out[i][14] = 0;
        out[i][15] = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "quantize basic" {
    var vs: [2][14]f32 = .{ .{0} ** 14, .{0} ** 14 };
    vs[0][0] = 0.5;
    vs[1][0] = -0.3;
    var out: [2][16]i16 = undefined;
    quantizeBatch(&out, &vs);
    try std.testing.expectEqual(@as(i16, 5000), out[0][0]);
    try std.testing.expectEqual(@as(i16, -3000), out[1][0]);
    try std.testing.expectEqual(@as(i16, 0), out[0][14]);
    try std.testing.expectEqual(@as(i16, 0), out[0][15]);
}

test "quantize saturates extremes" {
    var vs: [1][14]f32 = .{.{0} ** 14};
    vs[0][0] = 100.0; // 100 * 10_000 = 1_000_000 → clamped to 32767
    vs[0][1] = -100.0; // → clamped to -32768
    var out: [1][16]i16 = undefined;
    quantizeBatch(&out, &vs);
    try std.testing.expectEqual(@as(i16, 32767), out[0][0]);
    try std.testing.expectEqual(@as(i16, -32768), out[0][1]);
}

test "quantize rounds to nearest" {
    var vs: [1][14]f32 = .{.{0} ** 14};
    // 0.00005 * 10000 = 0.5 → rounds to 1 (banker's rounding via @round goes
    // to nearest even, but for 0.5 specifically @round in Zig 0.13 returns 1).
    vs[0][0] = 0.00015;
    var out: [1][16]i16 = undefined;
    quantizeBatch(&out, &vs);
    // 0.00015 * 10000 = 1.5 → @round → 2
    try std.testing.expectEqual(@as(i16, 2), out[0][0]);
}
