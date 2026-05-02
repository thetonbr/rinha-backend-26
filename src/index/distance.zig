const std = @import("std");

pub const DIM: usize = 14;
pub const DIM_PADDED: usize = 16;
const Vec8f = @Vector(8, f32);
const Vec16f = @Vector(16, f32);
const Vec16i16 = @Vector(16, i16);
const Vec16i32 = @Vector(16, i32);

pub inline fn euclideanSqF32(a: *const [DIM]f32, b: *const [DIM]f32) f32 {
    const va: Vec8f = a[0..8].*;
    const vb: Vec8f = b[0..8].*;
    const d = va - vb;
    var acc: f32 = @reduce(.Add, d * d);
    inline for (8..DIM) |i| {
        const x = a[i] - b[i];
        acc += x * x;
    }
    return acc;
}

// Squared Euclidean distance for 16-wide padded f32 vectors. Lanes 14/15
// must be zero on both inputs (the Rinha quantizer pads with zeros). This
// kernel is wider than euclideanSqF32 — use it when both query and reference
// are already padded, e.g. centroids and re-rank candidates.
pub inline fn euclideanSqF32Padded(a: *const [DIM_PADDED]f32, b: *const [DIM_PADDED]f32) f32 {
    const va: Vec16f = a.*;
    const vb: Vec16f = b.*;
    const d = va - vb;
    return @reduce(.Add, d * d);
}

// Squared Euclidean distance between an f32 query (padded to 16) and a
// quantized i16 reference (padded), with on-the-fly conversion via the
// scale. Replaces the inline-for scalar tail in Stage 3 re-rank.
pub inline fn euclideanSqF32QueryI16Ref(
    q: *const [DIM_PADDED]f32,
    r: *align(32) const [DIM_PADDED]i16,
    inv_scale: f32,
) f32 {
    const ri16: Vec16i16 = r.*;
    const ri32: Vec16i32 = ri16;
    const rf32: Vec16f = @floatFromInt(ri32);
    const inv: Vec16f = @splat(inv_scale);
    const ref: Vec16f = rf32 * inv;
    const qv: Vec16f = q.*;
    const d = qv - ref;
    return @reduce(.Add, d * d);
}

pub inline fn euclideanSqI16Padded(a: *align(32) const [DIM_PADDED]i16, b: *align(32) const [DIM_PADDED]i16) i64 {
    const va: Vec16i16 = a.*;
    const vb: Vec16i16 = b.*;
    const da: Vec16i32 = va;
    const db: Vec16i32 = vb;
    const d = da - db;
    const dsq: @Vector(16, i32) = d * d; // each lane <= (2*SCALE)^2 = 4e8, fits i32
    return @reduce(.Add, @as(@Vector(16, i64), dsq)); // widen before reduce: 14*4e8=5.6e9 > i32_max
}

fn scalarSqF32(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0;
    for (a, b) |x, y| sum += (x - y) * (x - y);
    return sum;
}

fn scalarSqI16(a: []const i16, b: []const i16) i64 {
    var sum: i64 = 0;
    for (a, b) |x, y| {
        const d: i64 = @as(i64, x) - @as(i64, y);
        sum += d * d;
    }
    return sum;
}

test "euclidean f32 matches scalar" {
    var a: [DIM]f32 = undefined;
    var b: [DIM]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(42);
    const r = prng.random();
    for (0..DIM) |i| {
        a[i] = r.float(f32);
        b[i] = r.float(f32);
    }
    const expected = scalarSqF32(a[0..DIM], b[0..DIM]);
    const got = euclideanSqF32(&a, &b);
    try std.testing.expectApproxEqRel(expected, got, 1e-5);
}

test "euclidean f32 padded matches scalar" {
    var a: [DIM_PADDED]f32 = .{0.0} ** DIM_PADDED;
    var b: [DIM_PADDED]f32 = .{0.0} ** DIM_PADDED;
    var prng = std.Random.DefaultPrng.init(13);
    const r = prng.random();
    for (0..DIM) |i| {
        a[i] = r.float(f32);
        b[i] = r.float(f32);
    }
    // Padding lanes stay 0 — must match the DIM-only scalar baseline.
    const expected = scalarSqF32(a[0..DIM], b[0..DIM]);
    const got = euclideanSqF32Padded(&a, &b);
    try std.testing.expectApproxEqRel(expected, got, 1e-5);
}

test "euclidean f32-query/i16-ref matches scalar" {
    var q: [DIM_PADDED]f32 = .{0.0} ** DIM_PADDED;
    var ref: [DIM_PADDED]i16 align(64) = .{0} ** DIM_PADDED;
    var prng = std.Random.DefaultPrng.init(99);
    const rnd = prng.random();
    const inv_scale: f32 = 1.0 / 10000.0;
    for (0..DIM) |i| {
        q[i] = rnd.float(f32);
        ref[i] = rnd.intRangeAtMost(i16, -10000, 10000);
    }
    var ref_f: [DIM]f32 = undefined;
    for (0..DIM) |i| ref_f[i] = @as(f32, @floatFromInt(ref[i])) * inv_scale;
    const expected = scalarSqF32(q[0..DIM], ref_f[0..DIM]);
    const got = euclideanSqF32QueryI16Ref(&q, &ref, inv_scale);
    try std.testing.expectApproxEqRel(expected, got, 1e-5);
}

test "euclidean i16 padded matches scalar" {
    var a: [DIM_PADDED]i16 align(64) = .{0} ** DIM_PADDED;
    var b: [DIM_PADDED]i16 align(64) = .{0} ** DIM_PADDED;
    var prng = std.Random.DefaultPrng.init(7);
    const r = prng.random();
    for (0..DIM) |i| {
        a[i] = r.intRangeAtMost(i16, -10000, 10000);
        b[i] = r.intRangeAtMost(i16, -10000, 10000);
    }
    // padding stays 0; matches scalar over 16-wide
    const expected = scalarSqI16(a[0..DIM_PADDED], b[0..DIM_PADDED]);
    const got = euclideanSqI16Padded(&a, &b);
    try std.testing.expectEqual(expected, got);
}
