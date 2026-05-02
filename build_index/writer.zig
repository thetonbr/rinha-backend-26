//! Build-time writer for the binary IVF index (`index.bin`).
//!
//! Layout (little-endian, x86_64):
//!
//!   [Header                      ] 64 bytes
//!   [centroids: nlist × 16 × f32 ]
//!   [invlist_offsets: (nlist+1) × u32]
//!   [pad to VECTORS_BLOCK_ALIGN  ] 0..63 bytes of zeros
//!   [vectors_q_sorted: n × 16 × i16] aligned to 64 bytes
//!   [fraud_bits: ⌈n/8⌉ bytes     ]
//!
//! The padding before the vectors block is mandatory: the runtime loader
//! exposes that block as `[]align(64) const i16` so AVX2 ymm loads are valid,
//! and `loader.load` skips bytes via `alignForward`. If the writer omits the
//! padding the offsets desynchronise and the index is corrupt.

const std = @import("std");
// `fmt` is exposed as a named module by build.zig (and by the standalone
// `zig test` invocation in the project's verify recipe), pointing at
// src/index/format.zig. We cannot import it via a relative path because Zig
// 0.13 forbids `..` escaping a module's root directory.
const fmt = @import("fmt");

pub const Args = struct {
    /// Absolute path of the destination index.bin file.
    out_path: []const u8,
    /// Total number of vectors in the index.
    n_vectors: u32,
    /// Number of inverted lists (must match `centroids_padded.len`).
    nlist: u32,
    /// nlist centroids, each padded from 14 → 16 f32 lanes.
    centroids_padded: [][16]f32,
    /// nlist + 1 prefix-sum offsets into `vectors_q_sorted`.
    invlist_offsets: []u32,
    /// n_vectors quantized vectors, sorted so each invlist is contiguous.
    vectors_q_sorted: [][16]i16,
    /// Bitset (LSB-first within each byte) flagging fraud labels per vector.
    fraud_bits: []u8,
};

/// Emit `index.bin` at `args.out_path` according to the layout above.
pub fn write(args: Args) !void {
    const f = try std.fs.createFileAbsolute(args.out_path, .{ .truncate = true });
    defer f.close();
    var bw = std.io.bufferedWriter(f.writer());
    var w = bw.writer();

    const hdr = fmt.Header{
        .magic = fmt.MAGIC,
        .version = fmt.VERSION,
        .n_vectors = args.n_vectors,
        .dim = fmt.DIM,
        .dim_padded = fmt.DIM_PADDED,
        .nlist = args.nlist,
        .scale = fmt.SCALE,
        .reserved = std.mem.zeroes([36]u8),
    };
    try w.writeAll(std.mem.asBytes(&hdr));
    try w.writeAll(std.mem.sliceAsBytes(args.centroids_padded));
    try w.writeAll(std.mem.sliceAsBytes(args.invlist_offsets));

    // Pad to VECTORS_BLOCK_ALIGN (64) so the vectors block starts on a 64-byte
    // boundary; the loader expects to read it as []align(64) const i16 for
    // AVX2. Without this the vectors block lands on the wrong offset and the
    // index is unreadable.
    const written: usize = @sizeOf(fmt.Header) + args.centroids_padded.len * @sizeOf([16]f32) + args.invlist_offsets.len * @sizeOf(u32);
    const aligned: usize = std.mem.alignForward(usize, written, fmt.VECTORS_BLOCK_ALIGN);
    if (aligned > written) {
        const pad = [_]u8{0} ** fmt.VECTORS_BLOCK_ALIGN;
        try w.writeAll(pad[0 .. aligned - written]);
    }

    try w.writeAll(std.mem.sliceAsBytes(args.vectors_q_sorted));
    try w.writeAll(args.fraud_bits);

    try bw.flush();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "writer emits a roundtrippable binary index" {
    const allocator = std.testing.allocator;

    const tmp_path = "/tmp/rinha2026_writer_test_index.bin";
    defer std.fs.deleteFileAbsolute(tmp_path) catch {};

    // Two centroids, two invlists, eight vectors. Use distinguishable values
    // so we can pinpoint mis-offsets by inspecting the raw bytes.
    const centroids = try allocator.alloc([16]f32, 2);
    defer allocator.free(centroids);
    for (centroids, 0..) |*c, i| {
        c.* = .{0} ** 16;
        c[0] = @floatFromInt(i + 1);
    }

    var offsets = try allocator.alloc(u32, 3);
    defer allocator.free(offsets);
    offsets[0] = 0;
    offsets[1] = 4;
    offsets[2] = 8;

    const vectors = try allocator.alloc([16]i16, 8);
    defer allocator.free(vectors);
    for (vectors, 0..) |*v, i| {
        v.* = .{0} ** 16;
        v[0] = @intCast(i + 100);
    }

    var fraud_bits = try allocator.alloc(u8, 1);
    defer allocator.free(fraud_bits);
    fraud_bits[0] = 0b00001111;

    try write(.{
        .out_path = tmp_path,
        .n_vectors = 8,
        .nlist = 2,
        .centroids_padded = centroids,
        .invlist_offsets = offsets,
        .vectors_q_sorted = vectors,
        .fraud_bits = fraud_bits,
    });

    // Read the file back and validate the layout manually (we cannot import
    // src/index/loader.zig from this module — Zig 0.13 forbids `..` paths).
    const data = try std.fs.cwd().readFileAlloc(allocator, tmp_path, 1 << 20);
    defer allocator.free(data);

    // Header.
    const hdr: *const fmt.Header = @ptrCast(@alignCast(data.ptr));
    try std.testing.expectEqual(fmt.MAGIC, hdr.magic);
    try std.testing.expectEqual(fmt.VERSION, hdr.version);
    try std.testing.expectEqual(@as(u32, 8), hdr.n_vectors);
    try std.testing.expectEqual(fmt.DIM, hdr.dim);
    try std.testing.expectEqual(fmt.DIM_PADDED, hdr.dim_padded);
    try std.testing.expectEqual(@as(u32, 2), hdr.nlist);
    try std.testing.expectEqual(fmt.SCALE, hdr.scale);

    // Compute expected offset of the vectors block.
    const after_header: usize = @sizeOf(fmt.Header);
    const after_centroids: usize = after_header + 2 * @sizeOf([16]f32);
    const after_offsets: usize = after_centroids + 3 * @sizeOf(u32);
    const vec_off: usize = std.mem.alignForward(usize, after_offsets, fmt.VECTORS_BLOCK_ALIGN);

    // The padding bytes between offsets and vectors must all be zero.
    for (data[after_offsets..vec_off]) |b| {
        try std.testing.expectEqual(@as(u8, 0), b);
    }

    // The vectors block itself must round-trip the lane-0 sentinels.
    const vec_ptr: [*]const i16 = @ptrCast(@alignCast(data[vec_off..].ptr));
    for (0..8) |i| {
        try std.testing.expectEqual(@as(i16, @intCast(i + 100)), vec_ptr[i * 16]);
    }

    // Fraud bits live immediately after the vectors block.
    const fraud_off: usize = vec_off + 8 * @sizeOf([16]i16);
    try std.testing.expectEqual(@as(u8, 0b00001111), data[fraud_off]);

    // And the file ends right after the bitset.
    try std.testing.expectEqual(fraud_off + 1, data.len);
}
