//! Build-time writer for the V2 binary IVF index (`index.bin`).
//!
//! Layout (little-endian, x86_64):
//!
//!   [Header                                  ] 64 bytes
//!   [centroids: nlist × 16 × f32             ]
//!   [invlist_offsets: (nlist+1) × u32        ]
//!   [bbox_min: nlist × 14 × i16              ]
//!   [bbox_max: nlist × 14 × i16              ]
//!   [pad to BLOCK_ALIGN                      ]
//!   [orig_ids: n_vectors × u32               ] aligned to 64 bytes
//!   [pad to BLOCK_ALIGN] [dim0: n × i16      ] aligned to 64 bytes
//!   ... 14 dim blocks total ...
//!   [pad to BLOCK_ALIGN] [dim13: n × i16     ] aligned to 64 bytes
//!   [fraud_bits: ⌈n/8⌉ bytes                 ]
//!
//! Padding before each aligned block is mandatory: the runtime loader exposes
//! orig_ids and each dim as `[]align(64) const T` for AVX2. Without padding
//! the offsets desynchronise and the index is unreadable.

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
    /// nlist centroids, each padded from 14 → 16 f32 lanes (lanes 14/15 zero).
    centroids_padded: [][16]f32,
    /// nlist + 1 prefix-sum offsets into the per-dim arrays.
    invlist_offsets: []u32,
    /// nlist × 14 i16 minimum coordinates per cluster (axis-aligned bbox).
    bbox_min: [][fmt.DIM]i16,
    /// nlist × 14 i16 maximum coordinates per cluster.
    bbox_max: [][fmt.DIM]i16,
    /// n_vectors original-dataset positions, in invlist-sorted order. Used as
    /// the deterministic tie-break key in the runtime top-5.
    orig_ids: []u32,
    /// dims[j][i] = quantized i16 value of dim j for the i-th invlist-sorted
    /// vector. The 14 arrays are written contiguously, each aligned to 64.
    dims: [fmt.DIM][]i16,
    /// Bitset (LSB-first within each byte) flagging fraud labels per vector,
    /// in the same invlist-sorted order as `orig_ids` and `dims`.
    fraud_bits: []u8,
};

/// Emit `index.bin` at `args.out_path` according to the V2 layout above.
pub fn write(args: Args) !void {
    std.debug.assert(args.centroids_padded.len == args.nlist);
    std.debug.assert(args.invlist_offsets.len == args.nlist + 1);
    std.debug.assert(args.bbox_min.len == args.nlist);
    std.debug.assert(args.bbox_max.len == args.nlist);
    std.debug.assert(args.orig_ids.len == args.n_vectors);
    inline for (0..fmt.DIM) |j| {
        std.debug.assert(args.dims[j].len == args.n_vectors);
    }

    const f = try std.fs.createFileAbsolute(args.out_path, .{ .truncate = true });
    defer f.close();
    var bw = std.io.bufferedWriter(f.writer());
    var w = bw.writer();

    const hdr = fmt.Header{
        .magic = fmt.MAGIC,
        .version = fmt.VERSION,
        .n_vectors = args.n_vectors,
        .dim = fmt.DIM,
        .nlist = args.nlist,
        .scale = fmt.SCALE,
        .reserved = std.mem.zeroes([40]u8),
    };
    try w.writeAll(std.mem.asBytes(&hdr));
    try w.writeAll(std.mem.sliceAsBytes(args.centroids_padded));
    try w.writeAll(std.mem.sliceAsBytes(args.invlist_offsets));
    try w.writeAll(std.mem.sliceAsBytes(args.bbox_min));
    try w.writeAll(std.mem.sliceAsBytes(args.bbox_max));

    var off: usize =
        @sizeOf(fmt.Header) +
        args.centroids_padded.len * @sizeOf([16]f32) +
        args.invlist_offsets.len * @sizeOf(u32) +
        args.bbox_min.len * @sizeOf([fmt.DIM]i16) * 2;

    // Pad before orig_ids so it starts on a 64-byte boundary.
    {
        const aligned: usize = std.mem.alignForward(usize, off, fmt.BLOCK_ALIGN);
        if (aligned > off) {
            const pad = [_]u8{0} ** fmt.BLOCK_ALIGN;
            try w.writeAll(pad[0 .. aligned - off]);
        }
        off = aligned;
    }
    try w.writeAll(std.mem.sliceAsBytes(args.orig_ids));
    off += args.orig_ids.len * @sizeOf(u32);

    // 14 dim blocks each aligned to 64 bytes.
    inline for (0..fmt.DIM) |j| {
        const aligned: usize = std.mem.alignForward(usize, off, fmt.BLOCK_ALIGN);
        if (aligned > off) {
            const pad = [_]u8{0} ** fmt.BLOCK_ALIGN;
            try w.writeAll(pad[0 .. aligned - off]);
        }
        off = aligned;
        try w.writeAll(std.mem.sliceAsBytes(args.dims[j]));
        off += args.dims[j].len * @sizeOf(i16);
    }

    try w.writeAll(args.fraud_bits);

    try bw.flush();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "writer emits a roundtrippable V2 binary index" {
    const allocator = std.testing.allocator;

    const tmp_path = "/tmp/rinha2026_writer_test_index_v2.bin";
    defer std.fs.deleteFileAbsolute(tmp_path) catch {};

    const n_vectors: u32 = 8;
    const nlist: u32 = 2;

    const centroids = try allocator.alloc([16]f32, nlist);
    defer allocator.free(centroids);
    for (centroids, 0..) |*c, i| {
        c.* = .{0} ** 16;
        c[0] = @floatFromInt(i + 1);
    }

    var offsets = try allocator.alloc(u32, nlist + 1);
    defer allocator.free(offsets);
    offsets[0] = 0;
    offsets[1] = 4;
    offsets[2] = 8;

    const bbox_min = try allocator.alloc([fmt.DIM]i16, nlist);
    defer allocator.free(bbox_min);
    const bbox_max = try allocator.alloc([fmt.DIM]i16, nlist);
    defer allocator.free(bbox_max);
    for (bbox_min, bbox_max) |*lo, *hi| {
        lo.* = .{0} ** fmt.DIM;
        hi.* = .{0} ** fmt.DIM;
    }

    const orig_ids = try allocator.alloc(u32, n_vectors);
    defer allocator.free(orig_ids);
    for (orig_ids, 0..) |*o, i| o.* = @intCast(100 + i);

    var dims_storage: [fmt.DIM][]i16 = undefined;
    inline for (0..fmt.DIM) |j| {
        dims_storage[j] = try allocator.alloc(i16, n_vectors);
        for (dims_storage[j], 0..) |*v, i| v.* = @intCast(j * 100 + i);
    }
    defer inline for (0..fmt.DIM) |j| allocator.free(dims_storage[j]);

    var fraud_bits = try allocator.alloc(u8, 1);
    defer allocator.free(fraud_bits);
    fraud_bits[0] = 0b00001111;

    try write(.{
        .out_path = tmp_path,
        .n_vectors = n_vectors,
        .nlist = nlist,
        .centroids_padded = centroids,
        .invlist_offsets = offsets,
        .bbox_min = bbox_min,
        .bbox_max = bbox_max,
        .orig_ids = orig_ids,
        .dims = dims_storage,
        .fraud_bits = fraud_bits,
    });

    // Read back and validate the layout manually (we cannot import the loader
    // from build_index/ — Zig 0.13 forbids `..` paths).
    const data = try std.fs.cwd().readFileAlloc(allocator, tmp_path, 1 << 20);
    defer allocator.free(data);

    const hdr: *const fmt.Header = @ptrCast(@alignCast(data.ptr));
    try std.testing.expectEqual(fmt.MAGIC, hdr.magic);
    try std.testing.expectEqual(fmt.VERSION, hdr.version);
    try std.testing.expectEqual(n_vectors, hdr.n_vectors);
    try std.testing.expectEqual(fmt.DIM, hdr.dim);
    try std.testing.expectEqual(nlist, hdr.nlist);
    try std.testing.expectEqual(fmt.SCALE, hdr.scale);

    // Walk the layout to find the orig_ids block.
    var off: usize =
        @sizeOf(fmt.Header) +
        nlist * @sizeOf([16]f32) +
        (nlist + 1) * @sizeOf(u32) +
        nlist * @sizeOf([fmt.DIM]i16) * 2;
    const oid_off: usize = std.mem.alignForward(usize, off, fmt.BLOCK_ALIGN);
    for (data[off..oid_off]) |b| try std.testing.expectEqual(@as(u8, 0), b);

    const oid_ptr: [*]const u32 = @ptrCast(@alignCast(data[oid_off..].ptr));
    for (0..n_vectors) |i| try std.testing.expectEqual(@as(u32, @intCast(100 + i)), oid_ptr[i]);
    off = oid_off + n_vectors * @sizeOf(u32);

    inline for (0..fmt.DIM) |j| {
        const dim_off: usize = std.mem.alignForward(usize, off, fmt.BLOCK_ALIGN);
        for (data[off..dim_off]) |b| try std.testing.expectEqual(@as(u8, 0), b);
        const dim_ptr: [*]const i16 = @ptrCast(@alignCast(data[dim_off..].ptr));
        for (0..n_vectors) |i| {
            try std.testing.expectEqual(@as(i16, @intCast(j * 100 + i)), dim_ptr[i]);
        }
        off = dim_off + n_vectors * @sizeOf(i16);
    }

    try std.testing.expectEqual(@as(u8, 0b00001111), data[off]);
    try std.testing.expectEqual(off + 1, data.len);
}
