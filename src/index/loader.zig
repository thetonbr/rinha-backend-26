const std = @import("std");
const fmt = @import("format.zig");

// V2 SoA index layout (mirrors writer.zig):
//
//   [Header                                         ] 64 bytes
//   [centroids: nlist × 16 × f32                    ]  align(64) implicit
//   [invlist_offsets: (nlist + 1) × u32             ]
//   [bbox_min: nlist × 14 × i16                     ]
//   [bbox_max: nlist × 14 × i16                     ]
//   [pad to BLOCK_ALIGN                             ]
//   [orig_ids: n_vectors × u32                      ]  align(64)
//   [pad to BLOCK_ALIGN]  [dim0: n_vectors × i16    ]  align(64)
//   ... 14 blocks total ...
//   [pad to BLOCK_ALIGN]  [dim13: n_vectors × i16   ]  align(64)
//   [labels: ⌈n_vectors / 8⌉ bytes                  ]
pub const Index = struct {
    header: *const fmt.Header,
    centroids: []align(64) const f32, // [nlist * 16]
    invlist_offsets: []const u32, // [nlist + 1]
    bbox_min: []const i16, // [nlist * 14]
    bbox_max: []const i16, // [nlist * 14]
    orig_ids: []align(64) const u32, // [n_vectors]
    dims: [fmt.DIM][]align(64) const i16, // [14][n_vectors]
    labels: []const u8, // bitset, ⌈n_vectors/8⌉
    raw: []align(std.mem.page_size) const u8,
    // Reciprocal of the quantization scale, pre-computed once at load time so
    // the hot path (query f32 conversion, centroid stage) avoids a runtime
    // divide and an int->float cast per request.
    inv_scale: f32,
};

pub fn load(path: []const u8) !Index {
    const fd = try std.posix.open(path, .{ .ACCMODE = .RDONLY }, 0);
    defer std.posix.close(fd);
    const stat = try std.posix.fstat(fd);
    const size: usize = @intCast(stat.size);

    // MAP_SHARED so kernel page cache is reused across forks; MAP_POPULATE
    // pre-faults pages so the first hot-path access does not trigger a minor
    // fault (saves ~1µs per cold page on the first request).
    const raw = try std.posix.mmap(
        null,
        size,
        std.posix.PROT.READ,
        .{ .TYPE = .SHARED, .POPULATE = true },
        fd,
        0,
    );
    errdefer std.posix.munmap(raw);

    // Best-effort: ask the kernel to back the mapping with 2 MB transparent
    // hugepages. mlock is omitted because the contest forbids extra container
    // capabilities; MAP_POPULATE above already prefaults every page so the
    // working set is resident before the first request.
    _ = std.os.linux.madvise(@ptrCast(raw.ptr), size, std.os.linux.MADV.HUGEPAGE);

    if (size < @sizeOf(fmt.Header)) return error.IndexTooSmall;
    const hdr: *const fmt.Header = @ptrCast(@alignCast(raw.ptr));
    if (hdr.magic != fmt.MAGIC) return error.BadMagic;
    if (hdr.version != fmt.VERSION) return error.BadVersion;
    if (hdr.dim != fmt.DIM) return error.BadDim;

    var off: usize = @sizeOf(fmt.Header);

    // Centroids: stored 16-wide (lanes 14/15 zero) so AVX2 ymm loads are valid.
    const cent_count: usize = hdr.nlist * 16;
    const cent_bytes = cent_count * @sizeOf(f32);
    const cent_ptr: [*]align(64) const f32 = @ptrCast(@alignCast(raw[off..].ptr));
    const centroids: []align(64) const f32 = cent_ptr[0..cent_count];
    off += cent_bytes;

    // Invlist offsets.
    const off_count: usize = hdr.nlist + 1;
    const off_bytes = off_count * @sizeOf(u32);
    const off_ptr: [*]const u32 = @ptrCast(@alignCast(raw[off..].ptr));
    const offsets: []const u32 = off_ptr[0..off_count];
    off += off_bytes;

    // Per-cluster bounding boxes (used by stage 3 repair).
    const bbox_count: usize = hdr.nlist * fmt.DIM;
    const bbox_bytes = bbox_count * @sizeOf(i16);
    const bmin_ptr: [*]const i16 = @ptrCast(@alignCast(raw[off..].ptr));
    const bbox_min: []const i16 = bmin_ptr[0..bbox_count];
    off += bbox_bytes;
    const bmax_ptr: [*]const i16 = @ptrCast(@alignCast(raw[off..].ptr));
    const bbox_max: []const i16 = bmax_ptr[0..bbox_count];
    off += bbox_bytes;

    // orig_ids: aligned to BLOCK_ALIGN so it lands on a fresh cache line.
    off = std.mem.alignForward(usize, off, fmt.BLOCK_ALIGN);
    const oid_count: usize = hdr.n_vectors;
    const oid_bytes = oid_count * @sizeOf(u32);
    const oid_ptr: [*]align(64) const u32 = @ptrCast(@alignCast(raw[off..].ptr));
    const orig_ids: []align(64) const u32 = oid_ptr[0..oid_count];
    off += oid_bytes;

    // 14 dim arrays, each aligned to BLOCK_ALIGN. 64-byte alignment lets the
    // scan loop read 32B (AVX2) chunks without a misaligned penalty regardless
    // of the per-dim length parity.
    var dims: [fmt.DIM][]align(64) const i16 = undefined;
    inline for (0..fmt.DIM) |j| {
        off = std.mem.alignForward(usize, off, fmt.BLOCK_ALIGN);
        const dj_ptr: [*]align(64) const i16 = @ptrCast(@alignCast(raw[off..].ptr));
        dims[j] = dj_ptr[0..hdr.n_vectors];
        off += hdr.n_vectors * @sizeOf(i16);
    }

    const labels_bytes: usize = (hdr.n_vectors + 7) / 8;
    const labels: []const u8 = raw[off..][0..labels_bytes];

    const inv_scale: f32 = 1.0 / @as(f32, @floatFromInt(hdr.scale));

    return .{
        .header = hdr,
        .centroids = centroids,
        .invlist_offsets = offsets,
        .bbox_min = bbox_min,
        .bbox_max = bbox_max,
        .orig_ids = orig_ids,
        .dims = dims,
        .labels = labels,
        .raw = raw,
        .inv_scale = inv_scale,
    };
}

pub fn unload(idx: *Index) void {
    std.posix.munmap(@constCast(idx.raw));
}

// Internal helper used by tests and by ivf.zig's own synthetic test to keep the
// V2 layout in lock-step with the writer. Emits a minimal index with `n_vectors`
// vectors split into 2 invlists (first half / second half), all bbox fields
// zero, identity orig_ids, and 14 dim blocks initialized from `dims_init`.
pub fn writeSyntheticV2(
    path: []const u8,
    n_vectors: u32,
    nlist: u32,
    centroids_padded16: []const [16]f32,
    invlist_offsets: []const u32,
    bbox_min: []const [fmt.DIM]i16,
    bbox_max: []const [fmt.DIM]i16,
    orig_ids: []const u32,
    dims_init: []const []const i16,
    labels: []const u8,
) !void {
    std.debug.assert(centroids_padded16.len == nlist);
    std.debug.assert(invlist_offsets.len == nlist + 1);
    std.debug.assert(bbox_min.len == nlist);
    std.debug.assert(bbox_max.len == nlist);
    std.debug.assert(orig_ids.len == n_vectors);
    std.debug.assert(dims_init.len == fmt.DIM);

    const f = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer f.close();
    const w = f.writer();

    const hdr = fmt.Header{
        .magic = fmt.MAGIC,
        .version = fmt.VERSION,
        .n_vectors = n_vectors,
        .dim = fmt.DIM,
        .nlist = nlist,
        .scale = fmt.SCALE,
        .reserved = std.mem.zeroes([40]u8),
    };
    try w.writeAll(std.mem.asBytes(&hdr));
    try w.writeAll(std.mem.sliceAsBytes(centroids_padded16));
    try w.writeAll(std.mem.sliceAsBytes(invlist_offsets));
    try w.writeAll(std.mem.sliceAsBytes(bbox_min));
    try w.writeAll(std.mem.sliceAsBytes(bbox_max));

    var off: usize = @sizeOf(fmt.Header) +
        nlist * @sizeOf([16]f32) +
        invlist_offsets.len * @sizeOf(u32) +
        nlist * @sizeOf([fmt.DIM]i16) * 2;

    // Pad before orig_ids.
    {
        const aligned = std.mem.alignForward(usize, off, fmt.BLOCK_ALIGN);
        const pad: [fmt.BLOCK_ALIGN]u8 = .{0} ** fmt.BLOCK_ALIGN;
        if (aligned > off) try w.writeAll(pad[0 .. aligned - off]);
        off = aligned;
    }
    try w.writeAll(std.mem.sliceAsBytes(orig_ids));
    off += n_vectors * @sizeOf(u32);

    // 14 dim blocks each aligned to 64 bytes.
    inline for (0..fmt.DIM) |j| {
        const aligned = std.mem.alignForward(usize, off, fmt.BLOCK_ALIGN);
        const pad: [fmt.BLOCK_ALIGN]u8 = .{0} ** fmt.BLOCK_ALIGN;
        if (aligned > off) try w.writeAll(pad[0 .. aligned - off]);
        off = aligned;
        try w.writeAll(std.mem.sliceAsBytes(dims_init[j]));
        off += n_vectors * @sizeOf(i16);
    }

    try w.writeAll(labels);
}

test "load synthetic V2 index" {
    const path = "/tmp/rinha_test_index_v2.bin";
    defer std.fs.cwd().deleteFile(path) catch {};

    const n_vectors: u32 = 8;
    const nlist: u32 = 2;
    const centroids: [2][16]f32 = .{ .{0.0} ** 16, .{0.0} ** 16 };
    const offsets: [3]u32 = .{ 0, 4, 8 };
    const bbox_min: [2][fmt.DIM]i16 = .{ .{0} ** fmt.DIM, .{0} ** fmt.DIM };
    const bbox_max: [2][fmt.DIM]i16 = .{ .{0} ** fmt.DIM, .{0} ** fmt.DIM };
    const orig_ids: [8]u32 = .{ 7, 6, 5, 4, 3, 2, 1, 0 };
    const dim_zero: [8]i16 = .{0} ** 8;
    var dim_init: [fmt.DIM][]const i16 = undefined;
    inline for (0..fmt.DIM) |j| dim_init[j] = dim_zero[0..];
    const labels: [1]u8 = .{0b00001111};

    try writeSyntheticV2(
        path,
        n_vectors,
        nlist,
        centroids[0..],
        offsets[0..],
        bbox_min[0..],
        bbox_max[0..],
        orig_ids[0..],
        dim_init[0..],
        labels[0..],
    );

    var idx = try load(path);
    defer unload(&idx);

    try std.testing.expectEqual(fmt.MAGIC, idx.header.magic);
    try std.testing.expectEqual(@as(u32, 2), idx.header.version);
    try std.testing.expectEqual(@as(u32, 8), idx.header.n_vectors);
    try std.testing.expectEqual(@as(u32, 2), idx.header.nlist);
    try std.testing.expectEqual(@as(usize, 2 * 16), idx.centroids.len);
    try std.testing.expectEqual(@as(usize, 3), idx.invlist_offsets.len);
    try std.testing.expectEqual(@as(u32, 4), idx.invlist_offsets[1]);
    try std.testing.expectEqual(@as(usize, nlist * fmt.DIM), idx.bbox_min.len);
    try std.testing.expectEqual(@as(usize, nlist * fmt.DIM), idx.bbox_max.len);
    try std.testing.expectEqual(@as(usize, 8), idx.orig_ids.len);
    try std.testing.expectEqual(@as(u32, 7), idx.orig_ids[0]);
    try std.testing.expectEqual(@as(u32, 0), idx.orig_ids[7]);
    inline for (0..fmt.DIM) |j| {
        try std.testing.expectEqual(@as(usize, 8), idx.dims[j].len);
    }
    try std.testing.expectEqual(@as(usize, 1), idx.labels.len);
    try std.testing.expectEqual(@as(u8, 0b00001111), idx.labels[0]);
}
