const std = @import("std");
const fmt = @import("format.zig");

pub const Index = struct {
    header: *const fmt.Header,
    centroids: []align(64) const f32, // [nlist * 16]
    invlist_offsets: []const u32, // [nlist + 1]
    vectors: []align(64) const i16, // [n_vectors * 16]
    labels: []const u8, // bitset, (n_vectors + 7) / 8
    raw: []align(std.mem.page_size) const u8,
    // Reciprocal of the quantization scale, pre-computed once at load time so
    // the hot path (Stage 3 re-rank, query f32 conversion) avoids a runtime
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

    if (size < @sizeOf(fmt.Header)) return error.IndexTooSmall;
    const hdr: *const fmt.Header = @ptrCast(@alignCast(raw.ptr));
    if (hdr.magic != fmt.MAGIC) return error.BadMagic;
    if (hdr.version != fmt.VERSION) return error.BadVersion;
    if (hdr.dim != fmt.DIM or hdr.dim_padded != fmt.DIM_PADDED) return error.BadDim;

    var off: usize = @sizeOf(fmt.Header);

    const cent_count: usize = hdr.nlist * fmt.DIM_PADDED;
    const cent_bytes = cent_count * @sizeOf(f32);
    const cent_ptr: [*]align(64) const f32 = @ptrCast(@alignCast(raw[off..].ptr));
    const centroids: []align(64) const f32 = cent_ptr[0..cent_count];
    off += cent_bytes;

    const off_count: usize = hdr.nlist + 1;
    const off_bytes = off_count * @sizeOf(u32);
    const off_ptr: [*]const u32 = @ptrCast(@alignCast(raw[off..].ptr));
    const offsets: []const u32 = off_ptr[0..off_count];
    off += off_bytes;

    // Vectors block requires 64-byte alignment for AVX2 loads.
    off = std.mem.alignForward(usize, off, fmt.VECTORS_BLOCK_ALIGN);

    const vec_count: usize = hdr.n_vectors * fmt.DIM_PADDED;
    const vec_bytes = vec_count * @sizeOf(i16);
    const vec_ptr: [*]align(64) const i16 = @ptrCast(@alignCast(raw[off..].ptr));
    const vectors: []align(64) const i16 = vec_ptr[0..vec_count];
    off += vec_bytes;

    const labels_bytes: usize = (hdr.n_vectors + 7) / 8;
    const labels: []const u8 = raw[off..][0..labels_bytes];

    const inv_scale: f32 = 1.0 / @as(f32, @floatFromInt(hdr.scale));

    return .{
        .header = hdr,
        .centroids = centroids,
        .invlist_offsets = offsets,
        .vectors = vectors,
        .labels = labels,
        .raw = raw,
        .inv_scale = inv_scale,
    };
}

pub fn unload(idx: *Index) void {
    std.posix.munmap(@constCast(idx.raw));
}

test "load synthetic index" {
    const path = "/tmp/rinha_test_index.bin";

    // Build synthetic index file: header + 2 centroids + offsets + 8 vectors + labels.
    // The vectors block must start on a 64-byte boundary so the loader can
    // expose it as `[]align(64) const i16` for AVX2 loads. The builder is
    // responsible for inserting padding; we mirror that layout here.
    {
        const f = try std.fs.cwd().createFile(path, .{ .truncate = true });
        defer f.close();
        const w = f.writer();

        const hdr = fmt.Header{
            .magic = fmt.MAGIC,
            .version = fmt.VERSION,
            .n_vectors = 8,
            .dim = fmt.DIM,
            .dim_padded = fmt.DIM_PADDED,
            .nlist = 2,
            .scale = fmt.SCALE,
            .reserved = std.mem.zeroes([36]u8),
        };
        try w.writeAll(std.mem.asBytes(&hdr));

        // 2 centroids of 16 f32 each, all zeros.
        const cent_zero: [2 * 16]f32 = .{0.0} ** 32;
        try w.writeAll(std.mem.sliceAsBytes(cent_zero[0..]));

        // invlist offsets [3]u32 = {0, 4, 8}.
        const offsets: [3]u32 = .{ 0, 4, 8 };
        try w.writeAll(std.mem.sliceAsBytes(offsets[0..]));

        // Pad to 64-byte boundary before vectors block.
        const written: usize = @sizeOf(fmt.Header) + @sizeOf(@TypeOf(cent_zero)) + @sizeOf(@TypeOf(offsets));
        const aligned: usize = std.mem.alignForward(usize, written, 64);
        const pad_zero: [64]u8 = .{0} ** 64;
        try w.writeAll(pad_zero[0 .. aligned - written]);

        // 8 vectors of 16 i16 each, all zeros.
        const vec_zero: [8 * 16]i16 = .{0} ** 128;
        try w.writeAll(std.mem.sliceAsBytes(vec_zero[0..]));

        // labels bitset: 8 bits → 1 byte, value 0b00001111.
        const labels: [1]u8 = .{0b00001111};
        try w.writeAll(labels[0..]);
    }
    defer std.fs.cwd().deleteFile(path) catch {};

    var idx = try load(path);
    defer unload(&idx);

    try std.testing.expectEqual(fmt.MAGIC, idx.header.magic);
    try std.testing.expectEqual(@as(u32, 8), idx.header.n_vectors);
    try std.testing.expectEqual(@as(u32, 2), idx.header.nlist);
    try std.testing.expectEqual(@as(usize, 2 * 16), idx.centroids.len);
    try std.testing.expectEqual(@as(usize, 3), idx.invlist_offsets.len);
    try std.testing.expectEqual(@as(u32, 0), idx.invlist_offsets[0]);
    try std.testing.expectEqual(@as(u32, 4), idx.invlist_offsets[1]);
    try std.testing.expectEqual(@as(u32, 8), idx.invlist_offsets[2]);
    try std.testing.expectEqual(@as(usize, 8 * 16), idx.vectors.len);
    try std.testing.expectEqual(@as(usize, 1), idx.labels.len);
    try std.testing.expectEqual(@as(u8, 0b00001111), idx.labels[0]);
}
