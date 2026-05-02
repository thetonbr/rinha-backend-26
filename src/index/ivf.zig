const std = @import("std");
const dist = @import("distance.zig");
const topk = @import("topk.zig");
const fmt = @import("format.zig");
const loader = @import("loader.zig");

pub const NPROBE: usize = 16;

// Vectors prefetched ahead of current scan position. Tune against the live
// invlist sizes if the benchmark shows the streamer overshooting L2.
pub const PREFETCH_AHEAD: u32 = 16;

pub const SearchResult = struct {
    fraud_count: u8,
};

// Extract the `idx`-th label bit from a packed bitset (LSB-first within each byte).
inline fn labelBit(labels: []const u8, idx: u32) u8 {
    return @intCast((labels[idx >> 3] >> @intCast(idx & 7)) & 1);
}

pub fn search(
    idx: *const loader.Index,
    query_padded: *align(64) const [fmt.DIM_PADDED]i16,
    query_f32_padded: *align(64) const [fmt.DIM_PADDED]f32,
) SearchResult {
    // Stage 1: pick top-NPROBE centroids by float32 distance.
    // probe_dist is f32 (centroids are stored as f32); probe_id is u32 since
    // a centroid id fits in 32 bits and matches invlist_offsets indexing.
    // NPROBE=16 is small enough that insertion sort beats a heap.
    var probe_dist: [NPROBE]f32 = .{std.math.inf(f32)} ** NPROBE;
    var probe_id: [NPROBE]u32 = .{0} ** NPROBE;

    const nlist: usize = idx.header.nlist;
    // Effective probe count when nlist < NPROBE (degenerate small indexes,
    // e.g. tests). In production nlist=2048, so this is just NPROBE.
    const probe_count: usize = if (nlist < NPROBE) nlist else NPROBE;
    var c: usize = 0;
    while (c < nlist) : (c += 1) {
        const c_off = c * fmt.DIM_PADDED;
        // Centroids are stored 16-wide with lanes 14/15 zero, so the wide
        // SIMD kernel produces the same result as the 14-dim scalar tail
        // version but folds the load into a single ymm op.
        const cent: *const [fmt.DIM_PADDED]f32 = idx.centroids[c_off..][0..fmt.DIM_PADDED];
        const d = dist.euclideanSqF32Padded(cent, query_f32_padded);
        if (d < probe_dist[NPROBE - 1]) {
            var pos: usize = NPROBE - 1;
            while (pos > 0 and d < probe_dist[pos - 1]) : (pos -= 1) {
                probe_dist[pos] = probe_dist[pos - 1];
                probe_id[pos] = probe_id[pos - 1];
            }
            probe_dist[pos] = d;
            probe_id[pos] = @intCast(c);
        }
    }

    // Stage 2: scan inverted lists in int16 and feed the top-K' max-heap.
    // Accumulator is i64: per-lane d^2 <= (2*SCALE)^2 fits i32, but the
    // 16-lane reduction can reach 14 * 4e8 = 5.6e9, exceeding i32_max.
    var topkp: topk.TopKp = .{};
    for (probe_id[0..probe_count]) |cid| {
        const start = idx.invlist_offsets[cid];
        const end = idx.invlist_offsets[cid + 1];
        var i = start;
        while (i < end) : (i += 1) {
            if (i + PREFETCH_AHEAD < end) {
                const pf_off = @as(usize, i + PREFETCH_AHEAD) * fmt.DIM_PADDED;
                @prefetch(&idx.vectors[pf_off], .{ .rw = .read, .locality = 1, .cache = .data });
            }
            const v_off = @as(usize, i) * fmt.DIM_PADDED;
            // Each 16xi16 vector is 32 bytes; idx.vectors is align(64), so
            // every other vector lands on align(32). AVX2 ymm loads only
            // require align(32), which is what the kernel signature accepts.
            const v_slice = idx.vectors[v_off..][0..fmt.DIM_PADDED];
            const v: *align(32) const [fmt.DIM_PADDED]i16 = @ptrCast(@alignCast(v_slice));
            const d_int = dist.euclideanSqI16Padded(query_padded, v);
            topkp.maybeInsert(d_int, i);
        }
    }

    // Stage 3: re-rank candidates in float32 and keep top-5 with labels.
    // The wide kernel does i16->f32 conversion + scale + distance in a single
    // 16-lane SIMD pass, replacing the inline-for scalar tail that LLVM was
    // emitting as a serialized chain of 14 scalar converts.
    var top5: topk.Top5 = .{};
    const inv_scale: f32 = idx.inv_scale;
    var n: usize = 0;
    while (n < topkp.size) : (n += 1) {
        const id = topkp.id[n];
        const v_off = @as(usize, id) * fmt.DIM_PADDED;
        const v_slice = idx.vectors[v_off..][0..fmt.DIM_PADDED];
        const v: *align(32) const [fmt.DIM_PADDED]i16 = @ptrCast(@alignCast(v_slice));
        const dx = dist.euclideanSqF32QueryI16Ref(query_f32_padded, v, inv_scale);
        const lbl = labelBit(idx.labels, id);
        top5.maybeInsert(dx, lbl);
    }
    return .{ .fraud_count = top5.fraudCount() };
}

test "search on synthetic 8-vector index returns valid fraud_count" {
    const path = "/tmp/rinha_test_ivf.bin";

    // Build synthetic index: 2 centroids, 8 vectors, labels 0b00001111.
    // Layout mirrors loader.zig's expectations including 64-byte padding
    // before the vectors block.
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

        // Centroid 0: all zeros. Centroid 1: first 14 dims = 1.0, rest 0.
        var cent: [2 * 16]f32 = .{0.0} ** 32;
        inline for (0..fmt.DIM) |j| cent[16 + j] = 1.0;
        try w.writeAll(std.mem.sliceAsBytes(cent[0..]));

        const offsets: [3]u32 = .{ 0, 4, 8 };
        try w.writeAll(std.mem.sliceAsBytes(offsets[0..]));

        // Pad to 64-byte boundary before vectors block.
        const written: usize = @sizeOf(fmt.Header) + @sizeOf(@TypeOf(cent)) + @sizeOf(@TypeOf(offsets));
        const aligned: usize = std.mem.alignForward(usize, written, 64);
        const pad_zero: [64]u8 = .{0} ** 64;
        try w.writeAll(pad_zero[0 .. aligned - written]);

        // Vectors 0..3: all zeros (close to centroid 0).
        // Vectors 4..7: first 14 dims = SCALE (close to centroid 1 after re-rank).
        var vecs: [8 * 16]i16 = .{0} ** 128;
        inline for (4..8) |vi| {
            inline for (0..fmt.DIM) |j| vecs[vi * 16 + j] = @intCast(fmt.SCALE);
        }
        try w.writeAll(std.mem.sliceAsBytes(vecs[0..]));

        // Labels: idx 0..3 fraud=1, idx 4..7 legit=0.
        const labels: [1]u8 = .{0b00001111};
        try w.writeAll(labels[0..]);
    }
    defer std.fs.cwd().deleteFile(path) catch {};

    var idx = try loader.load(path);
    defer loader.unload(&idx);

    var q_padded: [fmt.DIM_PADDED]i16 align(64) = .{0} ** fmt.DIM_PADDED;
    const q_f32: [fmt.DIM_PADDED]f32 align(64) = .{0.0} ** fmt.DIM_PADDED;

    const result = search(&idx, &q_padded, &q_f32);
    // Top-5 = 4 fraud vectors at distance 0 + 1 legit vector. Sum of labels = 4.
    try std.testing.expectEqual(@as(u8, 4), result.fraud_count);
}
