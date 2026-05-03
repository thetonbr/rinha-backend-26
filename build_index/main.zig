//! Build-time orchestrator for the V2 IVF index.
//!
//! Pipeline:
//!   1. parse references.json.gz into f32 vectors + fraud labels
//!   2. train k-means (k = nlist, default 256) on the float vectors
//!   3. quantize each vector to int16 and lay out the per-dim SoA arrays
//!      in invlist-sorted order (so each invlist is contiguous per dim)
//!   4. compute axis-aligned bounding boxes (min/max) per cluster
//!   5. pack the fraud labels into a bitset
//!   6. emit the binary index via writer.write
//!   7. (placeholder) validate exact-vs-approx recall on a query sample
//!
//! Usage: build_index <refs.json.gz> <out/index.bin> [nlist=256]
const std = @import("std");
const parser = @import("parser.zig");
const kmeans = @import("kmeans.zig");
const writer = @import("writer.zig");
const recall_mod = @import("recall.zig");
const fmt = @import("fmt");

const SCALE_F32: f32 = 10000.0;

inline fn quantize(v: f32) i16 {
    const x = v * SCALE_F32;
    const c = @max(-32768.0, @min(32767.0, x));
    return @intFromFloat(@round(c));
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();
    _ = args.skip();
    const refs_path = args.next() orelse return error.MissingArg;
    const out_path = args.next() orelse return error.MissingArg;
    const nlist_str = args.next() orelse "256";
    const nlist: u32 = try std.fmt.parseInt(u32, nlist_str, 10);

    std.debug.print("Reading {s}...\n", .{refs_path});
    var records = try parser.readAll(allocator, refs_path);
    defer records.deinit();

    const n: u32 = @intCast(records.items.len);
    std.debug.print("Loaded {d} records. Building float vectors...\n", .{n});
    const vectors = try allocator.alloc([14]f32, n);
    defer allocator.free(vectors);
    const labels = try allocator.alloc(bool, n);
    defer allocator.free(labels);
    for (records.items, vectors, labels) |r, *v, *l| {
        v.* = r.vector;
        l.* = r.is_fraud;
    }

    // 25 Lloyd iterations: 10 was leaving ~2% of vectors still flipping
    // clusters at exit, which inflates the runtime stage-3 bbox-repair cost
    // because looser centroids produce wider boxes that fail to prune. 25
    // hits the convergence criterion (<0.1% changes) on the real 3M dataset
    // and adds roughly 60-90 s to the offline image build — paid once, never
    // at request time.
    std.debug.print("Training k-means k={d}...\n", .{nlist});
    var km = try kmeans.train(allocator, vectors, nlist, 25, 42);
    defer km.deinit();

    // Pad centroids from 14 → 16 f32 lanes so they can be loaded with a wide
    // ymm op even though the centroid stage only reads 14. The trailing two
    // lanes are zero and contribute zero to the squared-distance kernel.
    const centroids_padded = try allocator.alloc([16]f32, nlist);
    defer allocator.free(centroids_padded);
    for (km.centroids, centroids_padded) |c, *cp| {
        inline for (0..fmt.DIM) |j| cp[j] = c[j];
        cp[14] = 0;
        cp[15] = 0;
    }

    // Histogram + prefix sum to derive invlist offsets.
    std.debug.print("Building invlists...\n", .{});
    const counts = try allocator.alloc(u32, nlist);
    defer allocator.free(counts);
    @memset(counts, 0);
    for (km.assignments) |a| counts[a] += 1;

    const offsets = try allocator.alloc(u32, nlist + 1);
    defer allocator.free(offsets);
    offsets[0] = 0;
    for (0..nlist) |i| offsets[i + 1] = offsets[i] + counts[i];

    // Allocate the SoA per-dim arrays (one per dimension), the orig_ids
    // remap, and the cluster-sorted label buffer. Each dim is its own
    // contiguous slice, n_vectors entries long.
    var dims: [fmt.DIM][]i16 = undefined;
    inline for (0..fmt.DIM) |j| {
        dims[j] = try allocator.alloc(i16, n);
    }
    defer inline for (0..fmt.DIM) |j| allocator.free(dims[j]);
    const orig_ids = try allocator.alloc(u32, n);
    defer allocator.free(orig_ids);
    const sorted_labels = try allocator.alloc(bool, n);
    defer allocator.free(sorted_labels);

    // Bounding boxes per cluster: bbox_min[c][j] and bbox_max[c][j] over j∈[0,14)
    // span the i16 coordinates of all members of cluster c. Initialize with
    // the i16 sentinel bounds so the first member sets the real values.
    const bbox_min = try allocator.alloc([fmt.DIM]i16, nlist);
    defer allocator.free(bbox_min);
    const bbox_max = try allocator.alloc([fmt.DIM]i16, nlist);
    defer allocator.free(bbox_max);
    for (bbox_min, bbox_max) |*lo, *hi| {
        lo.* = .{std.math.maxInt(i16)} ** fmt.DIM;
        hi.* = .{std.math.minInt(i16)} ** fmt.DIM;
    }

    const cursor = try allocator.alloc(u32, nlist);
    defer allocator.free(cursor);
    @memset(cursor, 0);

    // Single pass: quantize each vector, scatter its 14 dims into the SoA
    // arrays at its destination invlist slot, capture orig_id, and update the
    // cluster's bbox. We avoid a second sort/shuffle step entirely.
    for (vectors, km.assignments, labels, 0..) |v, a, l, src_idx| {
        const dest: u32 = offsets[a] + cursor[a];
        cursor[a] += 1;

        inline for (0..fmt.DIM) |j| {
            const q = quantize(v[j]);
            dims[j][dest] = q;
            if (q < bbox_min[a][j]) bbox_min[a][j] = q;
            if (q > bbox_max[a][j]) bbox_max[a][j] = q;
        }
        orig_ids[dest] = @intCast(src_idx);
        sorted_labels[dest] = l;
    }

    // Repair any cluster whose bbox stayed at the init sentinels (empty
    // cluster, possible for k-means with k > unique points). Collapse to a
    // zero-volume box at the origin so the runtime lower-bound stays sane.
    for (bbox_min, bbox_max, 0..) |*lo, *hi, ci| {
        if (counts[ci] == 0) {
            lo.* = .{0} ** fmt.DIM;
            hi.* = .{0} ** fmt.DIM;
        }
    }

    // Pack labels into an LSB-first bitset.
    const bits_len: usize = (n + 7) / 8;
    const bits = try allocator.alloc(u8, bits_len);
    defer allocator.free(bits);
    @memset(bits, 0);
    for (sorted_labels, 0..) |l, i| {
        if (l) bits[i >> 3] |= @as(u8, 1) << @intCast(i & 7);
    }

    std.debug.print("Writing {s}...\n", .{out_path});
    try writer.write(.{
        .out_path = out_path,
        .n_vectors = n,
        .nlist = nlist,
        .centroids_padded = centroids_padded,
        .invlist_offsets = offsets,
        .bbox_min = bbox_min,
        .bbox_max = bbox_max,
        .orig_ids = orig_ids,
        .dims = dims,
        .fraud_bits = bits,
    });

    // Single-cap mirror of runtime (src/index/ivf.zig MAX_CLUSTERS_VISITED).
    const RUNTIME_CAP: u32 = 16;
    const r = try recall_mod.validateExactVsApprox(
        allocator, vectors, labels, km.centroids, km.assignments,
        nlist, 2000, 0xdeadbeef, RUNTIME_CAP,
    );
    std.debug.print(
        "[nlist={d} cap={d}] Recall@5: {d:.4} | fraud match: {d:.4} | approval flip: {d:.4}\n" ++
            "                   clusters/query: avg={d:.2} p50={d} p99={d} p999={d} max={d}\n",
        .{
            nlist,                   RUNTIME_CAP,           r.recall_at_5,
            r.fraud_count_match_rate, r.approval_flip_rate, r.avg_clusters_visited,
            r.p50_clusters_visited,  r.p99_clusters_visited, r.p999_clusters_visited,
            r.max_clusters_visited,
        },
    );

    std.debug.print("Done.\n", .{});
}
