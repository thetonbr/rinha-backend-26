const std = @import("std");
const fmt = @import("format.zig");
const loader = @import("loader.zig");

// V2 search algorithm:
//
//   Stage 1 — Pick top-NPROBE centroids by f32 squared distance.
//   Stage 2 — Scan probed invlists with per-dimension early-exit (most
//             candidates are discarded after 2-3 dims when worst_d is tight).
//   Stage 3 — Bounding-box repair: visit any non-probed cluster whose bbox
//             lower bound is <= worst_d. Preserves recall at NPROBE=1.
//
// Top-5 is materialized inline with deterministic tie-break by orig_id.
pub const NPROBE: usize = 1;
pub const K: usize = 5;

// Used to cap stage 3's "already scanned" bitmap. 256 matches DEFAULT_NLIST
// and is the only configuration we ship; ivf.search asserts on overflow at
// runtime if a smaller nlist is loaded.
pub const MAX_NLIST: usize = 256;

pub const SearchResult = struct {
    fraud_count: u8,
};

// Order chosen to maximize early-exit pruning: dim5/dim6 hold sentinel -SCALE
// when last_transaction is null and otherwise carry the largest discriminative
// signal (recency / distance from last transaction). dim2 (amount-vs-avg
// ratio) and dim0 (raw amount) follow, then card-present / online flags, and
// finally the low-variance dims (day_of_week, MCC risk).
const SCAN_ORDER = [_]u8{ 5, 6, 2, 0, 7, 8, 11, 12, 9, 10, 1, 13, 3, 4 };

const Top5 = struct {
    dist: [K]u64 = .{std.math.maxInt(u64)} ** K,
    label: [K]u8 = .{0} ** K,
    id: [K]u32 = .{std.math.maxInt(u32)} ** K,
    worst: usize = 0,
    worst_d: u64 = std.math.maxInt(u64),
    worst_id: u32 = std.math.maxInt(u32),

    // Total order: smaller distance wins, with ties broken by smaller orig_id
    // for deterministic output (0 false positives / negatives across runs).
    inline fn isBetter(da: u64, ia: u32, db: u64, ib: u32) bool {
        return da < db or (da == db and ia < ib);
    }

    inline fn isWorse(da: u64, ia: u32, db: u64, ib: u32) bool {
        return da > db or (da == db and ia > ib);
    }

    inline fn worstIndex(self: *const Top5) usize {
        var w: usize = 0;
        inline for (1..K) |i| {
            if (isWorse(self.dist[i], self.id[i], self.dist[w], self.id[w])) w = i;
        }
        return w;
    }

    inline fn tryInsert(self: *Top5, d: u64, lbl: u8, id: u32) void {
        if (!isBetter(d, id, self.worst_d, self.worst_id)) return;
        self.dist[self.worst] = d;
        self.label[self.worst] = lbl;
        self.id[self.worst] = id;
        const w = self.worstIndex();
        self.worst = w;
        self.worst_d = self.dist[w];
        self.worst_id = self.id[w];
    }

    fn fraudCount(self: Top5) u8 {
        var c: u8 = 0;
        inline for (0..K) |i| c += self.label[i];
        return c;
    }
};

inline fn labelBit(labels: []const u8, idx: u32) u8 {
    return @intCast((labels[idx >> 3] >> @intCast(idx & 7)) & 1);
}

inline fn centroidSqDistF32(q: *const [fmt.DIM]f32, cent: *const [16]f32) f32 {
    var s: f32 = 0;
    inline for (0..fmt.DIM) |j| {
        const d = q[j] - cent[j];
        s += d * d;
    }
    return s;
}

// Squared lower-bound of the cluster's axis-aligned bounding box from `q_int`.
// If the query is inside the box on a dim, contribution is 0; otherwise it is
// the squared 1-D distance to the nearest box face.
inline fn bboxLowerBound(
    q_int: *const [fmt.DIM]i16,
    bmin: *const [fmt.DIM]i16,
    bmax: *const [fmt.DIM]i16,
) u64 {
    var s: u64 = 0;
    inline for (0..fmt.DIM) |j| {
        const qj: i32 = q_int[j];
        const lo: i32 = bmin[j];
        const hi: i32 = bmax[j];
        var d: i32 = 0;
        if (qj < lo) d = lo - qj else if (qj > hi) d = qj - hi;
        const dl: i64 = d;
        s += @as(u64, @intCast(dl * dl));
    }
    return s;
}

fn scanInvlist(
    idx: *const loader.Index,
    cluster_id: u32,
    q: *const [fmt.DIM]i16,
    top: *Top5,
) void {
    const start = idx.invlist_offsets[cluster_id];
    const end = idx.invlist_offsets[cluster_id + 1];
    if (end <= start) return;

    var i: u32 = start;
    while (i < end) : (i += 1) {
        var dist_acc: u64 = 0;
        // Scalar per-dim accumulation with early-exit. We use a runtime `for`
        // (not `inline for`) because Zig 0.13 forbids `break` inside an
        // `inline for`, and the early-exit is the whole point. SCAN_ORDER is
        // a comptime constant so the loop body still indexes via a known
        // sequence — branch predictors handle the fixed-order traversal well.
        var skip = false;
        for (SCAN_ORDER) |dim_idx| {
            const qj: i32 = q[dim_idx];
            const vj: i32 = idx.dims[dim_idx][i];
            const d: i64 = @as(i64, qj) - @as(i64, vj);
            dist_acc += @as(u64, @intCast(d * d));
            if (dist_acc > top.worst_d) {
                skip = true;
                break;
            }
        }
        if (skip) continue;
        const oid = idx.orig_ids[i];
        top.tryInsert(dist_acc, labelBit(idx.labels, i), oid);
    }
}

pub fn search(
    idx: *const loader.Index,
    q_int: *const [fmt.DIM]i16,
    q_f32: *const [fmt.DIM]f32,
) SearchResult {
    const nlist: u32 = idx.header.nlist;
    std.debug.assert(nlist <= MAX_NLIST);

    // Stage 1: select NPROBE closest centroids in f32. NPROBE=1 in production,
    // so this is just an argmin; we keep the small insertion-sort for clarity
    // and to make NPROBE configurable.
    var probe_id: [NPROBE]i32 = .{-1} ** NPROBE;
    var probe_dist: [NPROBE]f32 = .{std.math.inf(f32)} ** NPROBE;

    var c: u32 = 0;
    while (c < nlist) : (c += 1) {
        const cent_off: usize = @as(usize, c) * 16;
        const cent_full: *const [16]f32 = idx.centroids[cent_off..][0..16];
        const d = centroidSqDistF32(q_f32, cent_full);
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

    // Stage 2: scan probed clusters with per-dim early-exit feeding Top5.
    var top: Top5 = .{};
    var scanned: [MAX_NLIST]bool = .{false} ** MAX_NLIST;
    inline for (0..NPROBE) |k| {
        const cid_signed = probe_id[k];
        if (cid_signed >= 0) {
            const cid: u32 = @intCast(cid_signed);
            scanned[cid] = true;
            scanInvlist(idx, cid, q_int, &top);
        }
    }

    // Stage 3: bounding-box repair. For each non-probed cluster, compute the
    // squared lower-bound of its bbox vs the query in i16. If lb > worst_d we
    // can prove no member of that cluster can enter the top-5 and skip it.
    var ci: u32 = 0;
    while (ci < nlist) : (ci += 1) {
        if (scanned[ci]) continue;
        const start = idx.invlist_offsets[ci];
        const end = idx.invlist_offsets[ci + 1];
        if (end <= start) continue;
        const bb_off: usize = @as(usize, ci) * fmt.DIM;
        const bmin: *const [fmt.DIM]i16 = idx.bbox_min[bb_off..][0..fmt.DIM];
        const bmax: *const [fmt.DIM]i16 = idx.bbox_max[bb_off..][0..fmt.DIM];
        const lb = bboxLowerBound(q_int, bmin, bmax);
        if (lb <= top.worst_d) {
            scanInvlist(idx, ci, q_int, &top);
        }
    }

    return .{ .fraud_count = top.fraudCount() };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "Top5 deterministic insertion with tie-break" {
    var t: Top5 = .{};
    t.tryInsert(10, 1, 100);
    t.tryInsert(5, 0, 200);
    t.tryInsert(10, 1, 50); // ties dist=10; orig_id=50 beats 100
    t.tryInsert(1, 1, 300);
    t.tryInsert(7, 0, 400);
    t.tryInsert(7, 1, 350); // ties dist=7; orig_id=350 beats 400
    // Best 5 by (dist, id) lex order: (1,300), (5,200), (7,350), (7,400), (10,50)
    // Sum of labels: 1 + 0 + 1 + 0 + 1 = 3
    try std.testing.expectEqual(@as(u8, 3), t.fraudCount());
}

test "Top5 rejects worse than worst" {
    var t: Top5 = .{};
    inline for (0..K) |i| t.tryInsert(@as(u64, i + 1), 0, @as(u32, @intCast(i)));
    t.tryInsert(99, 1, 999);
    try std.testing.expectEqual(@as(u8, 0), t.fraudCount());
}

test "search on synthetic V2 index returns valid fraud_count" {
    const path = "/tmp/rinha_test_ivf_v2.bin";
    defer std.fs.cwd().deleteFile(path) catch {};

    const n_vectors: u32 = 8;
    const nlist: u32 = 2;

    // Centroid 0 at origin; centroid 1 with first 14 dims = 1.0.
    var centroids: [2][16]f32 = .{ .{0.0} ** 16, .{0.0} ** 16 };
    inline for (0..fmt.DIM) |j| centroids[1][j] = 1.0;

    const offsets: [3]u32 = .{ 0, 4, 8 };
    // Cluster 0 holds the all-zero vectors; cluster 1 holds the SCALE-valued
    // vectors. bbox for each is degenerate (min == max), so lb is exact.
    const scale_i16: i16 = @intCast(fmt.SCALE);
    const bbox_min: [2][fmt.DIM]i16 = .{ .{0} ** fmt.DIM, .{scale_i16} ** fmt.DIM };
    const bbox_max: [2][fmt.DIM]i16 = .{ .{0} ** fmt.DIM, .{scale_i16} ** fmt.DIM };
    // orig_ids: identity in invlist order.
    const orig_ids: [8]u32 = .{ 0, 1, 2, 3, 4, 5, 6, 7 };

    // Per-dim arrays. Cluster 0 (i=0..3) has all dims = 0. Cluster 1 (i=4..7)
    // has all dims = SCALE.
    var dim_storage: [fmt.DIM][8]i16 = undefined;
    inline for (0..fmt.DIM) |j| {
        inline for (0..4) |i| dim_storage[j][i] = 0;
        inline for (4..8) |i| dim_storage[j][i] = @intCast(fmt.SCALE);
    }
    var dim_init: [fmt.DIM][]const i16 = undefined;
    inline for (0..fmt.DIM) |j| dim_init[j] = dim_storage[j][0..];

    // Labels: idx 0..3 fraud=1, idx 4..7 legit=0.
    const labels: [1]u8 = .{0b00001111};

    try loader.writeSyntheticV2(
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

    var idx = try loader.load(path);
    defer loader.unload(&idx);

    // Query at the origin — should pick cluster 0 (all 4 fraud) and the bbox
    // repair should also visit cluster 1 (lb = 14 * SCALE^2 > worst_d=0 once
    // top-5 is full of zero-distance hits, so cluster 1 is pruned). Top-5
    // therefore contains all 4 fraud vectors plus one slot still at sentinel.
    const q_int: [fmt.DIM]i16 = .{0} ** fmt.DIM;
    const q_f32: [fmt.DIM]f32 = .{0.0} ** fmt.DIM;
    const result = search(&idx, &q_int, &q_f32);
    try std.testing.expectEqual(@as(u8, 4), result.fraud_count);
}
