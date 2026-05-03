//! Build-time recall validation harness.
//!
//! Runs the full IVF approximate search (stage 1 argmin + stage 2 invlist
//! scan + stage 3 bbox-repair) in pure f32 against an exhaustive top-5
//! brute-force scan over the same dataset, on a sampled set of queries.
//! Reports four metrics that map onto the Rinha 2026 score formula:
//!
//!   - `recall_at_5`            — ⟨ |approx ∩ exact| / 5 ⟩ over queries.
//!   - `fraud_count_match_rate` — fraction with identical fraud_count in 0..5.
//!   - `approval_flip_rate`     — fraction whose `approved` boolean flipped
//!                                between exact and approx (this drives FN/FP
//!                                in the contest score directly).
//!   - `avg_clusters_visited`   — mean cost proxy: clusters scanned per query.
//!
//! Querying with vectors drawn from the dataset itself produces a self-match
//! per query (distance 0). Both exact and approx see that, so the bias cancels
//! when comparing the two — the metric measures how often the approximate
//! algorithm picks the *same other 4* neighbours as the exhaustive one.

const std = @import("std");

const DIM: usize = 14;
const K: usize = 5;

pub const Result = struct {
    n_queries: u32,
    recall_at_5: f32,
    fraud_count_match_rate: f32,
    approval_flip_rate: f32,
    avg_clusters_visited: f32,
};

inline fn distSq(a: [DIM]f32, b: [DIM]f32) f32 {
    var s: f32 = 0;
    inline for (0..DIM) |j| {
        const d = a[j] - b[j];
        s += d * d;
    }
    return s;
}

inline fn bboxLowerBound(q: [DIM]f32, lo: [DIM]f32, hi: [DIM]f32) f32 {
    var s: f32 = 0;
    inline for (0..DIM) |j| {
        var d: f32 = 0;
        if (q[j] < lo[j]) d = lo[j] - q[j] else if (q[j] > hi[j]) d = q[j] - hi[j];
        s += d * d;
    }
    return s;
}

const Top5 = struct {
    dists: [K]f32 = .{std.math.inf(f32)} ** K,
    ids: [K]u32 = .{std.math.maxInt(u32)} ** K,

    inline fn worstIdx(self: *const Top5) usize {
        var w: usize = 0;
        inline for (1..K) |i| {
            // Tie-break by orig_id (smaller wins, so larger is worse) to mirror
            // the runtime Top5 in src/index/ivf.zig.
            if (self.dists[i] > self.dists[w] or
                (self.dists[i] == self.dists[w] and self.ids[i] > self.ids[w]))
            {
                w = i;
            }
        }
        return w;
    }

    inline fn tryInsert(self: *Top5, d: f32, id: u32) void {
        const w = self.worstIdx();
        if (d < self.dists[w] or (d == self.dists[w] and id < self.ids[w])) {
            self.dists[w] = d;
            self.ids[w] = id;
        }
    }

    fn worstDist(self: *const Top5) f32 {
        var m: f32 = self.dists[0];
        inline for (1..K) |i| {
            if (self.dists[i] > m) m = self.dists[i];
        }
        return m;
    }

    fn fraudCount(self: *const Top5, is_fraud: []const bool) u8 {
        var c: u8 = 0;
        for (self.ids) |id| {
            if (id == std.math.maxInt(u32)) continue;
            if (is_fraud[id]) c += 1;
        }
        return c;
    }
};

fn exactTopK(q: [DIM]f32, vectors: [][DIM]f32) Top5 {
    var top: Top5 = .{};
    for (vectors, 0..) |v, i| {
        const d = distSq(q, v);
        const id: u32 = @intCast(i);
        if (d < top.worstDist() or
            (d == top.worstDist() and id < top.ids[top.worstIdx()]))
        {
            top.tryInsert(d, id);
        }
    }
    return top;
}

fn scanInvlist(
    q: [DIM]f32,
    vectors: [][DIM]f32,
    inv_to_orig: []const u32,
    start: u32,
    end: u32,
    top: *Top5,
) void {
    var i: u32 = start;
    while (i < end) : (i += 1) {
        const orig: u32 = inv_to_orig[i];
        const d = distSq(q, vectors[orig]);
        top.tryInsert(d, orig);
    }
}

const ApproxOut = struct {
    top: Top5,
    clusters_visited: u32,
};

fn approxTopK(
    q: [DIM]f32,
    vectors: [][DIM]f32,
    centroids: [][DIM]f32,
    invlist_offsets: []const u32,
    inv_to_orig: []const u32,
    bbox_lo: []const [DIM]f32,
    bbox_hi: []const [DIM]f32,
    nlist: u32,
) ApproxOut {
    // Stage 1: argmin centroid.
    var best_id: u32 = 0;
    var best_d: f32 = std.math.inf(f32);
    var c: u32 = 0;
    while (c < nlist) : (c += 1) {
        const d = distSq(q, centroids[c]);
        if (d < best_d) {
            best_d = d;
            best_id = c;
        }
    }

    // Stage 2: scan probed cluster.
    var top: Top5 = .{};
    var visited: u32 = 0;
    {
        const start = invlist_offsets[best_id];
        const end = invlist_offsets[best_id + 1];
        if (end > start) {
            scanInvlist(q, vectors, inv_to_orig, start, end, &top);
            visited += 1;
        }
    }

    // Stage 3: bbox repair on the remaining nlist - 1 clusters.
    var ci: u32 = 0;
    while (ci < nlist) : (ci += 1) {
        if (ci == best_id) continue;
        const start = invlist_offsets[ci];
        const end = invlist_offsets[ci + 1];
        if (end <= start) continue;
        const lb = bboxLowerBound(q, bbox_lo[ci], bbox_hi[ci]);
        if (lb <= top.worstDist()) {
            scanInvlist(q, vectors, inv_to_orig, start, end, &top);
            visited += 1;
        }
    }

    return .{ .top = top, .clusters_visited = visited };
}

pub fn validateExactVsApprox(
    allocator: std.mem.Allocator,
    vectors: [][DIM]f32,
    is_fraud: []const bool,
    centroids: [][DIM]f32,
    assignments: []const u32,
    nlist: u32,
    n_queries: usize,
    seed: u64,
) !Result {
    std.debug.assert(vectors.len == is_fraud.len);
    std.debug.assert(vectors.len == assignments.len);
    std.debug.assert(centroids.len == nlist);

    // Build inverted-list offsets and the invlist→original index mapping.
    const counts = try allocator.alloc(u32, nlist);
    defer allocator.free(counts);
    @memset(counts, 0);
    for (assignments) |a| counts[a] += 1;

    const offsets = try allocator.alloc(u32, nlist + 1);
    defer allocator.free(offsets);
    offsets[0] = 0;
    for (0..nlist) |c| offsets[c + 1] = offsets[c] + counts[c];

    const cursor = try allocator.alloc(u32, nlist);
    defer allocator.free(cursor);
    @memset(cursor, 0);

    const inv_to_orig = try allocator.alloc(u32, vectors.len);
    defer allocator.free(inv_to_orig);
    for (assignments, 0..) |a, i| {
        const dest = offsets[a] + cursor[a];
        cursor[a] += 1;
        inv_to_orig[dest] = @intCast(i);
    }

    // Per-cluster axis-aligned bounding box in f32. Empty clusters collapse to
    // the origin to match main.zig:131-138 behavior so bbox lower-bound is
    // sane (never NaN/inf when feeding stage 3).
    const bbox_lo = try allocator.alloc([DIM]f32, nlist);
    defer allocator.free(bbox_lo);
    const bbox_hi = try allocator.alloc([DIM]f32, nlist);
    defer allocator.free(bbox_hi);
    for (bbox_lo, bbox_hi) |*lo, *hi| {
        lo.* = .{std.math.inf(f32)} ** DIM;
        hi.* = .{-std.math.inf(f32)} ** DIM;
    }
    for (vectors, assignments) |v, a| {
        inline for (0..DIM) |j| {
            if (v[j] < bbox_lo[a][j]) bbox_lo[a][j] = v[j];
            if (v[j] > bbox_hi[a][j]) bbox_hi[a][j] = v[j];
        }
    }
    for (counts, bbox_lo, bbox_hi) |cnt, *lo, *hi| {
        if (cnt == 0) {
            lo.* = .{0} ** DIM;
            hi.* = .{0} ** DIM;
        }
    }

    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();

    var sum_overlap: u32 = 0;
    var fraud_match: u32 = 0;
    var approval_flip: u32 = 0;
    var clusters_visited_sum: u64 = 0;

    var qi: usize = 0;
    while (qi < n_queries) : (qi += 1) {
        const q_idx = r.uintLessThan(usize, vectors.len);
        const q = vectors[q_idx];

        const exact = exactTopK(q, vectors);
        const approx = approxTopK(
            q,
            vectors,
            centroids,
            offsets,
            inv_to_orig,
            bbox_lo,
            bbox_hi,
            nlist,
        );

        // Top-5 overlap (orig_id sets).
        var ov: u32 = 0;
        for (exact.ids) |eid| {
            if (eid == std.math.maxInt(u32)) continue;
            for (approx.top.ids) |aid| {
                if (eid == aid) {
                    ov += 1;
                    break;
                }
            }
        }
        sum_overlap += ov;

        const exact_fc = exact.fraudCount(is_fraud);
        const approx_fc = approx.top.fraudCount(is_fraud);
        if (exact_fc == approx_fc) fraud_match += 1;
        if ((exact_fc < 3) != (approx_fc < 3)) approval_flip += 1;
        clusters_visited_sum += approx.clusters_visited;
    }

    const n_f: f32 = @floatFromInt(n_queries);
    return .{
        .n_queries = @intCast(n_queries),
        .recall_at_5 = @as(f32, @floatFromInt(sum_overlap)) / (n_f * @as(f32, K)),
        .fraud_count_match_rate = @as(f32, @floatFromInt(fraud_match)) / n_f,
        .approval_flip_rate = @as(f32, @floatFromInt(approval_flip)) / n_f,
        .avg_clusters_visited = @as(f32, @floatFromInt(clusters_visited_sum)) / n_f,
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "Top5 inserts and tie-breaks by orig_id" {
    var t: Top5 = .{};
    t.tryInsert(10.0, 100);
    t.tryInsert(5.0, 200);
    t.tryInsert(10.0, 50); // ties dist=10, smaller id wins
    t.tryInsert(1.0, 300);
    t.tryInsert(7.0, 400);
    t.tryInsert(7.0, 350); // ties dist=7, smaller id wins → kicks out 400
    // Top-5 (sorted): (1,300) (5,200) (7,350) (7,400) (10,50)
    var ids_sorted = t.ids;
    std.mem.sort(u32, &ids_sorted, {}, std.sort.asc(u32));
    try std.testing.expectEqualSlices(u32, &.{ 50, 200, 300, 350, 400 }, &ids_sorted);
}

test "validateExactVsApprox identifies a perfectly clusterable dataset" {
    const allocator = std.testing.allocator;

    // 200 vectors split exactly across 4 clusters at the corners of a 14-cube.
    const N: u32 = 200;
    const nlist: u32 = 4;
    const per: u32 = N / nlist;

    const vecs = try allocator.alloc([DIM]f32, N);
    defer allocator.free(vecs);
    const labels = try allocator.alloc(bool, N);
    defer allocator.free(labels);
    const assignments = try allocator.alloc(u32, N);
    defer allocator.free(assignments);
    const centroids = try allocator.alloc([DIM]f32, nlist);
    defer allocator.free(centroids);

    // Centroid k has dim0 = k, all other dims = 0; vectors near each centroid
    // get a tiny perturbation so each query has a unique top-5.
    var prng = std.Random.DefaultPrng.init(7);
    const r = prng.random();
    for (centroids, 0..) |*c, k| {
        c.* = .{0} ** DIM;
        c.*[0] = @floatFromInt(k);
    }
    for (vecs, labels, assignments, 0..) |*v, *l, *a, i| {
        const k: u32 = @intCast(i / per);
        v.* = centroids[k];
        // perturbation in dim 1 only, well inside the inter-cluster gap.
        v.*[1] = (r.float(f32) - 0.5) * 0.01;
        l.* = (i % 2) == 0;
        a.* = k;
    }

    const res = try validateExactVsApprox(
        allocator,
        vecs,
        labels,
        centroids,
        assignments,
        nlist,
        100,
        12345,
    );

    // With perfect cluster separation, the approx pick must agree with brute
    // force on every metric; bbox-repair scans only the chosen cluster.
    try std.testing.expect(res.recall_at_5 >= 0.99);
    try std.testing.expect(res.fraud_count_match_rate >= 0.99);
    try std.testing.expectEqual(@as(f32, 0.0), res.approval_flip_rate);
    try std.testing.expect(res.avg_clusters_visited <= 1.5);
}
