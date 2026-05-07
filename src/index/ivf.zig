const std = @import("std");
const fmt = @import("format.zig");
const loader = @import("loader.zig");

// V2 search algorithm with adaptive nprobe:
//
//   Stage 1 — Pick top-NPROBE centroids by f32 squared distance.
//   Stage 2 — Scan probed invlists with per-dimension early-exit.
//   Stage 3a — Fast bbox-repair: scan up to FAST_CLUSTERS clusters whose bbox
//              lower bound is <= worst_d. Stops early when the bound prunes
//              everything left.
//   Stage 3b — Adaptive escalation: if Stage 3a leaves fraud_count in the
//              ambiguous zone (∈ {2, 3}, i.e. on the approval boundary at
//              fraud_score = 0.4 / 0.6), continue bbox-repair up to
//              FULL_CLUSTERS. Easy queries (count ∈ {0, 1, 4, 5}) pay only
//              the fast budget — ~5× cheaper than a fixed cap of FULL_CLUSTERS
//              while preserving recall where it matters for the score.
//
// Top-5 is materialized inline with deterministic tie-break by orig_id.
//
// NPROBE=2 was tested on the judge in v21 (issue #929) and produced no
// measurable change in p99 vs NPROBE=1 (2.24ms vs 2.23ms — within noise).
// Reverting to NPROBE=1 keeps the stage-1 batched-argmin specialization
// on the hot path.
pub const NPROBE: usize = 1;
pub const K: usize = 5;

// Bitmap size for stage-3 "already scanned" tracking and stack allocation
// for the centroid-scan pre-allocations. Sized for nlist=4096; ivf.search
// asserts on overflow if a smaller index is loaded. v26 bumps the index to
// nlist=4096 (Dockerfile ARG NLIST=4096) so each invlist holds ~750 vectors
// and each scanInvlist call costs ~7 µs (down from ~28 µs at nlist=1024).
pub const MAX_NLIST: usize = 4096;

// Two-tier cluster budget. FAST_CLUSTERS bounds the easy-case latency;
// FULL_CLUSTERS bounds the ambiguous-case latency. Both include the stage-2
// probed cluster (NPROBE counts as 1 toward `visited`).
//
//   FAST_CLUSTERS  — cluster cap when fraud_count ∉ {2, 3}. The dominant
//                    case (~95% of queries on the reference dataset).
//   FULL_CLUSTERS  — cluster cap when fraud_count ∈ {2, 3}. The ambiguous
//                    boundary zone where one extra fraud or one extra legit
//                    flips approval — exactly where additional recall pays
//                    off in the contest score.
//
// Per-cluster scan cost at nlist=4096: ~7 µs. Worst-case latency budget:
//   easy:      ~56 µs total scan (8 clusters × 7 µs)
//   ambiguous: ~168 µs total scan (24 clusters × 7 µs)
//
// Compare against the v24 baseline (nlist=1024, fixed cap=12, ~28 µs/cluster):
//   v24 always: ~336 µs (one budget for every query)
//
// History of fixed-cap experiments (kept here so we don't relitigate):
//   v19 nlist=256  cap=8  → final 5652 (p99 2.23 ms, det 3000).
//   v20 nlist=256  cap=4  → final 5191 (FP=11 FN=11, det 2504).
//   v23 nlist=512  cap=12 → final 5722 (p99 1.90 ms).
//   v24 nlist=1024 cap=12 → final 5778 (p99 ~1.7 ms).
//   v25 nlist=2048 cap=16 → final 5505 (FP=2 FN=2, det 2914).
// v25's regression killed any plan to scale nlist further with a fixed cap.
// Adaptive nprobe sidesteps that: the ambiguous-zone trigger explicitly
// re-scans on uncertainty rather than betting on offline margin.
pub const FAST_CLUSTERS: u32 = 8;
pub const FULL_CLUSTERS: u32 = 24;

pub const SearchResult = struct {
    fraud_count: u8,
};

// Order chosen to maximize early-exit pruning: dim5/dim6 hold sentinel -SCALE
// when last_transaction is null and otherwise carry the largest discriminative
// signal (recency / distance from last transaction). dim2 (amount-vs-avg
// ratio) and dim0 (raw amount) follow, then card-present / online flags, and
// finally the low-variance dims (day_of_week, MCC risk).
//
// The first 8 entries are deliberately also the 8 most discriminative dims;
// the wide AVX2 scan computes them first and uses the running minimum across
// the 8-vector batch as a lower bound to prune the remaining 6 dims.
const SCAN_ORDER = [_]u8{ 5, 6, 2, 0, 7, 8, 11, 12, 9, 10, 1, 13, 3, 4 };
const SCAN_PREFIX: usize = 8;

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

// Centroids ship as 16-lane padded f32 (lanes 14-15 are zero in the index
// layout). Loading both operands as a 16-wide vector lets one ymm pair (vsubps
// + vmulps + vaddps) cover the whole distance, vs the 14-iteration unrolled
// scalar form. The query is padded to [16]f32 by the caller (lanes 14-15 = 0)
// so the extra lanes contribute zero to the squared distance.
const Vf16 = @Vector(16, f32);
inline fn centroidSqDistF32(q_padded: *const [16]f32, cent: *const [16]f32) f32 {
    const qv: Vf16 = q_padded.*;
    const cv: Vf16 = cent.*;
    const d = qv - cv;
    return @reduce(.Add, d * d);
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

// AVX2 wide-batch lane width. With i16 inputs, diff*diff fits in i32 (2*SCALE
// max, squared = 4e8 << 2^31), and 14 dims of squared i32 (each <= 4e8) sum
// to <= 5.6e9 — overflows i32 but fits i64 comfortably. We accumulate per-dim
// in i32 and widen once before adding to the i64 running sum, which keeps the
// inner multiply in fast 8-wide ymm lanes.
const VEC_LANES: u32 = 8;
const Vec8i16 = @Vector(VEC_LANES, i16);
const Vec8i32 = @Vector(VEC_LANES, i32);
const Vec8i64 = @Vector(VEC_LANES, i64);

// Lookahead distance in vectors for the per-dim @prefetch hint inside the wide
// scan. 64 vectors = 128 bytes per dim ≈ 2 cache lines, ≈ 8 wide-batches ahead.
// Sized empirically to cover ~500-700ns of memory latency: enough to mask DRAM
// fetch when an invlist exceeds L2, small enough not to evict the live working
// set. Too large evicts the in-flight cache lines; too small leaves the load
// stalled. The HW prefetcher tracks ~16 sequential streams; with 14 dims plus
// orig_ids/labels we are at the edge, so explicit hints close the gap.
const PREFETCH_AHEAD: u32 = 64;

inline fn loadDimVec(dj: []align(64) const i16, i: u32) Vec8i16 {
    // The slice arrives 64-byte aligned and `i` is always a multiple of 8 in
    // the wide path, so each chunk lands at a 16-byte boundary. AVX2 ymm
    // unaligned loads on 16-byte-aligned data are zero-penalty on Zen+.
    return dj[i..][0..VEC_LANES].*;
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

    // Wide AVX2 path: 8 candidates per iteration, two-phase early-exit.
    //
    // Phase 1: process the 8 most-discriminative dims (SCAN_ORDER prefix). The
    // partial squared distance after these 8 is a *lower bound* on the final
    // 14-dim distance, since each of the remaining 6 dims contributes a non-
    // negative squared delta. If the minimum across the 8-lane batch is already
    // >= worst_d, no member of this batch can enter the top-5 and we skip the
    // remaining 6 dims entirely.
    //
    // Phase 2: only when the lower bound clears, finish the remaining 6 dims
    // and run the per-lane scatter-insert against top.
    const wide_end: u32 = start + ((end - start) / VEC_LANES) * VEC_LANES;
    while (i < wide_end) : (i += VEC_LANES) {
        var acc: Vec8i64 = @splat(0);
        inline for (0..SCAN_PREFIX) |k| {
            const dim_idx = comptime SCAN_ORDER[k];
            const dj = idx.dims[dim_idx];
            // Hint the line for this dim 64 vectors ahead. Pointer arithmetic
            // bypasses bounds check; @prefetch on an invalid mapping is a no-op
            // on x86 (it cannot fault), so reading past `end` is harmless.
            @prefetch(dj.ptr + i + PREFETCH_AHEAD, .{ .rw = .read, .locality = 1, .cache = .data });
            const raw: Vec8i16 = loadDimVec(dj, i);
            const v32: Vec8i32 = raw; // sign-extend i16 -> i32
            const q_splat: Vec8i32 = @splat(@as(i32, q[dim_idx]));
            const diff = v32 - q_splat;
            const sq = diff * diff;
            acc += @as(Vec8i64, sq);
        }
        // acc[lane] >= 0 since each contribution is a square; cast is safe.
        const min_partial: u64 = @intCast(@reduce(.Min, acc));
        if (min_partial >= top.worst_d) continue;

        inline for (SCAN_PREFIX..fmt.DIM) |k| {
            const dim_idx = comptime SCAN_ORDER[k];
            const dj = idx.dims[dim_idx];
            // Phase 2 reaches dims SCAN_ORDER[8..14] only after the early-exit
            // misses, which means worst_d is loose and we will likely process
            // the next batch as well. Prefetch ahead so the cold dims do not
            // stall when phase 1 promotes them to the hot path next round.
            @prefetch(dj.ptr + i + PREFETCH_AHEAD, .{ .rw = .read, .locality = 1, .cache = .data });
            const raw: Vec8i16 = loadDimVec(dj, i);
            const v32: Vec8i32 = raw;
            const q_splat: Vec8i32 = @splat(@as(i32, q[dim_idx]));
            const diff = v32 - q_splat;
            const sq = diff * diff;
            acc += @as(Vec8i64, sq);
        }
        // Scatter-insert: each lane independently considers itself against
        // the current top-5 worst. The worst tightens as we go, so we serve
        // the lane sequence in invlist order.
        inline for (0..VEC_LANES) |lane| {
            const dist: u64 = @intCast(acc[lane]);
            const idx_i: u32 = i + @as(u32, lane);
            const oid = idx.orig_ids[idx_i];
            if (dist < top.worst_d or (dist == top.worst_d and oid < top.worst_id)) {
                top.tryInsert(dist, labelBit(idx.labels, idx_i), oid);
            }
        }
    }

    // Scalar tail with per-dim early-exit. Same loop body as before; only
    // the bounds change (`i` already advanced by the wide block).
    while (i < end) : (i += 1) {
        var dist_acc: u64 = 0;
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
    q_f32_padded: *const [16]f32,
) SearchResult {
    const nlist: u32 = idx.header.nlist;
    std.debug.assert(nlist <= MAX_NLIST);

    // Stage 1: select NPROBE closest centroids in f32. NPROBE=1 in production
    // so we specialise on argmin and let four 16-lane f32 distances run in
    // parallel — the LLVM scheduler will fuse them onto independent ymm
    // dependency chains, hiding the @reduce(.Add, d*d) cross-lane reduction
    // latency. Falls back to a serial insertion-sort for NPROBE > 1 so we
    // keep the code path flexible when re-evaluating recall.
    var probe_id: [NPROBE]i32 = .{-1} ** NPROBE;
    var probe_dist: [NPROBE]f32 = .{std.math.inf(f32)} ** NPROBE;

    if (comptime NPROBE == 1) {
        var best_id: u32 = 0;
        var best_d: f32 = std.math.inf(f32);
        var c: u32 = 0;
        const wide_end: u32 = nlist & ~@as(u32, 3);
        while (c < wide_end) : (c += 4) {
            const base: usize = @as(usize, c) * 16;
            const c0: *const [16]f32 = idx.centroids[base..][0..16];
            const c1: *const [16]f32 = idx.centroids[base + 16 ..][0..16];
            const c2: *const [16]f32 = idx.centroids[base + 32 ..][0..16];
            const c3: *const [16]f32 = idx.centroids[base + 48 ..][0..16];
            const d0 = centroidSqDistF32(q_f32_padded, c0);
            const d1 = centroidSqDistF32(q_f32_padded, c1);
            const d2 = centroidSqDistF32(q_f32_padded, c2);
            const d3 = centroidSqDistF32(q_f32_padded, c3);
            if (d0 < best_d) { best_d = d0; best_id = c; }
            if (d1 < best_d) { best_d = d1; best_id = c + 1; }
            if (d2 < best_d) { best_d = d2; best_id = c + 2; }
            if (d3 < best_d) { best_d = d3; best_id = c + 3; }
        }
        while (c < nlist) : (c += 1) {
            const cent_off: usize = @as(usize, c) * 16;
            const cent_full: *const [16]f32 = idx.centroids[cent_off..][0..16];
            const d = centroidSqDistF32(q_f32_padded, cent_full);
            if (d < best_d) { best_d = d; best_id = c; }
        }
        probe_id[0] = @intCast(best_id);
        probe_dist[0] = best_d;
    } else {
        var c: u32 = 0;
        while (c < nlist) : (c += 1) {
            const cent_off: usize = @as(usize, c) * 16;
            const cent_full: *const [16]f32 = idx.centroids[cent_off..][0..16];
            const d = centroidSqDistF32(q_f32_padded, cent_full);
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
    }

    // Stage 2: scan probed clusters with per-dim early-exit feeding Top5.
    // We track the running count of fully-scanned clusters so stage 3 can
    // honour MAX_CLUSTERS_VISITED.
    var top: Top5 = .{};
    var scanned: [MAX_NLIST]bool = .{false} ** MAX_NLIST;
    var visited: u32 = 0;
    inline for (0..NPROBE) |k| {
        const cid_signed = probe_id[k];
        if (cid_signed >= 0) {
            const cid: u32 = @intCast(cid_signed);
            scanned[cid] = true;
            scanInvlist(idx, cid, q_int, &top);
            visited += 1;
        }
    }

    // Stage 3a (fast): bbox-repair on non-probed clusters until `visited`
    // hits FAST_CLUSTERS. This is the only budget paid by ~95% of queries
    // (fraud_count ∉ {2, 3}). For each cluster, compute the squared lower-
    // bound of its bbox vs the query in i16; if lb > worst_d the cluster
    // cannot contribute to the top-5 and is skipped without a scan.
    var ci: u32 = 0;
    while (ci < nlist and visited < FAST_CLUSTERS) : (ci += 1) {
        if (scanned[ci]) continue;
        const start = idx.invlist_offsets[ci];
        const end = idx.invlist_offsets[ci + 1];
        if (end <= start) continue;
        const bb_off: usize = @as(usize, ci) * fmt.DIM;
        const bmin: *const [fmt.DIM]i16 = idx.bbox_min[bb_off..][0..fmt.DIM];
        const bmax: *const [fmt.DIM]i16 = idx.bbox_max[bb_off..][0..fmt.DIM];
        const lb = bboxLowerBound(q_int, bmin, bmax);
        if (lb <= top.worst_d) {
            scanned[ci] = true;
            scanInvlist(idx, ci, q_int, &top);
            visited += 1;
        }
    }

    // Stage 3b (adaptive escalation): only run when the post-fast count sits
    // on the approval boundary (fraud_score ∈ {0.4, 0.6}). One missed neighbour
    // there flips `approved`, which dominates the score's absolute_penalty
    // term. For unambiguous counts (0/1/4/5) the extra scans cannot change the
    // approval decision so we skip them and bank the latency.
    //
    // The fast loop's `ci` cursor resumes where it left off — the bbox-repair
    // pass is monotonic and untouched clusters never become eligible later.
    const fast_count = top.fraudCount();
    if (fast_count == 2 or fast_count == 3) {
        while (ci < nlist and visited < FULL_CLUSTERS) : (ci += 1) {
            if (scanned[ci]) continue;
            const start = idx.invlist_offsets[ci];
            const end = idx.invlist_offsets[ci + 1];
            if (end <= start) continue;
            const bb_off: usize = @as(usize, ci) * fmt.DIM;
            const bmin: *const [fmt.DIM]i16 = idx.bbox_min[bb_off..][0..fmt.DIM];
            const bmax: *const [fmt.DIM]i16 = idx.bbox_max[bb_off..][0..fmt.DIM];
            const lb = bboxLowerBound(q_int, bmin, bmax);
            if (lb <= top.worst_d) {
                scanned[ci] = true;
                scanInvlist(idx, ci, q_int, &top);
                visited += 1;
            }
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

test "stage-1 batched argmin agrees with serial argmin across 10 centroids" {
    // Build a 10-cluster index where centroids differ on a single dim each, so
    // a query aimed near centroid k must pick that cluster at Stage 1. Verifies
    // the 4-wide centroid batch + scalar-tail finds the same minimum index as
    // a serial scan would, including positions inside the tail (8, 9).
    const path = "/tmp/rinha_test_ivf_stage1_batch.bin";
    defer std.fs.cwd().deleteFile(path) catch {};

    const nlist: u32 = 10;
    // 8 vectors per cluster: enough to fill top-5 from a single cluster so
    // worst_d settles at 0 and the bbox-repair stage can soundly prune the
    // other 9 clusters in the test. Anything <5 leaves worst_d at sentinel
    // and bbox repair would visit every cluster regardless of Stage 1 pick.
    const per_cluster: u32 = 8;
    const n: u32 = nlist * per_cluster;

    // Centroids: c[k][0] = (k+1) * 1000.0, all other lanes 0. Query at
    // (k+1) * 1000.0 lands on centroid k uniquely.
    var centroids: [nlist][16]f32 = .{.{0.0} ** 16} ** nlist;
    inline for (0..nlist) |k| centroids[k][0] = @as(f32, @floatFromInt(k + 1)) * 0.1; // f32 unit space

    var offsets: [nlist + 1]u32 = undefined;
    inline for (0..nlist + 1) |k| offsets[k] = @intCast(k * per_cluster);

    // bbox per cluster: tight box around centroid coordinate.
    var bbox_min: [nlist][fmt.DIM]i16 = undefined;
    var bbox_max: [nlist][fmt.DIM]i16 = undefined;
    inline for (0..nlist) |k| {
        const v: i16 = @intCast((@as(i32, k) + 1) * 1000); // dim0 only
        bbox_min[k] = .{0} ** fmt.DIM;
        bbox_max[k] = .{0} ** fmt.DIM;
        bbox_min[k][0] = v;
        bbox_max[k][0] = v;
    }

    var orig_ids: [n]u32 = undefined;
    inline for (0..n) |i| orig_ids[i] = i;

    // Each cluster's `per_cluster` vectors share dim0 = (k+1)*1000, others 0.
    var dim_storage: [fmt.DIM][n]i16 = undefined;
    inline for (0..fmt.DIM) |j| {
        var k: u32 = 0;
        while (k < nlist) : (k += 1) {
            const v: i16 = if (j == 0) @intCast((@as(i32, @intCast(k)) + 1) * 1000) else 0;
            var slot: u32 = 0;
            while (slot < per_cluster) : (slot += 1) {
                dim_storage[j][k * per_cluster + slot] = v;
            }
        }
    }
    var dim_init: [fmt.DIM][]const i16 = undefined;
    inline for (0..fmt.DIM) |j| dim_init[j] = dim_storage[j][0..];

    // Cluster 0: all 8 fraud (top-5 fully fraud → fraud_count=5 if Stage 1
    // hits cluster 0). Other clusters: all legit (fraud_count=0). A wrong
    // Stage 1 pick or a missed prune would surface as a non-{0,5} count.
    var labels_buf: [(n + 7) / 8]u8 = .{0} ** ((n + 7) / 8);
    inline for (0..per_cluster) |slot| {
        labels_buf[slot >> 3] |= @as(u8, 1) << @intCast(slot & 7);
    }

    try loader.writeSyntheticV2(
        path,
        n,
        nlist,
        centroids[0..],
        offsets[0..],
        bbox_min[0..],
        bbox_max[0..],
        orig_ids[0..],
        dim_init[0..],
        labels_buf[0..],
    );
    var idx = try loader.load(path);
    defer loader.unload(&idx);

    // Aim at the tail (cluster 8 and 9) so the scalar-tail of the batched
    // argmin is exercised, not just the wide path.
    inline for ([_]u32{ 0, 1, 4, 7, 8, 9 }) |target| {
        var q_int: [fmt.DIM]i16 = .{0} ** fmt.DIM;
        q_int[0] = @intCast((target + 1) * 1000);
        var q_padded: [16]f32 = .{0.0} ** 16;
        inline for (0..fmt.DIM) |j| q_padded[j] = @as(f32, @floatFromInt(q_int[j])) * idx.inv_scale;
        const result = search(&idx, &q_int, &q_padded);
        const expected: u8 = if (target == 0) 5 else 0;
        try std.testing.expectEqual(expected, result.fraud_count);
    }
}

test "wide-path early-exit matches exhaustive top-5 on dense cluster" {
    // Build a synthetic index with 16 vectors (two wide-path batches) all in
    // a single cluster, then verify search() returns the exact same fraud_count
    // as a brute-force top-5 over the same 16 vectors. Exercises the new
    // SCAN_PREFIX lower-bound prune across multiple batches with varied data.
    const path = "/tmp/rinha_test_ivf_early_exit.bin";
    defer std.fs.cwd().deleteFile(path) catch {};

    const n: u32 = 16;
    const nlist: u32 = 1;
    const centroids: [1][16]f32 = .{.{0.0} ** 16};
    const offsets: [2]u32 = .{ 0, n };
    const bbox_min: [1][fmt.DIM]i16 = .{.{std.math.minInt(i16)} ** fmt.DIM};
    const bbox_max: [1][fmt.DIM]i16 = .{.{std.math.maxInt(i16)} ** fmt.DIM};
    var orig_ids: [n]u32 = undefined;
    inline for (0..n) |i| orig_ids[i] = i;

    // Each vector picks its 14 i16 values from a pseudo-random walk of the
    // index modulo the SCALE range, half marked fraud. Resulting top-5 is
    // non-trivial (not all at the origin) so the early-exit branch must be
    // taken some times and skipped others.
    var dim_storage: [fmt.DIM][n]i16 = undefined;
    inline for (0..fmt.DIM) |j| {
        inline for (0..n) |i| {
            const v: i32 = @intCast(((i + 1) * (j + 1) * 137) % 4000);
            dim_storage[j][i] = @intCast(v);
        }
    }
    var dim_init: [fmt.DIM][]const i16 = undefined;
    inline for (0..fmt.DIM) |j| dim_init[j] = dim_storage[j][0..];

    // labels: alternating fraud/legit gives 8 fraud, 8 legit
    const labels: [2]u8 = .{ 0b10101010, 0b10101010 };

    try loader.writeSyntheticV2(
        path,
        n,
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

    // Sweep multiple queries; for each, brute-force the top-5 (smallest
    // squared L2 distance, tie-break by orig_id) and compare fraud_count.
    const queries = [_][fmt.DIM]i16{
        .{0} ** fmt.DIM,
        .{500} ** fmt.DIM,
        .{ 100, -200, 300, -400, 500, -600, 700, -800, 100, 200, 300, 400, 500, 600 },
        .{ 3000, 2500, 2000, 1500, 1000, 500, 0, -500, -1000, -1500, -2000, -2500, -3000, 0 },
    };

    for (queries) |q| {
        var brute_dist: [n]u64 = undefined;
        for (0..n) |i| {
            var s: u64 = 0;
            inline for (0..fmt.DIM) |j| {
                const d: i64 = @as(i64, q[j]) - @as(i64, dim_storage[j][i]);
                s += @as(u64, @intCast(d * d));
            }
            brute_dist[i] = s;
        }
        // Top-5 by (dist, orig_id) lex order on this 16-element set.
        var ranks: [n]u32 = undefined;
        for (0..n) |i| ranks[i] = @intCast(i);
        std.mem.sort(u32, &ranks, &brute_dist, struct {
            fn lt(ctx: *const [n]u64, a: u32, b: u32) bool {
                const da = ctx[a];
                const db = ctx[b];
                return da < db or (da == db and a < b);
            }
        }.lt);
        var expected: u8 = 0;
        for (ranks[0..5]) |r| expected += labelBit(idx.labels, r);

        var q_padded: [16]f32 = .{0.0} ** 16;
        inline for (0..fmt.DIM) |j| q_padded[j] = @as(f32, @floatFromInt(q[j])) * idx.inv_scale;

        const result = search(&idx, &q, &q_padded);
        try std.testing.expectEqual(expected, result.fraud_count);
    }
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
    const q_f32_padded: [16]f32 = .{0.0} ** 16;
    const result = search(&idx, &q_int, &q_f32_padded);
    try std.testing.expectEqual(@as(u8, 4), result.fraud_count);
}
