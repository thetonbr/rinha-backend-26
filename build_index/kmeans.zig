//! Build-time k-means clustering used to train the IVF index centroids.
//!
//! The algorithm has two stages:
//!
//!   1. **k-means++ seeding** — pick the first centroid uniformly at random,
//!      then iteratively pick the next centroid with probability proportional
//!      to the squared distance from the closest already-chosen centroid. This
//!      gives a much better starting point than uniform random init and tends
//!      to converge in a handful of Lloyd iterations.
//!   2. **Lloyd iterations** — repeatedly assign each vector to its closest
//!      centroid and recompute centroids as the mean of the assigned vectors.
//!      We stop when fewer than 0.1% of the assignments change in a full pass
//!      (or when `max_iter` is reached).
//!
//! All work happens at build time, so the implementation favors simplicity and
//! determinism (fixed seed) over peak throughput. The returned `KMeans` owns
//! its `centroids` and `assignments` slices.

const std = @import("std");

/// Vector dimensionality of the Rinha 2026 dataset. Hard-coded so the inner
/// distance loop can be fully unrolled by the compiler.
pub const DIMS: usize = 14;

pub const KMeans = struct {
    centroids: [][DIMS]f32,
    assignments: []u32,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *KMeans) void {
        self.allocator.free(self.centroids);
        self.allocator.free(self.assignments);
    }
};

/// Squared euclidean distance between two 14-dim vectors.
///
/// Using the squared form avoids a sqrt that would not change the argmin and
/// keeps the k-means++ probability proportional to dist^2 directly.
fn distSq(a: [DIMS]f32, b: [DIMS]f32) f32 {
    var s: f32 = 0;
    inline for (0..DIMS) |i| {
        const d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

/// Train an IVF centroid table on `vectors` using k-means++ + Lloyd.
///
/// Arguments:
///   - `vectors`: training set, owned by the caller. Must contain at least
///     `nlist` vectors.
///   - `nlist`: number of centroids to produce.
///   - `max_iter`: hard cap on Lloyd iterations.
///   - `seed`: PRNG seed; same seed yields identical centroids across runs.
///
/// The returned `KMeans` owns its allocations; call `deinit` to free them.
pub fn train(
    allocator: std.mem.Allocator,
    vectors: [][DIMS]f32,
    nlist: u32,
    max_iter: u32,
    seed: u64,
) !KMeans {
    std.debug.assert(vectors.len >= nlist);

    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();

    const centroids = try allocator.alloc([DIMS]f32, nlist);
    errdefer allocator.free(centroids);

    const assignments = try allocator.alloc(u32, vectors.len);
    errdefer allocator.free(assignments);
    // Initialize assignments to a sentinel so the first Lloyd iteration counts
    // every vector as "changed" deterministically (allocator.alloc returns
    // uninitialized memory, which would otherwise leak into the convergence
    // metric in ReleaseFast).
    @memset(assignments, std.math.maxInt(u32));

    // ---------------------------------------------------------------------
    // k-means++ seeding
    // ---------------------------------------------------------------------
    // First centroid: uniform random pick from the training set.
    {
        const idx = r.uintLessThan(usize, vectors.len);
        centroids[0] = vectors[idx];
    }

    // d2[i] = squared distance from vectors[i] to its nearest already-chosen
    // centroid. Updated incrementally each round to avoid an O(nlist^2) pass.
    const d2 = try allocator.alloc(f32, vectors.len);
    defer allocator.free(d2);
    for (vectors, 0..) |v, i| d2[i] = distSq(v, centroids[0]);

    var c: usize = 1;
    while (c < nlist) : (c += 1) {
        // Sample the next centroid with probability proportional to d2.
        var total: f64 = 0;
        for (d2) |x| total += @as(f64, x);
        // Degenerate case: all training points already coincide with a
        // centroid. Fall back to a uniform random pick so we still produce
        // `nlist` distinct slots (duplicates are harmless for IVF).
        if (total <= 0) {
            const idx = r.uintLessThan(usize, vectors.len);
            centroids[c] = vectors[idx];
        } else {
            const target = r.float(f64) * total;
            var acc: f64 = 0;
            var picked: usize = 0;
            for (d2, 0..) |x, i| {
                acc += @as(f64, x);
                if (acc >= target) {
                    picked = i;
                    break;
                }
            }
            centroids[c] = vectors[picked];
        }

        // Refresh d2: each entry becomes min(d2[i], dist^2 to the new centroid).
        for (vectors, 0..) |v, i| {
            const nd = distSq(v, centroids[c]);
            if (nd < d2[i]) d2[i] = nd;
        }

        if (c % 100 == 0) {
            std.debug.print("kmeans++: {d}/{d}\n", .{ c, nlist });
        }
    }

    // ---------------------------------------------------------------------
    // Lloyd iterations
    // ---------------------------------------------------------------------
    const sums = try allocator.alloc([DIMS]f32, nlist);
    defer allocator.free(sums);
    const counts = try allocator.alloc(u32, nlist);
    defer allocator.free(counts);

    var iter: u32 = 0;
    while (iter < max_iter) : (iter += 1) {
        // Reset per-cluster accumulators.
        for (sums) |*s| {
            for (s) |*x| x.* = 0;
        }
        @memset(counts, 0);

        // Assign each vector to its closest centroid and accumulate stats.
        var changed: usize = 0;
        for (vectors, 0..) |v, i| {
            var best: u32 = 0;
            var best_d: f32 = distSq(v, centroids[0]);
            var ci: u32 = 1;
            while (ci < nlist) : (ci += 1) {
                const d = distSq(v, centroids[ci]);
                if (d < best_d) {
                    best_d = d;
                    best = ci;
                }
            }
            if (assignments[i] != best) changed += 1;
            assignments[i] = best;

            counts[best] += 1;
            inline for (0..DIMS) |k| sums[best][k] += v[k];
        }

        // Recompute centroids as the mean of the assigned vectors. Empty
        // clusters keep their previous centroid (rare with k-means++ init).
        var ci2: u32 = 0;
        while (ci2 < nlist) : (ci2 += 1) {
            const cnt = counts[ci2];
            if (cnt == 0) continue;
            const inv: f32 = 1.0 / @as(f32, @floatFromInt(cnt));
            inline for (0..DIMS) |k| centroids[ci2][k] = sums[ci2][k] * inv;
        }

        std.debug.print("kmeans iter {d}: changed={d}/{d}\n", .{ iter, changed, vectors.len });
        // Convergence: < 0.1% of vectors changed cluster this pass.
        if (changed * 1000 < vectors.len) break;
    }

    return .{
        .centroids = centroids,
        .assignments = assignments,
        .allocator = allocator,
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "train converges on tiny synthetic dataset" {
    const allocator = std.testing.allocator;

    // 8 vectors in 2 well-separated clusters: 4 at the origin and 4 at (1,...,1).
    var vecs: [8][DIMS]f32 = .{.{0} ** DIMS} ** 8;
    inline for (4..8) |i| {
        inline for (0..DIMS) |j| vecs[i][j] = 1.0;
    }

    var km = try train(allocator, vecs[0..], 2, 20, 42);
    defer km.deinit();

    var counts: [2]u32 = .{ 0, 0 };
    for (km.assignments) |a| counts[a] += 1;

    // Both clusters should be balanced regardless of which label k-means picks
    // for which group. We assert the partition is exactly 4/4.
    try std.testing.expect(counts[0] == 4 and counts[1] == 4);
}
