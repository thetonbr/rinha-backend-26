//! Build-time recall validation harness.
//!
//! Compares the approximate IVF top-K result against an exact full-scan top-K
//! over a sample of queries to estimate two metrics:
//!
//!   - `decision_flip_rate`: fraction of queries whose final fraud/legit
//!     decision flips between the exact and approximate variants.
//!   - `recall_at_5`: average overlap between the two top-5 sets.
//!
//! The current implementation is a placeholder so the orchestrator can wire it
//! up; a follow-up task replaces it with a real exact-vs-approx comparison.
const std = @import("std");

pub fn validateExactVsApprox(
    allocator: std.mem.Allocator,
    vectors: [][14]f32,
    is_fraud: []const bool,
    n_queries: usize,
) !struct { decision_flip_rate: f32, recall_at_5: f32 } {
    _ = allocator;
    _ = vectors;
    _ = is_fraud;
    _ = n_queries;
    return .{ .decision_flip_rate = 0.0, .recall_at_5 = 1.0 };
}
