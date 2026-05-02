//! Build-time orchestrator for the IVF index.
//!
//! Pipeline:
//!   1. parse references.json.gz into f32 vectors + fraud labels
//!   2. train k-means (k = nlist) on the float vectors
//!   3. quantize each vector to int16 and place it in cluster-sorted order
//!      so each invlist is contiguous in memory
//!   4. pack the fraud labels into a bitset
//!   5. emit the binary index via writer.write
//!   6. (placeholder) validate exact-vs-approx recall on a query sample
//!
//! Usage: build_index <refs.json.gz> <out/index.bin> [nlist=2048]
const std = @import("std");
const parser = @import("parser.zig");
const kmeans = @import("kmeans.zig");
const writer = @import("writer.zig");
const recall_mod = @import("recall.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();
    _ = args.skip();
    const refs_path = args.next() orelse return error.MissingArg;
    const out_path = args.next() orelse return error.MissingArg;
    const nlist_str = args.next() orelse "2048";
    const nlist: u32 = try std.fmt.parseInt(u32, nlist_str, 10);

    std.debug.print("Reading {s}...\n", .{refs_path});
    var records = try parser.readAll(allocator, refs_path);
    defer records.deinit();

    std.debug.print("Loaded {d} records. Building float vectors...\n", .{records.items.len});
    const vectors = try allocator.alloc([14]f32, records.items.len);
    defer allocator.free(vectors);
    const labels = try allocator.alloc(bool, records.items.len);
    defer allocator.free(labels);
    for (records.items, vectors, labels) |r, *v, *l| {
        v.* = r.vector;
        l.* = r.is_fraud;
    }

    std.debug.print("Training k-means k={d}...\n", .{nlist});
    var km = try kmeans.train(allocator, vectors, nlist, 10, 42);
    defer km.deinit();

    // Pad centroids from 14 to 16 lanes so the runtime distance kernel can
    // load them as a single ymm register without per-lane masking.
    const centroids_padded = try allocator.alloc([16]f32, nlist);
    defer allocator.free(centroids_padded);
    for (km.centroids, centroids_padded) |c, *cp| {
        inline for (0..14) |j| cp[j] = c[j];
        cp[14] = 0;
        cp[15] = 0;
    }

    // Build invlist offsets via histogram + prefix sum over assignments.
    std.debug.print("Building invlists...\n", .{});
    const counts = try allocator.alloc(u32, nlist);
    defer allocator.free(counts);
    @memset(counts, 0);
    for (km.assignments) |a| counts[a] += 1;

    const offsets = try allocator.alloc(u32, nlist + 1);
    defer allocator.free(offsets);
    offsets[0] = 0;
    for (0..nlist) |i| offsets[i + 1] = offsets[i] + counts[i];

    const sorted_vec_q = try allocator.alloc([16]i16, vectors.len);
    defer allocator.free(sorted_vec_q);
    const sorted_labels = try allocator.alloc(bool, vectors.len);
    defer allocator.free(sorted_labels);
    const cursor = try allocator.alloc(u32, nlist);
    defer allocator.free(cursor);
    @memset(cursor, 0);

    // Quantize and place in cluster-sorted order in a single pass: for each
    // input vector we know its destination invlist (offsets[a] + cursor[a])
    // so we can write the int16 form directly into its final slot, avoiding a
    // separate sort/shuffle step.
    for (vectors, km.assignments, labels) |v, a, l| {
        var qv: [16]i16 = .{0} ** 16;
        inline for (0..14) |j| {
            const x = v[j] * 10000.0;
            const c = @max(-32768.0, @min(32767.0, x));
            qv[j] = @intFromFloat(@round(c));
        }
        const dest = offsets[a] + cursor[a];
        sorted_vec_q[dest] = qv;
        sorted_labels[dest] = l;
        cursor[a] += 1;
    }

    // Pack the (cluster-sorted) fraud labels into a LSB-first bitset; the
    // runtime tests bit `i & 7` of byte `i >> 3` for vector i.
    const bits_len = (vectors.len + 7) / 8;
    const bits = try allocator.alloc(u8, bits_len);
    defer allocator.free(bits);
    @memset(bits, 0);
    for (sorted_labels, 0..) |l, i| {
        if (l) bits[i >> 3] |= @as(u8, 1) << @intCast(i & 7);
    }

    std.debug.print("Writing {s}...\n", .{out_path});
    try writer.write(.{
        .out_path = out_path,
        .n_vectors = @intCast(vectors.len),
        .nlist = nlist,
        .centroids_padded = centroids_padded,
        .invlist_offsets = offsets,
        .vectors_q_sorted = sorted_vec_q,
        .fraud_bits = bits,
    });

    // Placeholder recall validation; real implementation lands in a follow-up
    // task. Calling it here keeps the module referenced and validated at build
    // time so we catch signature drift early.
    const r = try recall_mod.validateExactVsApprox(allocator, vectors, labels, 1000);
    std.debug.print("Recall@5: {d:.4}, Decision flip rate: {d:.4}\n", .{ r.recall_at_5, r.decision_flip_rate });

    std.debug.print("Done.\n", .{});
}
