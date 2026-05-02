const std = @import("std");
const linux = std.os.linux;
const parser = @import("../http/parser.zig");
const responses = @import("../http/responses.zig");
const fraud_payload = @import("../json/fraud_payload.zig");
const builder = @import("../vector/builder.zig");
const ivf = @import("../index/ivf.zig");
const loader = @import("../index/loader.zig");
const fmt = @import("../index/format.zig");

pub const BUF_SIZE: usize = 8 * 1024;
pub const MAX_CONNS: usize = 256;

pub const ConnState = enum { idle, reading, writing };

// Per-connection slot. Owned by the epoll event loop; handleRead reads from
// `buf[0..buf_len]` and emits the response into write_ptr/write_len for the
// transport to drain. The fields are sized for the worst-case Rinha request
// (single fraud-score POST body, ≤4 KiB) and a comptime-baked response.
pub const Conn = struct {
    fd: i32 = -1,
    state: ConnState = .idle,
    buf: [BUF_SIZE]u8 = undefined,
    buf_len: usize = 0,
    // Reserved for dynamic response composition (e.g. Date header) in later
    // tasks; currently unused because handleRead returns pointers to comptime
    // pre-baked response bytes.
    write_buf: [256]u8 = undefined,
    write_ptr: [*]const u8 = undefined,
    write_len: usize = 0,
    write_done: usize = 0,
    keepalive: bool = false,
};

// Returns the response bytes to write back, or `error.NeedMore` if the buffer
// does not yet hold a complete request. The caller owns re-arming a read in
// the latter case. The `req.path` and `req.body` slices alias `conn.buf` and
// remain valid until the buffer is reused.
pub fn handleRead(
    c: *Conn,
    n: usize,
    idx: *const loader.Index,
    scratch: *fraud_payload.ParseScratch,
) ![]const u8 {
    c.buf_len += n;
    const req = parser.parse(c.buf[0..c.buf_len]) catch |e| switch (e) {
        error.NeedMore => return error.NeedMore,
        else => return responses.bad_request.bytes,
    };

    c.keepalive = req.keepalive;

    if (req.method == .GET and std.mem.eql(u8, req.path, "/ready")) {
        return responses.ready.bytes;
    }
    if (req.method == .POST and std.mem.eql(u8, req.path, "/fraud-score")) {
        const p = fraud_payload.parse(req.body, scratch) catch return responses.bad_request.bytes;
        var q_int: [fmt.DIM]i16 = undefined;
        builder.build(p, &q_int) catch return responses.bad_request.bytes;
        // Convert i16->f32 once for the centroid stage. With DIM=14 we just
        // unroll instead of building a wide vector — LLVM autovectorizes the
        // 14 scalar muls into ymm ops at -O3.
        var q_f32: [fmt.DIM]f32 = undefined;
        const inv_scale = idx.inv_scale;
        inline for (0..fmt.DIM) |j| {
            q_f32[j] = @as(f32, @floatFromInt(q_int[j])) * inv_scale;
        }
        const result = ivf.search(idx, &q_int, &q_f32);
        return responses.fraud[result.fraud_count].bytes;
    }
    return responses.not_found.bytes;
}

test "Conn defaults are valid" {
    const c: Conn = .{};
    try std.testing.expectEqual(@as(i32, -1), c.fd);
    try std.testing.expectEqual(ConnState.idle, c.state);
    try std.testing.expectEqual(@as(usize, 0), c.buf_len);
    try std.testing.expectEqual(false, c.keepalive);
}
