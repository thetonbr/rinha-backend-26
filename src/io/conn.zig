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
    keepalive: bool = true,
};

pub const ConnPool = struct {
    conns: [MAX_CONNS]Conn = .{.{}} ** MAX_CONNS,
    in_use: [MAX_CONNS]bool = .{false} ** MAX_CONNS,

    pub fn alloc(self: *ConnPool) ?u32 {
        for (&self.in_use, 0..) |*used, i| {
            if (!used.*) {
                used.* = true;
                self.conns[i] = .{};
                return @intCast(i);
            }
        }
        return null;
    }

    pub fn free(self: *ConnPool, id: u32) void {
        self.in_use[id] = false;
    }
};

// User-data layout: tag in high 32 bits, conn id in low 32 bits. The id space
// is bounded by MAX_CONNS (256) so 32 bits is plenty; we keep them in low bits
// so that masking is a single AND with 0xFFFF_FFFF.
pub const Tag = enum(u8) { accept = 1, read = 2, write = 3, close = 4 };

pub inline fn pack(tag: Tag, conn_id: u32) u64 {
    return (@as(u64, @intFromEnum(tag)) << 32) | @as(u64, conn_id);
}

pub inline fn unpack(ud: u64) struct { tag: Tag, id: u32 } {
    return .{
        .tag = @enumFromInt(@as(u8, @truncate(ud >> 32))),
        .id = @truncate(ud),
    };
}

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

test "pack/unpack roundtrip" {
    const ud = pack(.read, 42);
    const u = unpack(ud);
    try std.testing.expectEqual(Tag.read, u.tag);
    try std.testing.expectEqual(@as(u32, 42), u.id);
}

test "ConnPool alloc/free reuses slots" {
    var pool: ConnPool = .{};
    const a = pool.alloc().?;
    const b = pool.alloc().?;
    try std.testing.expect(a != b);
    pool.free(a);
    const c = pool.alloc().?;
    try std.testing.expectEqual(a, c);
}
