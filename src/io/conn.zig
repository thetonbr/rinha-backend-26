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

// Per-connection slot. handleRead reads from `buf[0..buf_len]` and emits the
// response into write_ptr/write_len for the transport (io_uring) to drain.
// The fields are sized for the worst-case Rinha request (single fraud-score
// POST body, ≤4 KiB) and a comptime-baked response.
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

// Bitmap-backed free-slot tracker. With MAX_CONNS=256 we get exactly four
// u64 words; finding a free slot is `@ctz(~word)` on the first non-zero word
// (~30 cycles vs ~256-iteration linear scan in the worst case).
pub const ConnPool = struct {
    conns: [MAX_CONNS]Conn = .{.{}} ** MAX_CONNS,
    in_use_bits: [MAX_CONNS / 64]u64 = .{0} ** (MAX_CONNS / 64),

    pub fn alloc(self: *ConnPool) ?u32 {
        comptime std.debug.assert(MAX_CONNS % 64 == 0);
        for (&self.in_use_bits, 0..) |*word, w| {
            const free_mask: u64 = ~word.*;
            if (free_mask == 0) continue;
            const bit: u6 = @intCast(@ctz(free_mask));
            word.* |= (@as(u64, 1) << bit);
            const idx: u32 = @intCast(w * 64 + @as(usize, bit));
            self.conns[idx] = .{};
            return idx;
        }
        return null;
    }

    pub fn free(self: *ConnPool, id: u32) void {
        const w = id / 64;
        const bit: u6 = @intCast(id % 64);
        self.in_use_bits[w] &= ~(@as(u64, 1) << bit);
    }
};

// io_uring user_data layout: tag in high 32 bits, conn id in low 32 bits.
// MAX_CONNS=256 fits in 32 bits with massive headroom; the layout lets the
// dispatch site mask out the tag with a single shift and read the id with a
// truncate, no extra arithmetic.
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
        // Convert i16->f32 once for the centroid stage and pad to 16 lanes so
        // the centroid distance kernel can do a single Vec16f32 subtract +
        // square-and-reduce. Lanes 14-15 are zero in both query and centroid
        // (the index layout pads centroids to 16) so they cancel in the diff.
        var q_f32_padded: [16]f32 = .{0.0} ** 16;
        const inv_scale = idx.inv_scale;
        inline for (0..fmt.DIM) |j| {
            q_f32_padded[j] = @as(f32, @floatFromInt(q_int[j])) * inv_scale;
        }
        const result = ivf.search(idx, &q_int, &q_f32_padded);
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

test "ConnPool fills 256 slots and rejects beyond" {
    var pool: ConnPool = .{};
    var i: u32 = 0;
    while (i < MAX_CONNS) : (i += 1) {
        _ = pool.alloc().?;
    }
    try std.testing.expectEqual(@as(?u32, null), pool.alloc());
}
