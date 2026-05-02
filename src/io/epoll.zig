const std = @import("std");
const linux = std.os.linux;
const posix = std.posix;
const c = @import("conn.zig");
const loader = @import("../index/loader.zig");
const fraud_payload = @import("../json/fraud_payload.zig");

// Synchronous epoll(7) edge-triggered event loop. Replaces the previous
// io_uring transport because, on the Rinha 2026 sub-millisecond workload,
// the overhead of staging and reaping SQEs outweighed the benefit of
// batched syscalls. The reference rank-#4 entry (pandrRe/rinhapuffer) lands
// p99 ~3.8ms with a plain epoll loop and we are reproducing that shape.
//
// Design notes:
//
//   * Single-thread, single-process per backend container. Concurrency is
//     handled by registering each accepted fd with EPOLLIN | EPOLLOUT | ET.
//     Edge-triggered means the kernel reports readiness once per state
//     change, so handlers must drain reads/writes until EAGAIN.
//
//   * Connection slots are c.Conn instances reused as-is — we keep the
//     existing handleRead() handler intact so only the transport changes.
//     The pool is a flat array of MAX_CONNS slots indexed by epoll_data.u64.
//     The listener fd uses LISTEN_TAG to disambiguate accept events from
//     connection events without a separate fd→idx hash.
//
//   * Keep-alive supported. After a response drains we reset buf_len and
//     return the connection to .reading so the next request on the same
//     socket can be parsed without paying connect/accept overhead. HAProxy
//     reuses backend connections across many client requests, so without
//     keep-alive every client request would open a fresh Unix socket and
//     dominate latency under sustained load.

pub const MAX_CONNS: u32 = 256;
const EVENTS_PER_WAIT: u32 = 64;
const ACCEPT_BURST: u32 = 64;

// Listener fd is registered with this tag in epoll_data.u64. It must not
// collide with any valid pool index (0..MAX_CONNS-1).
const LISTEN_TAG: u64 = std.math.maxInt(u64);

// in_use is tracked separately so c.Conn (carried over from the io_uring
// version) can stay shape-stable. Linear scan over 256 entries on accept is
// well under a cache line miss budget; we only optimise read/write paths.
var conn_pool: [MAX_CONNS]c.Conn = undefined;
var in_use: [MAX_CONNS]bool = undefined;

fn poolReset() void {
    for (&conn_pool) |*conn| conn.* = .{};
    for (&in_use) |*flag| flag.* = false;
}

fn poolAlloc() ?u32 {
    for (&in_use, 0..) |flag, i| {
        if (!flag) {
            in_use[i] = true;
            conn_pool[i] = .{};
            return @intCast(i);
        }
    }
    return null;
}

fn poolFree(idx: u32) void {
    in_use[idx] = false;
}

inline fn setNonblocking(fd: i32) !void {
    // F.GETFL returns the current flags as the syscall return value; we OR
    // in O_NONBLOCK and round-trip with F.SETFL. linux.fcntl returns usize
    // where the low bits are the flags on success or a (negated) errno on
    // failure; we rely on Linux signalling errors via the high bit.
    const cur_raw = linux.fcntl(fd, linux.F.GETFL, 0);
    const cur_signed: isize = @bitCast(cur_raw);
    if (cur_signed < 0) return error.FcntlFailed;
    const O_NONBLOCK: usize = 0o4000;
    const new_flags: usize = @as(usize, @intCast(cur_signed)) | O_NONBLOCK;
    const set_rc = linux.fcntl(fd, linux.F.SETFL, new_flags);
    if (@as(isize, @bitCast(set_rc)) < 0) return error.FcntlFailed;
}

inline fn epollAdd(epfd: i32, fd: i32, tag: u64, events: u32) !void {
    var ev: linux.epoll_event = .{
        .events = events,
        .data = .{ .u64 = tag },
    };
    try posix.epoll_ctl(epfd, linux.EPOLL.CTL_ADD, fd, &ev);
}

pub fn run(listen_fd: i32, idx: *const loader.Index) !void {
    // The listener is registered as edge-triggered as well; this means we
    // must drain accept() until EAGAIN every time EPOLLIN fires for it.
    try setNonblocking(listen_fd);

    const epfd = try posix.epoll_create1(linux.EPOLL.CLOEXEC);
    defer posix.close(epfd);

    poolReset();

    try epollAdd(epfd, listen_fd, LISTEN_TAG, linux.EPOLL.IN | linux.EPOLL.ET);

    var scratch: fraud_payload.ParseScratch = .{};
    var events: [EVENTS_PER_WAIT]linux.epoll_event = undefined;

    // Stash for the keep-alive re-arm path inside tryDrainWrite. This is a
    // single-threaded loop so a globals-as-statics pattern is fine.
    scratch_ref = &scratch;
    idx_ref = idx;

    while (true) {
        const n = posix.epoll_wait(epfd, events[0..], -1);
        if (n == 0) continue;

        for (events[0..n]) |ev| {
            if (ev.data.u64 == LISTEN_TAG) {
                acceptBurst(epfd, listen_fd);
                continue;
            }

            const conn_idx: u32 = @intCast(ev.data.u64);
            if (!in_use[conn_idx]) continue;

            const flags = ev.events;

            if ((flags & (linux.EPOLL.ERR | linux.EPOLL.HUP)) != 0) {
                // ERR/HUP are terminal; the only thing left to do is close.
                // RDHUP alone would still allow draining a queued response,
                // but we never have one in flight at this point in practice.
                closeConn(conn_idx);
                continue;
            }

            // Order matters: drain the input first (may produce a response
            // and flip state to .writing), then attempt to drain the output.
            // This collapses the read+write path into one epoll wakeup when
            // the kernel reports both EPOLLIN and EPOLLOUT, which is the
            // common case for short HTTP request/response pairs under ET.
            const conn = &conn_pool[conn_idx];
            if ((flags & linux.EPOLL.IN) != 0 and conn.state != .writing) {
                tryReadDispatch(conn_idx, idx, &scratch);
                if (!in_use[conn_idx]) continue;
            }
            if (conn.state == .writing and (flags & linux.EPOLL.OUT) != 0) {
                tryDrainWrite(conn_idx);
            }
        }
    }
}

fn acceptBurst(epfd: i32, listen_fd: i32) void {
    // Edge-triggered listener: drain until EAGAIN. ACCEPT_BURST caps the
    // number of accept() calls per wakeup so a connection storm cannot
    // starve already-accepted connections of CPU time.
    var accepted: u32 = 0;
    while (accepted < ACCEPT_BURST) : (accepted += 1) {
        const fd = posix.accept(
            listen_fd,
            null,
            null,
            linux.SOCK.NONBLOCK | linux.SOCK.CLOEXEC,
        ) catch |e| switch (e) {
            error.WouldBlock => return,
            // Any other accept error: bail out of the burst and let the
            // next epoll wakeup re-attempt. We deliberately do not crash —
            // ECONNABORTED, EMFILE, etc. are all transient.
            else => return,
        };

        const conn_idx_opt = poolAlloc();
        if (conn_idx_opt == null) {
            // Pool full: drop the new connection rather than blocking the
            // event loop. Comes back later when slots free up.
            posix.close(fd);
            continue;
        }
        const conn_idx = conn_idx_opt.?;
        const conn = &conn_pool[conn_idx];
        conn.fd = fd;
        conn.state = .reading;
        conn.buf_len = 0;
        conn.write_len = 0;
        conn.write_done = 0;
        conn.keepalive = false;

        // Always-armed EPOLLIN | EPOLLOUT | ET. We keep both bits set
        // permanently so the kernel reports readiness immediately for either
        // direction without an epoll_ctl(MOD) per state transition; the
        // handler dispatches based on conn.state.
        epollAdd(
            epfd,
            fd,
            @as(u64, conn_idx),
            linux.EPOLL.IN | linux.EPOLL.OUT | linux.EPOLL.ET | linux.EPOLL.RDHUP,
        ) catch {
            posix.close(fd);
            poolFree(conn_idx);
            continue;
        };
    }
}

fn closeConn(conn_idx: u32) void {
    const conn = &conn_pool[conn_idx];
    if (conn.fd >= 0) {
        // close() implicitly removes the fd from any epoll set we registered
        // it with (we never dup() so there is no extra reference).
        posix.close(conn.fd);
        conn.fd = -1;
    }
    poolFree(conn_idx);
}

// Module-local scratch reference set up once at the top of run(). We make
// it accessible to tryDrainWrite so it can chain back into the read path on
// keep-alive completion without threading an extra parameter everywhere.
var scratch_ref: ?*fraud_payload.ParseScratch = null;
var idx_ref: ?*const loader.Index = null;

fn tryReadDispatch(
    conn_idx: u32,
    idx: *const loader.Index,
    scratch: *fraud_payload.ParseScratch,
) void {
    const conn = &conn_pool[conn_idx];

    // Outer loop: drain syscalls until EAGAIN, NeedMore, or a complete
    // request. handleRead is shared with the legacy io_uring path and works
    // off conn.buf / conn.buf_len directly, so we just feed it the bytes we
    // read each iteration.
    while (true) {
        if (conn.buf_len == conn.buf.len) {
            // Buffer exhausted before we found a complete request; either a
            // malicious oversized request or a malformed pipeline. Close.
            closeConn(conn_idx);
            return;
        }

        const slice = conn.buf[conn.buf_len..];
        const n = posix.read(conn.fd, slice) catch |e| switch (e) {
            error.WouldBlock => return,
            else => {
                closeConn(conn_idx);
                return;
            },
        };
        if (n == 0) {
            // Peer closed.
            closeConn(conn_idx);
            return;
        }

        // handleRead increments buf_len internally with `n` and returns the
        // pre-baked response bytes for a complete request, or error.NeedMore
        // if the buffer does not yet hold a full request.
        const resp = c.handleRead(conn, n, idx, scratch) catch |e| switch (e) {
            error.NeedMore => continue,
        };
        conn.write_ptr = resp.ptr;
        conn.write_len = resp.len;
        conn.write_done = 0;
        conn.state = .writing;
        tryDrainWrite(conn_idx);
        return;
    }
}

fn tryDrainWrite(conn_idx: u32) void {
    const conn = &conn_pool[conn_idx];

    while (conn.write_done < conn.write_len) {
        const remaining_ptr = conn.write_ptr + conn.write_done;
        const remaining_len = conn.write_len - conn.write_done;
        const remaining = remaining_ptr[0..remaining_len];
        const written = posix.write(conn.fd, remaining) catch |e| switch (e) {
            error.WouldBlock => return,
            else => {
                closeConn(conn_idx);
                return;
            },
        };
        if (written == 0) {
            closeConn(conn_idx);
            return;
        }
        conn.write_done += written;
    }

    // Response fully delivered. If the client requested keep-alive (and our
    // response also signals it), recycle the slot for the next request: we
    // discard any bytes that arrived after the parsed request body (HTTP
    // pipelining is not used by the Rinha workload) and put the connection
    // back into .reading. Otherwise tear the fd down.
    if (conn.keepalive) {
        conn.buf_len = 0;
        conn.write_len = 0;
        conn.write_done = 0;
        conn.state = .reading;
        // Edge-triggered epoll only re-fires EPOLLIN when new bytes arrive
        // after a read returns EAGAIN. The next request may already be
        // queued in the socket buffer (HAProxy pipelines requests on a
        // pooled backend connection), so we must drain proactively here.
        if (scratch_ref) |s| if (idx_ref) |i| tryReadDispatch(conn_idx, i, s);
        return;
    }
    closeConn(conn_idx);
}

test "poolAlloc returns distinct slots and reuses freed ones" {
    poolReset();
    const a = poolAlloc().?;
    const b = poolAlloc().?;
    try std.testing.expect(a != b);
    poolFree(a);
    const reused = poolAlloc().?;
    try std.testing.expectEqual(a, reused);
}

test "poolAlloc returns null when full" {
    poolReset();
    var i: u32 = 0;
    while (i < MAX_CONNS) : (i += 1) {
        _ = poolAlloc().?;
    }
    try std.testing.expectEqual(@as(?u32, null), poolAlloc());
}
