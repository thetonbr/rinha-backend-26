const std = @import("std");
const linux = std.os.linux;
const conn = @import("../io/conn.zig");
const uring = @import("../io/uring.zig");
const loader = @import("../index/loader.zig");
const fraud_payload = @import("../json/fraud_payload.zig");

// Multishot accept (kernel >= 5.19) keeps the same SQE armed across all
// incoming connections, saving one prep_accept per accept. We saw the
// multishot variant interact badly with HAProxy mode http + http-reuse
// always under WSL2 5.15 smoke (the first probe never returned a CQE,
// HAProxy timed out the layer-7 health check at 1 s and the backend was
// marked DOWN before any request arrived). Single-shot reposted accept is
// the same shape the API used in v17/epoll-era and works reliably.
//
// In mode http + http-reuse, HAProxy holds the same backend Unix socket
// across many client requests, so accept rate is low and the per-accept
// re-arm cost (~50 ns of one extra prep_accept) is negligible.
pub const USE_MULTISHOT_ACCEPT: bool = false;

// Backing storage for the accept SQE address fields. Kernel writes here on
// every accepted connection (multishot mode); single-shot also reuses these.
// Must outlive every CQE produced by the SQE — i.e. for the lifetime of run().
var accept_addr: linux.sockaddr = undefined;
var accept_addr_len: linux.socklen_t = @sizeOf(linux.sockaddr);

inline fn armAccept(ring: *linux.IoUring, listen_fd: i32) !void {
    const sqe = try ring.get_sqe();
    if (USE_MULTISHOT_ACCEPT) {
        sqe.prep_multishot_accept(listen_fd, &accept_addr, &accept_addr_len, 0);
    } else {
        sqe.prep_accept(listen_fd, &accept_addr, &accept_addr_len, 0);
    }
    sqe.user_data = conn.pack(.accept, 0);
}

pub fn run(listen_fd: i32, idx: *const loader.Index) !void {
    var ring = try uring.init();
    defer ring.deinit();

    var pool: conn.ConnPool = .{};
    var scratch: fraud_payload.ParseScratch = .{};

    try armAccept(&ring, listen_fd);

    // Single batched submit_and_wait per loop iteration: handlers stage SQEs
    // (no per-handler ring.submit call). Accumulated SQEs flush together with
    // the kernel wait below, collapsing N enter() syscalls per iteration into
    // one. With DEFER_TASKRUN + SINGLE_ISSUER + COOP_TASKRUN (kernel >= 6.1)
    // the kernel batches task work anyway; even on the WSL2 5.15 fallback the
    // reduced syscall count is the dominant effect.
    while (true) {
        _ = try ring.submit_and_wait(1);
        // Drain the completion queue manually. We use the raw head/tail/cqes
        // accessors rather than `copy_cqes` because we want to avoid copying
        // every CQE through a stack buffer on the hot path; the kernel writes
        // them directly into the mmap'd ring and we can read them in place.
        var head = ring.cq.head.*;
        const tail = @atomicLoad(u32, ring.cq.tail, .acquire);
        while (head != tail) : (head +%= 1) {
            const cqe = ring.cq.cqes[head & ring.cq.mask];
            try dispatch(&ring, &pool, &scratch, idx, cqe, listen_fd);
        }
        @atomicStore(u32, ring.cq.head, head, .release);
    }
}

fn dispatch(
    ring: *linux.IoUring,
    pool: *conn.ConnPool,
    scratch: *fraud_payload.ParseScratch,
    idx: *const loader.Index,
    cqe: linux.io_uring_cqe,
    listen_fd: i32,
) !void {
    const u = conn.unpack(cqe.user_data);
    switch (u.tag) {
        .accept => {
            // Multishot accept produces one CQE per accepted connection; the
            // SQE stays armed as long as IORING_CQE_F_MORE is set in flags.
            // A negative res indicates failure of that single accept; the
            // armed SQE will keep firing for the next. In single-shot mode we
            // re-arm explicitly after each accept (success or failure). All
            // staged SQEs flush together in run()'s submit_and_wait — handlers
            // never call ring.submit() directly.
            if (!USE_MULTISHOT_ACCEPT) {
                try armAccept(ring, listen_fd);
            }
            if (cqe.res < 0) return;
            const fd: i32 = cqe.res;
            const cid = pool.alloc() orelse {
                _ = linux.close(fd);
                return;
            };
            const c = &pool.conns[cid];
            c.fd = fd;
            c.state = .reading;
            const sqe = try ring.get_sqe();
            sqe.prep_read(fd, c.buf[0..], 0);
            sqe.user_data = conn.pack(.read, cid);
        },
        .read => {
            const c = &pool.conns[u.id];
            if (cqe.res <= 0) {
                _ = linux.close(c.fd);
                pool.free(u.id);
                return;
            }
            const n: usize = @intCast(cqe.res);
            // The only error handleRead can surface is `NeedMore`; every
            // other failure mode is folded into a pre-baked bad_request
            // response inside handleRead itself.
            const resp = conn.handleRead(c, n, idx, scratch) catch {
                // Buffer does not contain a complete request yet; refill
                // into the unused tail of the buffer.
                const sqe = try ring.get_sqe();
                sqe.prep_read(c.fd, c.buf[c.buf_len..], 0);
                sqe.user_data = conn.pack(.read, u.id);
                return;
            };
            c.write_ptr = resp.ptr;
            c.write_len = resp.len;
            c.write_done = 0;
            c.state = .writing;
            const sqe = try ring.get_sqe();
            sqe.prep_write(c.fd, resp, 0);
            sqe.user_data = conn.pack(.write, u.id);
        },
        .write => {
            const c = &pool.conns[u.id];
            if (cqe.res <= 0) {
                _ = linux.close(c.fd);
                pool.free(u.id);
                return;
            }
            const written: usize = @intCast(cqe.res);
            c.write_done += written;
            if (c.write_done < c.write_len) {
                const remaining = c.write_ptr[c.write_done..c.write_len];
                const sqe = try ring.get_sqe();
                sqe.prep_write(c.fd, remaining, 0);
                sqe.user_data = conn.pack(.write, u.id);
                return;
            }
            if (!c.keepalive) {
                _ = linux.close(c.fd);
                pool.free(u.id);
                return;
            }
            // Keep-alive: rearm read. We discard any bytes that arrived after
            // the request body (HTTP pipelining) — the Rinha workload sends
            // one request per connection so this is fine for now.
            c.buf_len = 0;
            c.state = .reading;
            const sqe = try ring.get_sqe();
            sqe.prep_read(c.fd, c.buf[0..], 0);
            sqe.user_data = conn.pack(.read, u.id);
        },
        .close => {},
    }
}
