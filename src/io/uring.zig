const std = @import("std");
const linux = std.os.linux;

pub const ENTRIES: u16 = 4096;

// Toggle to compare SQPOLL vs DEFER_TASKRUN under benchmark. Defaults to false
// (DEFER_TASKRUN) — kernel ≥ 6.1, lower CPU usage at idle, predictable latency.
// Flip to true to evaluate SQPOLL (kernel ≥ 5.13, higher peak throughput at the
// cost of a kernel polling thread that always burns CPU).
pub const USE_SQPOLL: bool = false;

pub fn init() !linux.IoUring {
    var params = std.mem.zeroes(linux.io_uring_params);
    if (USE_SQPOLL) {
        params.flags = linux.IORING_SETUP_SQPOLL
            | linux.IORING_SETUP_SINGLE_ISSUER
            | linux.IORING_SETUP_COOP_TASKRUN;
        params.sq_thread_idle = 2_000;
    } else {
        params.flags = linux.IORING_SETUP_SINGLE_ISSUER
            | linux.IORING_SETUP_DEFER_TASKRUN
            | linux.IORING_SETUP_COOP_TASKRUN;
    }
    // Older kernels (< 6.1) reject DEFER_TASKRUN with EINVAL; SINGLE_ISSUER
    // requires >= 6.0. The Rinha judge runs Ubuntu 24.04 with kernel 6.x, but
    // local WSL2 / dev hosts often ship 5.15. Fall back to a plain init so the
    // binary still boots for smoke testing without sacrificing the fast path
    // in production.
    return linux.IoUring.init_params(ENTRIES, &params) catch |e| switch (e) {
        error.ArgumentsInvalid => {
            var fallback = std.mem.zeroes(linux.io_uring_params);
            return linux.IoUring.init_params(ENTRIES, &fallback);
        },
        else => return e,
    };
}
