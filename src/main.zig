const std = @import("std");
const linux = std.os.linux;
const loader = @import("index/loader.zig");
const epoll = @import("io/epoll.zig");
const fmt = @import("index/format.zig");

const DEFAULT_INDEX_PATH = "/index/index.bin";
const PORT: u16 = 8080;
const LISTEN_BACKLOG: u31 = 4096;

pub fn main() !void {
    const path = std.posix.getenv("INDEX_PATH") orelse DEFAULT_INDEX_PATH;
    var idx = try loader.load(path);
    defer loader.unload(&idx);

    // Pre-warm: touch the first byte of every page in each per-dim block and
    // orig_ids. MAP_POPULATE faults pages eagerly but does not necessarily
    // populate per-CPU TLB entries — the first hot-path access still pays a
    // few hundred ns of TLB-walk + L1 miss. Walking once at boot moves that
    // cost out of the latency-critical request path.
    prewarm(&idx);

    // Listener selection: when LISTEN_SOCKET_PATH is set, bind a Unix domain
    // socket at that path (used in production behind HAProxy). Otherwise fall
    // back to TCP on PORT for local smoke testing and backwards compatibility.
    const sock_path_opt = std.posix.getenv("LISTEN_SOCKET_PATH");
    const fd = if (sock_path_opt) |p| try openUnixListener(p) else try openTcpListener();
    defer std.posix.close(fd);

    try epoll.run(fd, &idx);
}

fn openTcpListener() !i32 {
    const fd = try std.posix.socket(
        linux.AF.INET,
        linux.SOCK.STREAM | linux.SOCK.CLOEXEC,
        linux.IPPROTO.TCP,
    );
    errdefer std.posix.close(fd);

    const one: c_int = 1;
    try std.posix.setsockopt(fd, linux.SOL.SOCKET, linux.SO.REUSEADDR, std.mem.asBytes(&one));
    try std.posix.setsockopt(fd, linux.SOL.SOCKET, linux.SO.REUSEPORT, std.mem.asBytes(&one));
    try std.posix.setsockopt(fd, linux.IPPROTO.TCP, linux.TCP.NODELAY, std.mem.asBytes(&one));

    const addr = linux.sockaddr.in{
        .port = std.mem.nativeToBig(u16, PORT),
        .addr = 0,
    };
    try std.posix.bind(fd, @ptrCast(&addr), @sizeOf(linux.sockaddr.in));
    try std.posix.listen(fd, LISTEN_BACKLOG);
    return fd;
}

fn openUnixListener(path: []const u8) !i32 {
    // sockaddr.un.path is a fixed 108-byte buffer that must be NUL-terminated
    // for non-abstract sockets. Reject paths that would not fit (with room
    // for the trailing NUL).
    if (path.len >= @sizeOf(@TypeOf(@as(linux.sockaddr.un, undefined).path))) {
        return error.SocketPathTooLong;
    }

    const fd = try std.posix.socket(
        linux.AF.UNIX,
        linux.SOCK.STREAM | linux.SOCK.CLOEXEC,
        0,
    );
    errdefer std.posix.close(fd);

    // Stale socket from a previous run would cause bind() to EADDRINUSE. The
    // unlink is best-effort: ENOENT is fine, anything else surfaces at bind.
    std.posix.unlink(path) catch |e| switch (e) {
        error.FileNotFound => {},
        else => return e,
    };

    var addr: linux.sockaddr.un = .{ .path = std.mem.zeroes([108]u8) };
    @memcpy(addr.path[0..path.len], path);
    addr.path[path.len] = 0;

    // socklen for SUN sockets = offsetof(path) + strlen(path) + 1. Using the
    // full sizeof() also works on Linux but pads with NULs the kernel ignores;
    // we use the strict form for portability with strace/tools.
    const addrlen: linux.socklen_t = @intCast(@offsetOf(linux.sockaddr.un, "path") + path.len + 1);
    try std.posix.bind(fd, @ptrCast(&addr), addrlen);
    // The default mode of bind() inherits umask, leaving the socket as 0755.
    // HAProxy in a sibling container connects with a different effective uid,
    // so without the world-write bit it gets EACCES on connect(). 0666 keeps
    // every user on the shared volume able to talk to the API.
    try std.posix.fchmodat(linux.AT.FDCWD, path, 0o666, 0);
    try std.posix.listen(fd, LISTEN_BACKLOG);
    return fd;
}

// Walk one byte per 4 KiB page across each per-dim block + orig_ids so all
// page-table entries land in the CPU's TLB and every page is resident before
// the first request. Cost is ~30-80ms at startup for a 750k-vector index.
fn prewarm(idx: *const loader.Index) void {
    const PAGE: u32 = 4096;
    const stride_i16: u32 = PAGE / @sizeOf(i16);
    const stride_u32: u32 = PAGE / @sizeOf(u32);
    var sink: u32 = 0;

    inline for (0..fmt.DIM) |j| {
        const dj = idx.dims[j];
        var i: u32 = 0;
        while (i < dj.len) : (i += stride_i16) {
            sink +%= @as(u32, @bitCast(@as(i32, dj[i])));
        }
    }
    var i: u32 = 0;
    while (i < idx.orig_ids.len) : (i += stride_u32) {
        sink +%= idx.orig_ids[i];
    }
    // `sink` must escape the optimizer or LLVM will delete the whole walk.
    asm volatile (""
        :
        : [sink] "r" (sink),
        : "memory"
    );
}
