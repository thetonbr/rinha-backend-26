const std = @import("std");
const linux = std.os.linux;
const loader = @import("index/loader.zig");
const server = @import("http/server.zig");

const DEFAULT_INDEX_PATH = "/index/index.bin";
const PORT: u16 = 8080;

pub fn main() !void {
    const path = std.posix.getenv("INDEX_PATH") orelse DEFAULT_INDEX_PATH;
    var idx = try loader.load(path);
    defer loader.unload(&idx);

    const fd = try std.posix.socket(
        linux.AF.INET,
        linux.SOCK.STREAM | linux.SOCK.CLOEXEC,
        linux.IPPROTO.TCP,
    );
    defer std.posix.close(fd);

    const one: c_int = 1;
    try std.posix.setsockopt(fd, linux.SOL.SOCKET, linux.SO.REUSEADDR, std.mem.asBytes(&one));
    try std.posix.setsockopt(fd, linux.SOL.SOCKET, linux.SO.REUSEPORT, std.mem.asBytes(&one));
    try std.posix.setsockopt(fd, linux.IPPROTO.TCP, linux.TCP.NODELAY, std.mem.asBytes(&one));

    const addr = linux.sockaddr.in{
        .port = std.mem.nativeToBig(u16, PORT),
        .addr = 0,
    };
    try std.posix.bind(fd, @ptrCast(&addr), @sizeOf(linux.sockaddr.in));
    try std.posix.listen(fd, 4096);

    try server.run(fd, &idx);
}
