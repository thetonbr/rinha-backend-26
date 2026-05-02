const std = @import("std");

pub const Response = struct { bytes: []const u8 };

// Successful responses use `Connection: keep-alive` so HAProxy can pool the
// backend Unix socket across many client requests. Without this, every
// front-end request opens and closes a fresh backend socket, which dominates
// latency under sustained QPS (TIME_WAIT exhaustion + per-request connect
// overhead). 4xx errors keep `Connection: close` because they signal a
// malformed client and we want them to drop immediately.
pub const fraud: [6]Response = blk: {
    var arr: [6]Response = undefined;
    for (0..6) |i| {
        const fc = i; // fraud_count 0..5
        const score_str = switch (fc) {
            0 => "0.0",
            1 => "0.2",
            2 => "0.4",
            3 => "0.6",
            4 => "0.8",
            5 => "1.0",
            else => unreachable,
        };
        const approved_str = if (fc < 3) "true" else "false";
        const body = "{\"approved\":" ++ approved_str ++ ",\"fraud_score\":" ++ score_str ++ "}";
        const cl = std.fmt.comptimePrint("{d}", .{body.len});
        const full = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " ++ cl ++ "\r\nConnection: keep-alive\r\n\r\n" ++ body;
        arr[i] = .{ .bytes = full };
    }
    break :blk arr;
};

pub const ready: Response = .{
    .bytes = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n",
};

pub const bad_request: Response = .{
    .bytes = "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
};

pub const not_found: Response = .{
    .bytes = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
};

test "fraud responses contain expected approved values" {
    try std.testing.expect(std.mem.indexOf(u8, fraud[0].bytes, "\"approved\":true,\"fraud_score\":0.0") != null);
    try std.testing.expect(std.mem.indexOf(u8, fraud[1].bytes, "\"approved\":true,\"fraud_score\":0.2") != null);
    try std.testing.expect(std.mem.indexOf(u8, fraud[2].bytes, "\"approved\":true,\"fraud_score\":0.4") != null);
    try std.testing.expect(std.mem.indexOf(u8, fraud[3].bytes, "\"approved\":false,\"fraud_score\":0.6") != null);
    try std.testing.expect(std.mem.indexOf(u8, fraud[4].bytes, "\"approved\":false,\"fraud_score\":0.8") != null);
    try std.testing.expect(std.mem.indexOf(u8, fraud[5].bytes, "\"approved\":false,\"fraud_score\":1.0") != null);
}

test "fraud responses include keep-alive and json content-type" {
    try std.testing.expect(std.mem.indexOf(u8, fraud[0].bytes, "Content-Type: application/json") != null);
    try std.testing.expect(std.mem.indexOf(u8, fraud[0].bytes, "Connection: keep-alive") != null);
}

test "ready response is 200 keep-alive" {
    try std.testing.expect(std.mem.startsWith(u8, ready.bytes, "HTTP/1.1 200 OK"));
    try std.testing.expect(std.mem.indexOf(u8, ready.bytes, "Connection: keep-alive") != null);
}

test "bad_request and not_found use Connection: close" {
    try std.testing.expect(std.mem.startsWith(u8, bad_request.bytes, "HTTP/1.1 400"));
    try std.testing.expect(std.mem.indexOf(u8, bad_request.bytes, "Connection: close") != null);
    try std.testing.expect(std.mem.startsWith(u8, not_found.bytes, "HTTP/1.1 404"));
    try std.testing.expect(std.mem.indexOf(u8, not_found.bytes, "Connection: close") != null);
}
