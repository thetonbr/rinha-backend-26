const std = @import("std");

// Cap matches conn.zig BUF_SIZE — anything larger cannot fit in the read buffer
// and would block the connection in NeedMore forever, an obvious DoS lever.
pub const MAX_BODY: usize = 8 * 1024;

pub const Method = enum { GET, POST, OTHER };

pub const Request = struct {
    method: Method,
    path: []const u8,
    content_length: usize = 0,
    body: []const u8 = "",
    keepalive: bool = true,
};

pub const ParseError = error{
    NeedMore,
    BadRequest,
    Unsupported,
};

const CRLF = "\r\n";

fn caseEqlAscii(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    for (a, b) |ca, cb| {
        const la = if (ca >= 'A' and ca <= 'Z') ca + 32 else ca;
        const lb = if (cb >= 'A' and cb <= 'Z') cb + 32 else cb;
        if (la != lb) return false;
    }
    return true;
}

pub fn parse(buf: []const u8) !Request {
    const headers_end = std.mem.indexOf(u8, buf, "\r\n\r\n") orelse return error.NeedMore;
    // include the CRLF that precedes the blank line so every header ends with CRLF
    const head = buf[0 .. headers_end + 2];

    const first_line_end = std.mem.indexOf(u8, head, CRLF) orelse return error.BadRequest;
    const line = head[0..first_line_end];

    var it = std.mem.splitScalar(u8, line, ' ');
    const method_str = it.next() orelse return error.BadRequest;
    const path = it.next() orelse return error.BadRequest;
    _ = it.next() orelse return error.BadRequest;

    const method: Method = if (std.mem.eql(u8, method_str, "GET")) .GET
        else if (std.mem.eql(u8, method_str, "POST")) .POST
        else .OTHER;

    var req = Request{ .method = method, .path = path };

    var hpos: usize = first_line_end + 2;
    while (hpos < head.len) {
        const eol = std.mem.indexOfPos(u8, head, hpos, CRLF) orelse break;
        const header = head[hpos..eol];
        if (std.mem.indexOfScalar(u8, header, ':')) |colon| {
            const name = header[0..colon];
            var val_start = colon + 1;
            while (val_start < header.len and header[val_start] == ' ') val_start += 1;
            const value = header[val_start..];
            if (caseEqlAscii(name, "Content-Length")) {
                const cl = try std.fmt.parseInt(usize, value, 10);
                if (cl > MAX_BODY) return error.BadRequest;
                req.content_length = cl;
            } else if (caseEqlAscii(name, "Connection")) {
                if (caseEqlAscii(value, "close")) req.keepalive = false;
            }
        }
        hpos = eol + 2;
    }

    const body_start = headers_end + 4;
    const have_body = buf.len - body_start;
    if (have_body < req.content_length) return error.NeedMore;
    req.body = buf[body_start .. body_start + req.content_length];
    return req;
}

test "parse GET /ready" {
    const req = try parse("GET /ready HTTP/1.1\r\nHost: x\r\n\r\n");
    try std.testing.expectEqual(Method.GET, req.method);
    try std.testing.expectEqualStrings("/ready", req.path);
    try std.testing.expectEqual(@as(usize, 0), req.content_length);
}

test "parse POST /fraud-score with body" {
    const raw = "POST /fraud-score HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\n\r\nhello";
    const req = try parse(raw);
    try std.testing.expectEqual(Method.POST, req.method);
    try std.testing.expectEqualStrings("/fraud-score", req.path);
    try std.testing.expectEqual(@as(usize, 5), req.content_length);
    try std.testing.expectEqualStrings("hello", req.body);
}

test "parse incomplete returns NeedMore" {
    const result = parse("POST /fraud-score HTTP/1.1\r\nContent-Length: 100\r\n\r\nshort");
    try std.testing.expectError(error.NeedMore, result);
}

test "parse Connection: close" {
    const req = try parse("GET /ready HTTP/1.1\r\nConnection: close\r\n\r\n");
    try std.testing.expect(!req.keepalive);
}

test "parse rejects oversized Content-Length" {
    const raw = "POST /fraud-score HTTP/1.1\r\nContent-Length: 999999999\r\n\r\n";
    try std.testing.expectError(error.BadRequest, parse(raw));
}
