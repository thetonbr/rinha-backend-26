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

// Locate the position of "\r\n\r\n" in `buf`. Returns null if not present.
//
// Implementation: scan in 16-byte windows using four overlapping byte vectors
// at offsets +0/+1/+2/+3, then AND the four equality masks against the
// constant {\r, \n, \r, \n} pattern and reduce. The hot HTTP request headers
// fit in well under 1 KiB, but std.mem.indexOf in Zig 0.13 is byte-at-a-time
// and dominates parser cost on a sub-millisecond budget. The 4-byte aligned
// SIMD form retires the search in ~3-5 instructions per 16 bytes vs ~8 for
// scalar boyer-moore. Falls back to scalar near the buffer tail.
const VEC_BYTES: usize = 16;
const Vu8 = @Vector(VEC_BYTES, u8);

inline fn findHeaderEnd(buf: []const u8) ?usize {
    if (buf.len < 4) return null;
    const cr_splat: Vu8 = @splat('\r');
    const lf_splat: Vu8 = @splat('\n');

    var i: usize = 0;
    // Need 16 bytes at i + 3 bytes of right-shift lookahead = 19 bytes resident.
    while (i + VEC_BYTES + 3 <= buf.len) : (i += VEC_BYTES) {
        const v0: Vu8 = buf[i..][0..VEC_BYTES].*;
        const v1: Vu8 = buf[i + 1 ..][0..VEC_BYTES].*;
        const v2: Vu8 = buf[i + 2 ..][0..VEC_BYTES].*;
        const v3: Vu8 = buf[i + 3 ..][0..VEC_BYTES].*;
        const m0 = v0 == cr_splat;
        const m1 = v1 == lf_splat;
        const m2 = v2 == cr_splat;
        const m3 = v3 == lf_splat;
        const both: @Vector(VEC_BYTES, bool) = @select(bool, m0, m1, @as(@Vector(VEC_BYTES, bool), @splat(false)));
        const trip: @Vector(VEC_BYTES, bool) = @select(bool, both, m2, @as(@Vector(VEC_BYTES, bool), @splat(false)));
        const quad: @Vector(VEC_BYTES, bool) = @select(bool, trip, m3, @as(@Vector(VEC_BYTES, bool), @splat(false)));
        if (@reduce(.Or, quad)) {
            inline for (0..VEC_BYTES) |k| {
                if (quad[k]) return i + k;
            }
        }
    }
    // Scalar tail: at most VEC_BYTES + 2 bytes left to inspect.
    while (i + 4 <= buf.len) : (i += 1) {
        if (buf[i] == '\r' and buf[i + 1] == '\n' and buf[i + 2] == '\r' and buf[i + 3] == '\n') {
            return i;
        }
    }
    return null;
}

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
    const headers_end = findHeaderEnd(buf) orelse return error.NeedMore;
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

test "findHeaderEnd matches std.mem.indexOf across alignments and corner cases" {
    // Drive the SIMD path past every 16-byte boundary so off-by-one bugs
    // surface. Each input has its delimiter at a different offset modulo
    // VEC_BYTES; result must equal std.mem.indexOf, including null returns.
    const cases = [_][]const u8{
        "GET / HTTP/1.1\r\n\r\n",                                            // delim @14, in vec 0
        "GET /xyz HTTP/1.1\r\n\r\n",                                         // delim @17, in vec 1 region
        "POST /fraud-score HTTP/1.1\r\nHost: x\r\nContent-Length: 1\r\n\r\n", // delim ~62
        "AAAAAAAAAAAAAAAAAAAAAAAA\r\n\r\n",                                  // delim @24
        "AAAAAAAAAAAAAAA\r\n\r\nXXXX",                                       // delim @15 (boundary)
        "BBBBBBBBBBBBBBBB\r\n\r\nYYYY",                                      // delim @16
        "CCCCCCCCCCCCCCCCC\r\n\r\nZZZZ",                                     // delim @17
        "noheaders here at all",                                             // none → null
        "shortbut\r\n",                                                      // truncated → null
        "\r\n\r\n",                                                          // shortest valid: at offset 0
    };
    for (cases) |c| {
        const expected = std.mem.indexOf(u8, c, "\r\n\r\n");
        const got = findHeaderEnd(c);
        try std.testing.expectEqual(expected, got);
    }
}

test "parse rejects oversized Content-Length" {
    const raw = "POST /fraud-score HTTP/1.1\r\nContent-Length: 999999999\r\n\r\n";
    try std.testing.expectError(error.BadRequest, parse(raw));
}
