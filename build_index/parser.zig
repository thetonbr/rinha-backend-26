//! Build-time parser for the references.json.gz dataset.
//!
//! Reads a gzip-compressed JSON array of objects of the form
//! `[{"vector":[v1,v2,...,v14],"label":"fraud"|"legit"}, ...]` and returns a
//! list of `Record` values. Used to feed k-means and emit the binary index.
//!
//! The official Rinha 2026 references.json.gz is a single-line JSON array with
//! ~3M elements (no newlines), so we cannot rely on NDJSON line splitting. The
//! reader streams the array by counting balanced braces (with quote/escape
//! awareness) to slice out each object, then defers field extraction to
//! `parseLine`.
//!
//! `parseLine` is intentionally simple and tolerant: it scans for the `[`/`]`
//! pair to extract the 14 floats and for the `"label"` key to read the label
//! string. Any malformed/empty input is skipped silently.

const std = @import("std");

pub const VECTOR_DIMS: usize = 14;

pub const Record = struct {
    vector: [VECTOR_DIMS]f32,
    is_fraud: bool,
};

/// Read all records from a gzip-compressed JSON-array file at `path`.
///
/// The caller owns the returned `ArrayList` and must call `deinit` on it.
pub fn readAll(allocator: std.mem.Allocator, path: []const u8) !std.ArrayList(Record) {
    const file = try std.fs.openFileAbsolute(path, .{});
    defer file.close();

    var br = std.io.bufferedReader(file.reader());
    var gz = std.compress.gzip.decompressor(br.reader());

    return readAllFromReader(allocator, gz.reader());
}

fn isWhitespace(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\n' or c == '\r';
}

/// Read all records from any reader producing the decompressed JSON payload.
/// Exposed to make unit testing easier (avoids touching the filesystem).
///
/// The payload must be a JSON array of objects (`[{...},{...},...]`), which
/// matches the official Rinha 2026 references.json.gz layout. The parser does
/// NOT do general JSON parsing: it scans the byte stream tracking string state
/// (with backslash-escape awareness) and a brace depth counter to slice out
/// each top-level object, then forwards the slice to `parseLine`. This works
/// as long as no `{` or `}` appears inside a JSON string literal at the top
/// level of the array — true for the dataset.
pub fn readAllFromReader(allocator: std.mem.Allocator, reader: anytype) !std.ArrayList(Record) {
    var line_buf = std.ArrayList(u8).init(allocator);
    defer line_buf.deinit();

    var out = std.ArrayList(Record).init(allocator);
    errdefer out.deinit();

    // 1. Skip whitespace until the first significant byte; expect '['.
    const first: u8 = while (true) {
        const b = reader.readByte() catch |e| switch (e) {
            error.EndOfStream => return out, // empty input is OK
            else => return e,
        };
        if (!isWhitespace(b)) break b;
    };
    if (first != '[') return error.InvalidJsonRoot;

    // 2. Stream objects from the array.
    outer: while (true) {
        // 2a. Skip whitespace and item separators between elements.
        const b = reader.readByte() catch |e| switch (e) {
            error.EndOfStream => return error.UnexpectedEof,
            else => return e,
        };
        switch (b) {
            ' ', '\t', '\n', '\r', ',' => continue :outer,
            ']' => return out, // end of array
            '{' => {}, // start of object (fall through)
            else => return error.UnexpectedByte,
        }

        // 2b. Accumulate bytes for this object. We've already consumed the
        // opening '{', so seed the buffer and start at depth 1.
        line_buf.clearRetainingCapacity();
        try line_buf.append('{');
        var depth: usize = 1;
        var in_string = false;
        var escape = false;

        while (depth > 0) {
            const c = reader.readByte() catch |e| switch (e) {
                error.EndOfStream => return error.UnexpectedEof,
                else => return e,
            };
            try line_buf.append(c);

            if (in_string) {
                if (escape) {
                    // Previous byte was a backslash; this byte is literal.
                    escape = false;
                } else if (c == '\\') {
                    escape = true;
                } else if (c == '"') {
                    in_string = false;
                }
            } else {
                switch (c) {
                    '"' => in_string = true,
                    '{' => depth += 1,
                    '}' => depth -= 1,
                    else => {},
                }
            }
        }

        // 2c. depth == 0: object complete. Parse it.
        if (try parseLine(line_buf.items)) |rec| try out.append(rec);
    }
}

fn parseLine(raw: []const u8) !?Record {
    const line = std.mem.trim(u8, raw, " \r\t");
    if (line.len == 0) return null;

    // Extract vector between '[' and ']'.
    const lbracket = std.mem.indexOfScalar(u8, line, '[') orelse return null;
    const rbracket = std.mem.indexOfScalarPos(u8, line, lbracket + 1, ']') orelse return null;
    const vec_slice = line[lbracket + 1 .. rbracket];

    var rec: Record = undefined;
    var it = std.mem.splitScalar(u8, vec_slice, ',');
    var i: usize = 0;
    while (it.next()) |tok| : (i += 1) {
        if (i >= VECTOR_DIMS) return null; // too many components
        const trimmed = std.mem.trim(u8, tok, " \t");
        rec.vector[i] = std.fmt.parseFloat(f32, trimmed) catch return null;
    }
    if (i != VECTOR_DIMS) return null; // wrong number of components

    // Extract label string after the "label" key.
    const label_key = "\"label\"";
    const key_pos = std.mem.indexOfPos(u8, line, rbracket, label_key) orelse return null;
    const after_key = key_pos + label_key.len;
    const lq1 = (std.mem.indexOfScalarPos(u8, line, after_key, '"') orelse return null) + 1;
    const lq2 = std.mem.indexOfScalarPos(u8, line, lq1, '"') orelse return null;
    const label = line[lq1..lq2];

    rec.is_fraud = std.mem.eql(u8, label, "fraud");
    return rec;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "parseLine extracts vector and fraud label" {
    const line = "{\"vector\":[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0],\"label\":\"fraud\"}";
    const rec = (try parseLine(line)).?;
    try std.testing.expect(rec.is_fraud);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), rec.vector[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 14.0), rec.vector[13], 1e-6);
}

test "parseLine handles legit label and whitespace" {
    const line = "  {\"vector\":[ 0.5 , 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.5 ], \"label\": \"legit\" }  ";
    const rec = (try parseLine(line)).?;
    try std.testing.expect(!rec.is_fraud);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), rec.vector[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -1.5), rec.vector[13], 1e-6);
}

test "parseLine returns null on malformed input" {
    try std.testing.expect((try parseLine("")) == null);
    try std.testing.expect((try parseLine("not json")) == null);
    // Wrong dim count (only 3 floats).
    try std.testing.expect((try parseLine("{\"vector\":[1,2,3],\"label\":\"fraud\"}")) == null);
}

test "readAll round-trips a tiny gzipped JSON-array file" {
    const allocator = std.testing.allocator;

    // Single-line JSON array: matches the official references.json.gz layout.
    const payload =
        "[{\"vector\":[1,2,3,4,5,6,7,8,9,10,11,12,13,14],\"label\":\"fraud\"}," ++
        "{\"vector\":[0,0,0,0,0,0,0,0,0,0,0,0,0,0],\"label\":\"legit\"}]";

    // Write a temporary .gz file under /tmp/ with the compressed payload.
    const tmp_path = "/tmp/rinha2026_parser_test.json.gz";
    {
        const tmp = try std.fs.createFileAbsolute(tmp_path, .{ .truncate = true });
        defer tmp.close();
        var bw = std.io.bufferedWriter(tmp.writer());
        var src = std.io.fixedBufferStream(payload);
        try std.compress.gzip.compress(src.reader(), bw.writer(), .{});
        try bw.flush();
    }
    defer std.fs.deleteFileAbsolute(tmp_path) catch {};

    var records = try readAll(allocator, tmp_path);
    defer records.deinit();

    try std.testing.expectEqual(@as(usize, 2), records.items.len);
    try std.testing.expect(records.items[0].is_fraud);
    try std.testing.expectApproxEqAbs(@as(f32, 14.0), records.items[0].vector[13], 1e-6);
    try std.testing.expect(!records.items[1].is_fraud);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), records.items[1].vector[0], 1e-6);
}
