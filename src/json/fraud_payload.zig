const std = @import("std");
const builder = @import("../vector/builder.zig");
const time = @import("../vector/time.zig");
const PayloadValues = builder.PayloadValues;

pub const ParseError = error{
    InvalidJson,
    MissingField,
    OutOfRange,
};

const KNOWN_MERCHANTS_MAX = 32;

pub const ParseScratch = struct {
    known_buf: [KNOWN_MERCHANTS_MAX][]const u8 = undefined,
    known_hashes: [KNOWN_MERCHANTS_MAX]u64 = undefined,
    known_len: usize = 0,
};

// FNV-1a u64 over a small string (~16 bytes for "MERC-XXXX..."). Cheap enough
// to compute once per known_merchant during parse, then turn the O(N*len)
// merchant-membership test in builder.build into an O(N) word compare with a
// std.mem.eql fallback for the (vanishingly rare) hash collision.
inline fn fnv1aU64(s: []const u8) u64 {
    var h: u64 = 0xcbf29ce484222325;
    for (s) |c| {
        h ^= c;
        h = h *% 0x100000001b3;
    }
    return h;
}

// Schema-strict decimal parser: matches "[-]?\d+(\.\d{0,N})?" from DATASET.md.
// All numeric fields in the official payload follow this shape (≤ 2 fractional
// digits, never scientific). Avoids the dispatch + locale machinery in
// std.fmt.parseFloat — measured ~80 ns/call vs ~250 ns. Falls back to the
// stdlib parser if a non-schema char (e.g. 'e') appears, preserving correctness
// against adversarial inputs.
inline fn parseDecimal2(s: []const u8) !f64 {
    if (s.len == 0) return error.InvalidJson;
    var i: usize = 0;
    var neg = false;
    if (s[0] == '-') { neg = true; i = 1; }
    if (i >= s.len) return error.InvalidJson;
    var int_part: u64 = 0;
    const int_start = i;
    while (i < s.len) : (i += 1) {
        const c = s[i];
        if (c < '0' or c > '9') break;
        int_part = int_part * 10 + (c - '0');
    }
    if (i == int_start) return error.InvalidJson;
    var frac: u64 = 0;
    var frac_div: f64 = 1.0;
    if (i < s.len and s[i] == '.') {
        i += 1;
        while (i < s.len) : (i += 1) {
            const c = s[i];
            if (c < '0' or c > '9') break;
            frac = frac * 10 + (c - '0');
            frac_div *= 10.0;
        }
    }
    // Anything left (e.g. 'e', '+') means scientific or malformed input —
    // delegate to the stdlib parser so we stay correct on the slow path.
    if (i != s.len) return std.fmt.parseFloat(f64, s);
    var v: f64 = @as(f64, @floatFromInt(int_part)) + @as(f64, @floatFromInt(frac)) / frac_div;
    if (neg) v = -v;
    return v;
}

const Scanner = struct {
    buf: []const u8,
    pos: usize = 0,

    fn skipWs(self: *Scanner) void {
        while (self.pos < self.buf.len) : (self.pos += 1) {
            const c = self.buf[self.pos];
            if (c != ' ' and c != '\t' and c != '\n' and c != '\r') break;
        }
    }

    fn peek(self: *Scanner) ?u8 {
        return if (self.pos < self.buf.len) self.buf[self.pos] else null;
    }

    fn expect(self: *Scanner, c: u8) !void {
        self.skipWs();
        if (self.pos >= self.buf.len or self.buf[self.pos] != c) return error.InvalidJson;
        self.pos += 1;
    }

    fn readString(self: *Scanner) ![]const u8 {
        self.skipWs();
        if (self.pos >= self.buf.len or self.buf[self.pos] != '"') return error.InvalidJson;
        self.pos += 1;
        const start = self.pos;
        while (self.pos < self.buf.len and self.buf[self.pos] != '"') : (self.pos += 1) {}
        if (self.pos >= self.buf.len) return error.InvalidJson;
        const slice = self.buf[start..self.pos];
        self.pos += 1;
        return slice;
    }

    fn readNumber(self: *Scanner) !f64 {
        self.skipWs();
        const start = self.pos;
        if (self.peek() == @as(u8, '-')) self.pos += 1;
        while (self.pos < self.buf.len) : (self.pos += 1) {
            const c = self.buf[self.pos];
            if (!((c >= '0' and c <= '9') or c == '.' or c == 'e' or c == 'E' or c == '+' or c == '-')) break;
        }
        if (start == self.pos) return error.InvalidJson;
        return try parseDecimal2(self.buf[start..self.pos]);
    }

    fn readBool(self: *Scanner) !bool {
        self.skipWs();
        if (self.pos + 4 <= self.buf.len and std.mem.eql(u8, self.buf[self.pos..self.pos+4], "true")) {
            self.pos += 4; return true;
        }
        if (self.pos + 5 <= self.buf.len and std.mem.eql(u8, self.buf[self.pos..self.pos+5], "false")) {
            self.pos += 5; return false;
        }
        return error.InvalidJson;
    }

    fn isNull(self: *Scanner) bool {
        self.skipWs();
        if (self.pos + 4 <= self.buf.len and std.mem.eql(u8, self.buf[self.pos..self.pos+4], "null")) {
            self.pos += 4; return true;
        }
        return false;
    }

    // Explicit error set required because skipValue/skipObject/skipArray are
    // mutually recursive — inferred error sets cannot resolve cycles.
    const ScanError = error{InvalidJson} || std.fmt.ParseFloatError;

    fn skipValue(self: *Scanner) ScanError!void {
        self.skipWs();
        const c = self.peek() orelse return error.InvalidJson;
        switch (c) {
            '{' => try self.skipObject(),
            '[' => try self.skipArray(),
            '"' => _ = try self.readString(),
            't', 'f' => _ = try self.readBool(),
            'n' => _ = self.isNull(),
            else => _ = try self.readNumber(),
        }
    }

    fn skipObject(self: *Scanner) ScanError!void {
        try self.expect('{');
        self.skipWs();
        if (self.peek() == @as(u8, '}')) { self.pos += 1; return; }
        while (true) {
            _ = try self.readString();
            try self.expect(':');
            try self.skipValue();
            self.skipWs();
            const c = self.peek() orelse return error.InvalidJson;
            if (c == ',') { self.pos += 1; continue; }
            if (c == '}') { self.pos += 1; return; }
            return error.InvalidJson;
        }
    }

    fn skipArray(self: *Scanner) ScanError!void {
        try self.expect('[');
        self.skipWs();
        if (self.peek() == @as(u8, ']')) { self.pos += 1; return; }
        while (true) {
            try self.skipValue();
            self.skipWs();
            const c = self.peek() orelse return error.InvalidJson;
            if (c == ',') { self.pos += 1; continue; }
            if (c == ']') { self.pos += 1; return; }
            return error.InvalidJson;
        }
    }
};

fn parseTransaction(sc: *Scanner, p: *PayloadValues) !void {
    try sc.expect('{');
    while (true) {
        const k = try sc.readString();
        try sc.expect(':');
        if (std.mem.eql(u8, k, "amount")) {
            p.amount = @floatCast(try sc.readNumber());
        } else if (std.mem.eql(u8, k, "installments")) {
            // @intFromFloat is UB in ReleaseFast when value is out of range or NaN.
            // Clamp explicitly so adversarial payloads cannot wrap silently.
            const inst_f = try sc.readNumber();
            if (std.math.isNan(inst_f) or inst_f < 0 or inst_f > 255) return error.OutOfRange;
            p.installments = @intFromFloat(inst_f);
        } else if (std.mem.eql(u8, k, "requested_at")) {
            p.requested_at_iso = try sc.readString();
        } else {
            try sc.skipValue();
        }
        sc.skipWs();
        const c = sc.peek() orelse return error.InvalidJson;
        if (c == ',') { sc.pos += 1; continue; }
        if (c == '}') { sc.pos += 1; return; }
        return error.InvalidJson;
    }
}

fn parseCustomer(sc: *Scanner, p: *PayloadValues, scratch: *ParseScratch) !void {
    try sc.expect('{');
    while (true) {
        const k = try sc.readString();
        try sc.expect(':');
        if (std.mem.eql(u8, k, "avg_amount")) {
            p.cust_avg_amount = @floatCast(try sc.readNumber());
        } else if (std.mem.eql(u8, k, "tx_count_24h")) {
            // @intFromFloat is UB in ReleaseFast when value is out of range or NaN.
            const tx_f = try sc.readNumber();
            if (std.math.isNan(tx_f) or tx_f < 0 or tx_f > @as(f64, std.math.maxInt(u32))) return error.OutOfRange;
            p.cust_tx_count_24h = @intFromFloat(tx_f);
        } else if (std.mem.eql(u8, k, "known_merchants")) {
            try sc.expect('[');
            sc.skipWs();
            scratch.known_len = 0;
            if (sc.peek() == @as(u8, ']')) { sc.pos += 1; }
            else {
                while (true) {
                    if (scratch.known_len >= KNOWN_MERCHANTS_MAX) return error.OutOfRange;
                    const s = try sc.readString();
                    scratch.known_buf[scratch.known_len] = s;
                    scratch.known_hashes[scratch.known_len] = fnv1aU64(s);
                    scratch.known_len += 1;
                    sc.skipWs();
                    const c = sc.peek() orelse return error.InvalidJson;
                    if (c == ',') { sc.pos += 1; continue; }
                    if (c == ']') { sc.pos += 1; break; }
                    return error.InvalidJson;
                }
            }
            p.cust_known_merchants = scratch.known_buf[0..scratch.known_len];
            p.cust_known_hashes = scratch.known_hashes[0..scratch.known_len];
        } else {
            try sc.skipValue();
        }
        sc.skipWs();
        const c = sc.peek() orelse return error.InvalidJson;
        if (c == ',') { sc.pos += 1; continue; }
        if (c == '}') { sc.pos += 1; return; }
        return error.InvalidJson;
    }
}

fn parseMerchant(sc: *Scanner, p: *PayloadValues) !void {
    try sc.expect('{');
    while (true) {
        const k = try sc.readString();
        try sc.expect(':');
        if (std.mem.eql(u8, k, "id")) {
            const id = try sc.readString();
            p.merch_id = id;
            p.merch_id_hash = fnv1aU64(id);
        } else if (std.mem.eql(u8, k, "mcc")) {
            const s = try sc.readString();
            p.merch_mcc = blk: {
                var v: u32 = 0;
                for (s) |c| {
                    if (c < '0' or c > '9') return error.InvalidJson;
                    v = v * 10 + (c - '0');
                }
                break :blk v;
            };
        } else if (std.mem.eql(u8, k, "avg_amount")) {
            p.merch_avg_amount = @floatCast(try sc.readNumber());
        } else {
            try sc.skipValue();
        }
        sc.skipWs();
        const c = sc.peek() orelse return error.InvalidJson;
        if (c == ',') { sc.pos += 1; continue; }
        if (c == '}') { sc.pos += 1; return; }
        return error.InvalidJson;
    }
}

fn parseTerminal(sc: *Scanner, p: *PayloadValues) !void {
    try sc.expect('{');
    while (true) {
        const k = try sc.readString();
        try sc.expect(':');
        if (std.mem.eql(u8, k, "is_online")) {
            p.term_is_online = try sc.readBool();
        } else if (std.mem.eql(u8, k, "card_present")) {
            p.term_card_present = try sc.readBool();
        } else if (std.mem.eql(u8, k, "km_from_home")) {
            p.term_km_from_home = @floatCast(try sc.readNumber());
        } else {
            try sc.skipValue();
        }
        sc.skipWs();
        const c = sc.peek() orelse return error.InvalidJson;
        if (c == ',') { sc.pos += 1; continue; }
        if (c == '}') { sc.pos += 1; return; }
        return error.InvalidJson;
    }
}

fn parseLastTx(sc: *Scanner, p: *PayloadValues, requested_at: []const u8) !void {
    if (sc.isNull()) {
        p.last_tx_minutes = null;
        p.last_tx_km = null;
        return;
    }
    try sc.expect('{');
    var ts: []const u8 = "";
    var km: f32 = 0.0;
    var has_ts = false;
    var has_km = false;
    while (true) {
        const k = try sc.readString();
        try sc.expect(':');
        if (std.mem.eql(u8, k, "timestamp")) {
            ts = try sc.readString();
            has_ts = true;
        } else if (std.mem.eql(u8, k, "km_from_current")) {
            km = @floatCast(try sc.readNumber());
            has_km = true;
        } else {
            try sc.skipValue();
        }
        sc.skipWs();
        const c = sc.peek() orelse return error.InvalidJson;
        if (c == ',') { sc.pos += 1; continue; }
        if (c == '}') { sc.pos += 1; break; }
        return error.InvalidJson;
    }
    if (!has_ts or !has_km) return error.MissingField;
    const t_now = try time.parseIso8601(requested_at);
    const t_prev = try time.parseIso8601(ts);
    const minutes = @as(f32, @floatFromInt(t_now.epoch_s - t_prev.epoch_s)) / 60.0;
    p.last_tx_minutes = minutes;
    p.last_tx_km = km;
}

// Field order assumption: `transaction` precedes `last_transaction` so
// `requested_at_iso` is populated before parseLastTx needs it. Canonical
// payload from the dataset always follows this ordering.
pub fn parse(buf: []const u8, scratch: *ParseScratch) !PayloadValues {
    var sc = Scanner{ .buf = buf };
    var p: PayloadValues = std.mem.zeroInit(PayloadValues, .{
        .last_tx_minutes = @as(?f32, null),
        .last_tx_km = @as(?f32, null),
    });
    p.cust_known_merchants = scratch.known_buf[0..0];
    p.cust_known_hashes = scratch.known_hashes[0..0];
    try sc.expect('{');
    while (true) {
        const k = try sc.readString();
        try sc.expect(':');
        if (std.mem.eql(u8, k, "transaction")) {
            try parseTransaction(&sc, &p);
        } else if (std.mem.eql(u8, k, "customer")) {
            try parseCustomer(&sc, &p, scratch);
        } else if (std.mem.eql(u8, k, "merchant")) {
            try parseMerchant(&sc, &p);
        } else if (std.mem.eql(u8, k, "terminal")) {
            try parseTerminal(&sc, &p);
        } else if (std.mem.eql(u8, k, "last_transaction")) {
            try parseLastTx(&sc, &p, p.requested_at_iso);
        } else {
            try sc.skipValue();
        }
        sc.skipWs();
        const c = sc.peek() orelse return error.InvalidJson;
        if (c == ',') { sc.pos += 1; continue; }
        if (c == '}') { sc.pos += 1; break; }
        return error.InvalidJson;
    }
    return p;
}

const sample =
    \\{
    \\  "id": "tx-1",
    \\  "transaction": {"amount": 384.88, "installments": 3, "requested_at": "2025-09-22T19:24:51Z"},
    \\  "customer": {"avg_amount": 230.50, "tx_count_24h": 3, "known_merchants": ["MERC-100", "MERC-208"]},
    \\  "merchant": {"id": "MERC-001", "mcc": "5912", "avg_amount": 312.00},
    \\  "terminal": {"is_online": true, "card_present": false, "km_from_home": 13.7},
    \\  "last_transaction": {"timestamp": "2025-09-22T18:30:00Z", "km_from_current": 0.4}
    \\}
;

test "parse canonical payload" {
    var s: ParseScratch = .{};
    const p = try parse(sample, &s);
    try std.testing.expectEqual(@as(f32, 384.88), p.amount);
    try std.testing.expectEqual(@as(u8, 3), p.installments);
    try std.testing.expectEqualStrings("2025-09-22T19:24:51Z", p.requested_at_iso);
    try std.testing.expectEqual(@as(u32, 5912), p.merch_mcc);
    try std.testing.expectEqualStrings("MERC-001", p.merch_id);
    try std.testing.expect(p.term_is_online);
    try std.testing.expect(!p.term_card_present);
    try std.testing.expectEqual(@as(usize, 2), p.cust_known_merchants.len);
}

const sample_null =
    \\{"id":"x","transaction":{"amount":1.0,"installments":1,"requested_at":"2025-01-01T00:00:00Z"},
    \\"customer":{"avg_amount":1.0,"tx_count_24h":0,"known_merchants":[]},
    \\"merchant":{"id":"M","mcc":"5411","avg_amount":1.0},
    \\"terminal":{"is_online":false,"card_present":true,"km_from_home":0.0},
    \\"last_transaction":null}
;

test "parse with null last_transaction" {
    var s: ParseScratch = .{};
    const p = try parse(sample_null, &s);
    try std.testing.expect(p.last_tx_minutes == null);
    try std.testing.expect(p.last_tx_km == null);
}

test "parseDecimal2 covers the schema-allowed shapes" {
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), try parseDecimal2("0"), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), try parseDecimal2("1"), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 384.88), try parseDecimal2("384.88"), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 312.0), try parseDecimal2("312.00"), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 0.4), try parseDecimal2("0.4"), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, -1.5), try parseDecimal2("-1.5"), 1e-9);
    try std.testing.expectError(error.InvalidJson, parseDecimal2(""));
    try std.testing.expectError(error.InvalidJson, parseDecimal2("-"));
}

test "parseDecimal2 falls back for scientific notation" {
    // Schema doesn't produce these, but malformed adversarial input must not
    // crash; we delegate to the stdlib parser on any non-schema char.
    try std.testing.expectApproxEqAbs(@as(f64, 1e3), try parseDecimal2("1e3"), 1e-9);
}

test "parse populates merchant id hash and known hashes" {
    var s: ParseScratch = .{};
    const p = try parse(sample, &s);
    try std.testing.expectEqual(fnv1aU64("MERC-001"), p.merch_id_hash);
    try std.testing.expectEqual(@as(usize, 2), p.cust_known_hashes.len);
    try std.testing.expectEqual(fnv1aU64("MERC-100"), p.cust_known_hashes[0]);
    try std.testing.expectEqual(fnv1aU64("MERC-208"), p.cust_known_hashes[1]);
}
