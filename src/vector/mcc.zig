const std = @import("std");

const Entry = struct { mcc: u32, risk: f32 };

const entries = [_]Entry{
    .{ .mcc = 4511, .risk = 0.35 },
    .{ .mcc = 5311, .risk = 0.25 },
    .{ .mcc = 5411, .risk = 0.15 },
    .{ .mcc = 5812, .risk = 0.30 },
    .{ .mcc = 5912, .risk = 0.20 },
    .{ .mcc = 5944, .risk = 0.45 },
    .{ .mcc = 5999, .risk = 0.50 },
    .{ .mcc = 7801, .risk = 0.80 },
    .{ .mcc = 7802, .risk = 0.75 },
    .{ .mcc = 7995, .risk = 0.85 },
};

pub inline fn risk(mcc: u32) f32 {
    inline for (entries) |e| {
        if (e.mcc == mcc) return e.risk;
    }
    return 0.5;
}

pub inline fn parseMccString(s: []const u8) u32 {
    var v: u32 = 0;
    for (s) |c| v = v * 10 + (c - '0');
    return v;
}

test "known MCC returns mapped value" {
    try std.testing.expectEqual(@as(f32, 0.15), risk(5411));
    try std.testing.expectEqual(@as(f32, 0.85), risk(7995));
    try std.testing.expectEqual(@as(f32, 0.20), risk(5912));
    try std.testing.expectEqual(@as(f32, 0.30), risk(5812));
    try std.testing.expectEqual(@as(f32, 0.45), risk(5944));
    try std.testing.expectEqual(@as(f32, 0.80), risk(7801));
    try std.testing.expectEqual(@as(f32, 0.75), risk(7802));
    try std.testing.expectEqual(@as(f32, 0.35), risk(4511));
    try std.testing.expectEqual(@as(f32, 0.25), risk(5311));
    try std.testing.expectEqual(@as(f32, 0.50), risk(5999));
}

test "unknown MCC returns default 0.5" {
    try std.testing.expectEqual(@as(f32, 0.5), risk(1234));
}

test "parseMccString" {
    try std.testing.expectEqual(@as(u32, 5912), parseMccString("5912"));
}
