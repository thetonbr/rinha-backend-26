const std = @import("std");

const Entry = struct { mcc: u32, risk: f32 };

const entries = [_]Entry{
    .{ .mcc = 5411, .risk = 0.15 },
    .{ .mcc = 5811, .risk = 0.20 },
    .{ .mcc = 5912, .risk = 0.42 },
    .{ .mcc = 5942, .risk = 0.30 },
    .{ .mcc = 5999, .risk = 0.50 },
    .{ .mcc = 6011, .risk = 0.55 },
    .{ .mcc = 7011, .risk = 0.40 },
    .{ .mcc = 7372, .risk = 0.45 },
    .{ .mcc = 7995, .risk = 0.85 },
    .{ .mcc = 9999, .risk = 0.65 },
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
}

test "unknown MCC returns default 0.5" {
    try std.testing.expectEqual(@as(f32, 0.5), risk(1234));
}

test "parseMccString" {
    try std.testing.expectEqual(@as(u32, 5912), parseMccString("5912"));
}
