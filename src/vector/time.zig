const std = @import("std");

pub const ParsedTime = struct {
    epoch_s: i64,
    hour: u8,
    day_of_week: u8,
};

pub fn parseIso8601(s: []const u8) !ParsedTime {
    if (s.len < 20) return error.InvalidFormat;
    // Format: YYYY-MM-DDTHH:MM:SSZ
    const year = try std.fmt.parseInt(u16, s[0..4], 10);
    const month = try std.fmt.parseInt(u8, s[5..7], 10);
    const day = try std.fmt.parseInt(u8, s[8..10], 10);
    const hour = try std.fmt.parseInt(u8, s[11..13], 10);
    const minute = try std.fmt.parseInt(u8, s[14..16], 10);
    const second = try std.fmt.parseInt(u8, s[17..19], 10);
    if (s[4] != '-' or s[7] != '-' or s[10] != 'T' or
        s[13] != ':' or s[16] != ':' or s[19] != 'Z') return error.InvalidFormat;

    const days_from_epoch = daysFromEpoch(year, month, day);
    const epoch_s: i64 = @as(i64, days_from_epoch) * 86400
        + @as(i64, hour) * 3600
        + @as(i64, minute) * 60
        + @as(i64, second);
    // Unix epoch (1970-01-01) was Thursday → day_of_week relative to Monday=0:
    // (days_from_epoch + 3) % 7 → Mon=0, Tue=1, ..., Sun=6
    const dow: u8 = @intCast(@mod(days_from_epoch + 3, 7));
    return .{ .epoch_s = epoch_s, .hour = hour, .day_of_week = dow };
}

fn daysFromEpoch(year: u16, month: u8, day: u8) i32 {
    // Howard Hinnant's days_from_civil: days since 1970-01-01 (proleptic Gregorian)
    const y: i32 = if (month <= 2) @as(i32, year) - 1 else @as(i32, year);
    const era = @divFloor(y, 400);
    const yoe: u32 = @intCast(y - era * 400);
    const doy: u32 = blk: {
        const m: i32 = if (month > 2) @as(i32, month) - 3 else @as(i32, month) + 9;
        break :blk @intCast(@divFloor(153 * m + 2, 5) + day - 1);
    };
    const doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + @as(i32, @intCast(doe)) - 719468;
}

test "parseIso8601 canonical" {
    const t = try parseIso8601("2025-09-22T19:24:51Z");
    try std.testing.expectEqual(@as(u8, 19), t.hour);
    try std.testing.expectEqual(@as(u8, 0), t.day_of_week); // 2025-09-22 is Monday
    try std.testing.expectEqual(@as(i64, 1758569091), t.epoch_s);
}

test "parseIso8601 sunday" {
    const t = try parseIso8601("2025-09-21T00:00:00Z");
    try std.testing.expectEqual(@as(u8, 0), t.hour);
    try std.testing.expectEqual(@as(u8, 6), t.day_of_week); // Sunday
}

test "parseIso8601 leap year" {
    const t = try parseIso8601("2024-02-29T12:00:00Z");
    try std.testing.expectEqual(@as(u8, 12), t.hour);
    try std.testing.expectEqual(@as(u8, 3), t.day_of_week); // Thursday
}
