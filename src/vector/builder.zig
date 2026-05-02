const std = @import("std");
const norm = @import("normalize.zig");
const mcc_mod = @import("mcc.zig");
const time = @import("time.zig");

pub const PayloadValues = struct {
    amount: f32,
    installments: u8,
    requested_at_iso: []const u8,
    cust_avg_amount: f32,
    cust_tx_count_24h: u32,
    cust_known_merchants: []const []const u8,
    merch_id: []const u8,
    merch_mcc: u32,
    merch_avg_amount: f32,
    term_is_online: bool,
    term_card_present: bool,
    term_km_from_home: f32,
    last_tx_minutes: ?f32,
    last_tx_km: ?f32,
};

pub const SCALE: i16 = 10000;
pub const DIM: usize = 14;
pub const DIM_PADDED: usize = 16;

fn quantize(v: f32) i16 {
    // @max/@min propagate NaN, and @intFromFloat(NaN) is UB in ReleaseFast.
    // Defensive: collapse non-finite inputs to 0 before scaling.
    if (std.math.isNan(v) or std.math.isInf(v)) return 0;
    const x = v * @as(f32, @floatFromInt(SCALE));
    const clamped = @max(-32768.0, @min(32767.0, x));
    return @intFromFloat(@round(clamped));
}

pub fn build(p: PayloadValues, out: *[DIM_PADDED]i16) !void {
    const t = try time.parseIso8601(p.requested_at_iso);

    out[0] = quantize(norm.clamp01(p.amount * norm.inv_max_amount));
    out[1] = quantize(norm.clamp01(@as(f32, @floatFromInt(p.installments)) * norm.inv_max_installments));
    const ratio = if (p.cust_avg_amount > 0) p.amount / p.cust_avg_amount else 0.0;
    out[2] = quantize(norm.clamp01(ratio * norm.inv_amount_vs_avg_ratio));
    out[3] = quantize(@as(f32, @floatFromInt(t.hour)) * norm.inv_23);
    out[4] = quantize(@as(f32, @floatFromInt(t.day_of_week)) * norm.inv_6);
    if (p.last_tx_minutes) |m| {
        out[5] = quantize(norm.clamp01(m * norm.inv_max_minutes));
    } else {
        out[5] = -SCALE;
    }
    if (p.last_tx_km) |km| {
        out[6] = quantize(norm.clamp01(km * norm.inv_max_km));
    } else {
        out[6] = -SCALE;
    }
    out[7] = quantize(norm.clamp01(p.term_km_from_home * norm.inv_max_km));
    out[8] = quantize(norm.clamp01(@as(f32, @floatFromInt(p.cust_tx_count_24h)) * norm.inv_max_tx_count_24h));
    out[9] = if (p.term_is_online) SCALE else 0;
    out[10] = if (p.term_card_present) SCALE else 0;
    out[11] = SCALE;
    for (p.cust_known_merchants) |m| {
        if (std.mem.eql(u8, m, p.merch_id)) {
            out[11] = 0;
            break;
        }
    }
    out[12] = quantize(mcc_mod.risk(p.merch_mcc));
    out[13] = quantize(norm.clamp01(p.merch_avg_amount * norm.inv_max_merchant_avg_amount));
    out[14] = 0;
    out[15] = 0;
}

test "build basic payload yields expected vector" {
    var out: [DIM_PADDED]i16 align(64) = undefined;
    const known: [1][]const u8 = .{"MERC-100"};
    try build(.{
        .amount = 384.88,
        .installments = 3,
        .requested_at_iso = "2025-09-22T19:24:51Z",
        .cust_avg_amount = 230.50,
        .cust_tx_count_24h = 3,
        .cust_known_merchants = known[0..],
        .merch_id = "MERC-001",
        .merch_mcc = 5912,
        .merch_avg_amount = 312.0,
        .term_is_online = true,
        .term_card_present = false,
        .term_km_from_home = 13.7,
        .last_tx_minutes = 54.85,
        .last_tx_km = 0.4,
    }, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 385.0), @as(f32, @floatFromInt(out[0])), 2.0);
    try std.testing.expectEqual(@as(i16, SCALE), out[9]);
    try std.testing.expectEqual(@as(i16, 0), out[10]);
    try std.testing.expectEqual(@as(i16, SCALE), out[11]);
    try std.testing.expectEqual(@as(i16, 0), out[14]);
    try std.testing.expectEqual(@as(i16, 0), out[15]);
}

test "build with null last_transaction uses sentinel" {
    var out: [DIM_PADDED]i16 align(64) = undefined;
    try build(.{
        .amount = 100.0, .installments = 1, .requested_at_iso = "2025-09-22T19:00:00Z",
        .cust_avg_amount = 100.0, .cust_tx_count_24h = 0, .cust_known_merchants = &.{},
        .merch_id = "X", .merch_mcc = 5411, .merch_avg_amount = 100.0,
        .term_is_online = false, .term_card_present = true, .term_km_from_home = 0.0,
        .last_tx_minutes = null, .last_tx_km = null,
    }, &out);
    try std.testing.expectEqual(@as(i16, -10000), out[5]);
    try std.testing.expectEqual(@as(i16, -10000), out[6]);
}
