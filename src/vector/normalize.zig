const std = @import("std");

pub const max_amount: f32 = 10_000.0;
pub const max_installments: f32 = 12.0;
pub const max_minutes: f32 = 1_440.0;
pub const max_km: f32 = 1_000.0;
pub const max_tx_count_24h: f32 = 20.0;
pub const amount_vs_avg_ratio: f32 = 10.0;
pub const max_merchant_avg_amount: f32 = 10_000.0;

pub const inv_max_amount: f32 = 1.0 / max_amount;
pub const inv_max_installments: f32 = 1.0 / max_installments;
pub const inv_max_minutes: f32 = 1.0 / max_minutes;
pub const inv_max_km: f32 = 1.0 / max_km;
pub const inv_max_tx_count_24h: f32 = 1.0 / max_tx_count_24h;
pub const inv_amount_vs_avg_ratio: f32 = 1.0 / amount_vs_avg_ratio;
pub const inv_max_merchant_avg_amount: f32 = 1.0 / max_merchant_avg_amount;
pub const inv_23: f32 = 1.0 / 23.0;
pub const inv_6: f32 = 1.0 / 6.0;

pub inline fn clamp01(x: f32) f32 {
    return @max(0.0, @min(1.0, x));
}

test "clamp01 boundaries" {
    try std.testing.expectEqual(@as(f32, 0.0), clamp01(-0.5));
    try std.testing.expectEqual(@as(f32, 1.0), clamp01(1.5));
    try std.testing.expectEqual(@as(f32, 0.5), clamp01(0.5));
}

test "inverses match constants" {
    try std.testing.expectApproxEqRel(@as(f32, 1.0 / 10_000.0), inv_max_amount, 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, 1.0 / 23.0), inv_23, 1e-6);
}
