const std = @import("std");

// In little-endian, "RNHA" = 'R'(52) 'N'(4E) 'H'(48) 'A'(41) → u32 0x41_48_4E_52
pub const MAGIC: u32 = 0x4148_4E52;
// V2: SoA layout (one i16 array per dim) + bbox tables per cluster + orig_ids,
// designed for per-dim early-exit scan with NPROBE=1 + bbox repair.
pub const VERSION: u32 = 2;
pub const DIM: u32 = 14;
pub const DEFAULT_NLIST: u32 = 256;
pub const SCALE: u32 = 10000;

// Each per-dim block (and the orig_ids block) starts on a 64-byte boundary so
// the loader can expose it as []align(64) const i16 / const u32 with AVX2-safe
// loads. Builders must insert padding via std.mem.alignForward.
pub const BLOCK_ALIGN: usize = 64;

pub const Header = extern struct {
    magic: u32,
    version: u32,
    n_vectors: u32,
    dim: u32,
    nlist: u32,
    scale: u32,
    reserved: [40]u8,
};

comptime {
    std.debug.assert(@sizeOf(Header) == 64);
}

test "Header is 64 bytes" {
    try std.testing.expectEqual(@as(usize, 64), @sizeOf(Header));
}

test "MAGIC bytes spell RNHA" {
    const bytes = std.mem.toBytes(MAGIC);
    try std.testing.expectEqualStrings("RNHA", bytes[0..4]);
}
