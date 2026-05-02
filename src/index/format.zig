const std = @import("std");

// In little-endian, "RNHA" = 'R'(52) 'N'(4E) 'H'(48) 'A'(41) → u32 0x41_48_4E_52
pub const MAGIC: u32 = 0x4148_4E52;
pub const VERSION: u32 = 1;
pub const DIM: u32 = 14;
pub const DIM_PADDED: u32 = 16;
pub const DEFAULT_NLIST: u32 = 2048;
pub const SCALE: u32 = 10000;

// The vectors block must start on a 64-byte boundary so the loader can expose
// it as []align(64) const i16 for AVX2 ymm loads. Builders must insert padding
// after invlist_offsets via std.mem.alignForward(usize, off, VECTORS_BLOCK_ALIGN).
pub const VECTORS_BLOCK_ALIGN: usize = 64;

pub const Header = extern struct {
    magic: u32,
    version: u32,
    n_vectors: u32,
    dim: u32,
    dim_padded: u32,
    nlist: u32,
    scale: u32,
    reserved: [36]u8,
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
