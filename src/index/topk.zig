const std = @import("std");

pub const KPRIME: usize = 50;
pub const K: usize = 5;

pub const TopKp = struct {
    dist: [KPRIME]i64 = .{std.math.maxInt(i64)} ** KPRIME,
    id: [KPRIME]u32 = .{0} ** KPRIME,
    size: usize = 0,

    pub inline fn maybeInsert(self: *TopKp, d: i64, id: u32) void {
        if (self.size < KPRIME) {
            self.dist[self.size] = d;
            self.id[self.size] = id;
            self.size += 1;
            if (self.size == KPRIME) self.heapify();
            return;
        }
        if (d >= self.dist[0]) return;
        self.dist[0] = d;
        self.id[0] = id;
        siftDown(&self.dist, &self.id, 0, KPRIME);
    }

    fn heapify(self: *TopKp) void {
        var i: isize = @as(isize, KPRIME / 2) - 1;
        while (i >= 0) : (i -= 1) {
            siftDown(&self.dist, &self.id, @intCast(i), KPRIME);
        }
    }
};

fn siftDown(d: *[KPRIME]i64, id: *[KPRIME]u32, root: usize, n: usize) void {
    var i = root;
    while (true) {
        const l = 2 * i + 1;
        const r = 2 * i + 2;
        var max = i;
        if (l < n and d[l] > d[max]) max = l;
        if (r < n and d[r] > d[max]) max = r;
        if (max == i) return;
        const td = d[i]; d[i] = d[max]; d[max] = td;
        const tid = id[i]; id[i] = id[max]; id[max] = tid;
        i = max;
    }
}

pub const Top5 = struct {
    dist: [K]f32 = .{std.math.inf(f32)} ** K,
    label: [K]u8 = .{0} ** K,

    pub inline fn maybeInsert(self: *Top5, d: f32, label: u8) void {
        if (d >= self.dist[K - 1]) return;
        var pos: usize = K - 1;
        while (pos > 0 and d < self.dist[pos - 1]) : (pos -= 1) {
            self.dist[pos] = self.dist[pos - 1];
            self.label[pos] = self.label[pos - 1];
        }
        self.dist[pos] = d;
        self.label[pos] = label;
    }

    pub fn fraudCount(self: Top5) u8 {
        var c: u8 = 0;
        for (self.label) |l| c += l;
        return c;
    }
};

test "Top5 insertion preserves order" {
    var t: Top5 = .{};
    t.maybeInsert(5.0, 1);
    t.maybeInsert(1.0, 0);
    t.maybeInsert(3.0, 1);
    t.maybeInsert(0.5, 0);
    t.maybeInsert(2.0, 1);
    try std.testing.expectEqual(@as(f32, 0.5), t.dist[0]);
    try std.testing.expectEqual(@as(f32, 5.0), t.dist[4]);
    try std.testing.expectEqual(@as(u8, 3), t.fraudCount());
}

test "Top5 rejects worse than worst" {
    var t: Top5 = .{};
    inline for (0..5) |i| t.maybeInsert(@as(f32, @floatFromInt(i)), 0);
    t.maybeInsert(99.0, 1);
    try std.testing.expectEqual(@as(f32, 4.0), t.dist[4]);
    try std.testing.expectEqual(@as(u8, 0), t.fraudCount());
}

test "TopKp accepts up to KPRIME and rejects worse" {
    var t: TopKp = .{};
    for (0..100) |i| t.maybeInsert(@as(i64, @intCast(i)), @as(u32, @intCast(i)));
    try std.testing.expectEqual(@as(usize, KPRIME), t.size);
    var max: i64 = 0;
    for (t.dist) |d| { if (d > max) max = d; }
    try std.testing.expect(max < 50);
}
