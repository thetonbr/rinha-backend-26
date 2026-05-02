test {
    _ = @import("vector/time.zig");
    _ = @import("vector/normalize.zig");
    _ = @import("vector/mcc.zig");
    _ = @import("vector/builder.zig");
    _ = @import("json/fraud_payload.zig");
    _ = @import("http/parser.zig");
    _ = @import("http/responses.zig");
    _ = @import("index/format.zig");
    _ = @import("index/ivf.zig");
    _ = @import("index/loader.zig");
    _ = @import("io/uring.zig");
    _ = @import("io/conn.zig");
    _ = @import("http/server.zig");
    // build_index/parser.zig lives outside src/'s module path and is exercised
    // separately via `zig test build_index/parser.zig` (it is a build-time
    // tool, not part of the api binary).
}
