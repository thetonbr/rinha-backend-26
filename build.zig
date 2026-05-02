const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{
        .default_target = .{
            .cpu_arch = .x86_64,
            .os_tag = .linux,
            .abi = .musl,
            .cpu_model = .{ .explicit = &std.Target.x86.cpu.haswell },
        },
    });
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseFast });

    const exe = b.addExecutable(.{
        .name = "api",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .strip = true,
        .single_threaded = true,
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run the api binary");
    run_step.dependOn(&run_cmd.step);

    // build_index binary: offline IVF index builder. It lives in a separate
    // module tree (build_index/) and imports src/index/format.zig as the named
    // module `fmt`; Zig 0.13 disallows escaping a module's root with `..`,
    // so we wire format.zig in explicitly here.
    const fmt_module = b.createModule(.{
        .root_source_file = b.path("src/index/format.zig"),
    });

    const idx_exe = b.addExecutable(.{
        .name = "build_index",
        .root_source_file = b.path("build_index/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    idx_exe.root_module.addImport("fmt", fmt_module);
    b.installArtifact(idx_exe);

    const tests = b.addTest(.{
        .root_source_file = b.path("src/test_all.zig"),
        .target = target,
        .optimize = optimize,
    });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
}
