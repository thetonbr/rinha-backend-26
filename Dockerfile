# syntax=docker/dockerfile:1.6
#
# Multi-stage build for the Rinha de Backend 2026 entry.
#
#   1. zig            — fetch the Zig 0.13 toolchain (cacheable base).
#   2. builder        — compile both `api` and `build_index` via build.zig.
#   3. index-builder  — download references.json.gz and run build_index to
#                       produce /index/index.bin offline at image build time.
#   4. api (final)    — FROM scratch image with only the api binary and the
#                       prebuilt index. Target size ~88 MB.

# ---------------------------------------------------------------------------
# Stage 1: Zig toolchain (shared by builder and index-builder).
# ---------------------------------------------------------------------------
FROM alpine:3.19 AS zig
ARG ZIG_VERSION=0.13.0
RUN apk add --no-cache curl xz ca-certificates
RUN curl -fsSL -o /tmp/zig.tar.xz \
        https://ziglang.org/download/${ZIG_VERSION}/zig-linux-x86_64-${ZIG_VERSION}.tar.xz \
 && mkdir -p /opt/zig \
 && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 \
 && ln -s /opt/zig/zig /usr/local/bin/zig \
 && rm -f /tmp/zig.tar.xz

# ---------------------------------------------------------------------------
# Stage 2: build both binaries (api + build_index).
# We invoke `zig build` so build.zig wires the `fmt` module that build_index
# imports — `zig build-exe` standalone would skip that wiring and fail.
# ---------------------------------------------------------------------------
FROM zig AS builder
WORKDIR /src
COPY build.zig ./
COPY src/ ./src/
COPY build_index/ ./build_index/
RUN zig build --release=fast
RUN ls -la zig-out/bin/api zig-out/bin/build_index

# ---------------------------------------------------------------------------
# Stage 3: run build_index against the references dataset to produce the
# binary IVF index consumed by the api at runtime.
# ---------------------------------------------------------------------------
FROM zig AS index-builder
ARG REFS_URL=https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources/references.json.gz
ARG NLIST=512
WORKDIR /work
COPY --from=builder /src/zig-out/bin/build_index /usr/local/bin/build_index
RUN curl -fsSL -o /tmp/refs.json.gz "${REFS_URL}" \
 && mkdir -p /index \
 && build_index /tmp/refs.json.gz /index/index.bin ${NLIST} \
 && rm -f /tmp/refs.json.gz

# ---------------------------------------------------------------------------
# Stage 4: minimal final image — scratch + api binary + prebuilt index.
# ---------------------------------------------------------------------------
FROM scratch AS api
COPY --from=builder /src/zig-out/bin/api /api
COPY --from=index-builder /index/index.bin /index/index.bin
EXPOSE 8080
ENTRYPOINT ["/api"]
