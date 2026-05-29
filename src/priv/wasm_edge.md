# WebAssembly Edge Architecture

## Overview

Our edge compute platform uses **WebAssembly (WASM)** for function execution at the edge. We run on Cloudflare Workers with WASM-based functions for request processing, authentication, and response transformation.

## Motivation

- **Cold start**: WASM modules start in < 1ms vs 50-200ms for Node.js workers
- **Multi-language**: Teams write in Rust, Go, or AssemblyScript — same runtime
- **Sandboxing**: Each WASM instance is fully isolated by default
- **Determinism**: Same binary runs identically in dev, CI, and prod

## Architecture

- **Runtime**: Wasmtime-based Workers runtime with WASI Preview 2
- **Functions**:
  - `auth-check` (Rust) — validates JWT tokens at the edge
  - `image-transform` (Go) — resizes/optimizes images on-the-fly
  - `api-router` (AssemblyScript) — routes requests to origin services
- **Orchestration**: Components connected via WASM Component Model interfaces

## Performance

```
Metric          WASM        Node.js     Improvement
──────────      ────        ──────      ──────────
Cold start      0.3 ms      85 ms       280x
P50 latency     1.2 ms      8.4 ms      7x
P99 latency     4.1 ms      45 ms       11x
Memory/call     2.1 MB      18 MB       8.5x
```

## Deployment

- **CI pipeline**: Build WASM targets via `cargo component` / `tinygo`
- **Registry**: WASM modules pushed to `oci://registry.edge.internal/wasm/`
- **Release**: Workers deploy via `wrangler` with canary (10% → 50% → 100%)

## Risks

- WASI networking is still evolving — HTTP client support differs across runtimes
- Debugging WASM modules requires DWARF symbol support (limited tooling)
- Some Go standard library features (net, os) have incomplete WASI support

## Migration Path

1. **Phase 1** (Q1 2026): Port `auth-check` from JS to Rust WASM — perf-critical path
2. **Phase 2** (Q2 2026): Port `image-transform` to Go WASM — reduces origin load
3. **Phase 3** (Q3 2026): All edge functions in WASM, decommission JS workers
