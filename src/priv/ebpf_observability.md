# eBPF Observability Platform

## Overview

Our infrastructure observability stack uses **eBPF** for kernel-level instrumentation. We run eBPF agents as DaemonSets across all Kubernetes nodes to capture network flows, file I/O, and system calls — without modifying application code.

## Architecture

- **eBPF Agent**: Pixie on each node — attaches kprobes, tracepoints, and uprobes
- **Data pipeline**: eBPF maps → ring buffer → OTel Collector → ClickHouse
- **Visualization**: Grafana dashboards with custom eBPF-derived metrics
- **Network observability**: Cilium/Hubble for service mesh visibility

## Key Use Cases

### 1. Network Flow Monitoring
- TCP connection tracking via `trace_tcp_connect` / `trace_tcp_close`
- Per-pod bandwidth, retransmit rates, connection churn
- No sidecars needed — works at kernel level via Cilium

### 2. Request Tracing (Zero-Code)
- Up到 `SSL_read` / `SSL_write` in OpenSSL to capture HTTP metadata
- Correlate PID ↔ Kubernetes pod ↔ HTTP trace via cgroup metadata
- Distributed traces without app instrumentation — covers 92% of HTTP traffic

### 3. File I/O Analysis
- `kprobe:vfs_read` / `kprobe:vfs_write` per node
- Detect write amplification, slow disk I/O, file contention
- Alert on `avg I/O latency > 100ms` per pod

## Instrumentation Points

```
Hook Type        Target                        Data
──────────       ──────                        ────
kprobe           tcp_connect                   src_ip:dst_ip:port
kprobe           tcp_close                     duration, bytes tx/rx
kprobe           vfs_read/vfs_write            fd, size, latency
uprobe           SSL_read/SSL_write            HTTP method, path, status
tracepoint       sys_enter_execve              process start/stop
```

## Performance Overhead

- CPU: < 3% per core on instrumented nodes
- Memory: ~150 MB per node for eBPF maps and ring buffers
- Network: zero additional wire traffic (all data stays local until scraped)

## Comparison

| Approach       | Code Changes | Coverage | Overhead | Latency Data |
| ────           | ──────       | ──────── | ──────── | ──────       |
| eBPF (Pixie)   | None         | 92%      | < 3%     | Kernel-level |
| OTel SDK       | Required     | 100%*    | 5-15%    | App-level    |
| Sidecar        | None         | 40-60%   | 8-20%    | Network-only |

*OTel SDK provides deeper app context but requires manual instrumentation

## Roadmap

- Q3 2026: Add USDT probes for PostgreSQL query tracing
- Q4 2026: eBPF-based continuous profiling (Parca/Pyroscope integration)
- Q1 2027: AI-driven anomaly detection on eBPF metric streams
