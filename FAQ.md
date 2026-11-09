# Frequently Asked Questions (`FAQ`) — `NALA-Colibri-RS`

## 1. How does `NALA-Colibri-RS` relate to `JustVugg/colibri`?
`NALA-Colibri-RS` (`v2.2.0-nala.6.6.6`) is an Apache-2.0 clean-room 100% Pure Safe Rust (`Edition 2024`, `#![forbid(unsafe_code)]`, zero external crates) derivative fork of [`JustVugg/colibri`](https://github.com/JustVugg/colibri) (`v2.1.0`, Copyright 2026 Vincenzo Fornaro). All upstream license and third-party attributions are preserved in [LICENSE](./LICENSE), [NOTICE](./NOTICE), and [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).

## 2. Why rewrite the 32,000+ line Python + Node.js control plane in Pure Rust?
- **Zero Interpreter & Dependency Bloat**: Eliminates Python 3.10+, pip wheels, Node.js/npm (`73 KB` `package-lock.json`), and login-shell subprocess calls.
- **Compile-Time Lease Safety**: Upstream C `expert_store.h` relies on manual `release()` calls; `NALA-Colibri-RS` enforces `ExpertLease<'a>` via Rust's `Drop` trait so an active expert slot can never be evicted while borrowed.
- **Bit-Exact IEEE-754 GEMV Parity**: `matmul_q_gs64_int8` and `matmul_q_gs64_int4` preserve exact 4-row group-64 `(gsum * scale)` reduction order so `TP=1` and `TP=N` produce identical `f32` bit patterns.

## 3. What security issues (`SEC-COLIBRI-01..04`) were fixed?
1. **`SEC-COLIBRI-01` (`HTTP 413` + Origin CSRF Shield)**: Caps request bodies at `10 MiB` (fixing unbounded `Content-Length` RAM allocation in upstream `ui/server.py`) and blocks cross-origin browser CSRF against `127.0.0.1`.
2. **`SEC-COLIBRI-02` (IP-Pinned SSRF Blocker)**: Resolves DNS once, rejects loopback/RFC1918/link-local (`169.254.169.254`)/CGNAT (`100.64.0.0/10`)/IPv6-ULA ranges, and pins the verified `SocketAddr` (eliminating `TTL=0` DNS-rebinding TOCTOU in upstream `ui/web.py`).
3. **`SEC-COLIBRI-03` (HMAC-SHA256 Cluster Authentication)**: Requires constant-time `X-NALA-Cluster-HMAC` verification on `/v1/cluster/register` and `/v1/cluster/heartbeat`.
4. **`SEC-COLIBRI-04` (`64 MiB` Header Cap + Canonical Path Guard)**: Lowers Safetensors header allocation cap from `512 MiB` to `64 MiB` with `checked_mul` overflow validation and blocks `..` / symlink escapes.

## 4. Why does Multi-Token Prediction (`MTP`) auto-disable below `88%` cache hit rate?
When cold NVMe misses exceed `12%`, speculative draft tokens trigger extra cold SSD expert reads that stall the main decode pipeline (causing up to `-32%` throughput regression in upstream benchmarks). `NALA-Colibri-RS` automatically gates MTP `OFF` whenever `learned_cache_hit_pct < 88.0%`.

## 5. What is `System One` (`POST /v1/systemone`)?
`System One` evaluates closed-set decisions (`choice`, `noul` yes/no, or `score` `0.0..1.0`) in a single calibrated pass with **0 generated tokens**, returning exact probabilities, normalized Shannon entropy confidence `1 - H(p)/ln(K)`, and optional deterministic `[Dok-ID, S. X]` citation verification (`--evidence-lock`).
