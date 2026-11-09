# NALA-Colibri-RS (`v2.2.0-nala.6.6.6`) — English Documentation (`EN`)

[Back to Root README](../README.md) · [Deutsch](./README_DE.md) · [Français](./README_FR.md) · [Italiano](./README_IT.md) · [Rumantsch](./README_RM.md) · [5-Language License Summary](../LICENSE_SUMMARY_5LANG.md)

---

## Overview

**`NALA-Colibri-RS`** (`nala-colibri-rs` / `coli-rs`) is a **100% Pure Safe Rust (`Edition 2024`, `#![forbid(unsafe_code)]`, `0` external crates, Zero-Python, Zero-Node.js)** clean-room derivative fork of [`JustVugg/colibri`](https://github.com/JustVugg/colibri) (`v2.1.0`, Apache-2.0, Copyright 2026 Vincenzo Fornaro).

Frontier Mixture-of-Experts models (`GLM-5.3 744B`, `DeepSeek-V4.1-Flash 552B`, `Mistral-Large-4 1.02T`, `Kimi K3 2.8T`) activate only `top-k` routed experts per token. `NALA-Colibri-RS` orchestrates them across three memory tiers:
- **Tier 0 (`VRAM Hot`)**: Hot routed experts + shared trunk + `57x` compressed MLA latent KV cache on GPU.
- **Tier 1 (`Host RAM Warm`)**: Warm experts in pinned host memory governed by `.coli_usage` FNV-1a heat counters and RAII `ExpertLease<'a>` eviction locks.
- **Tier 2 (`Striped NVMe Cold`)**: Cold experts streamed across weighted dual-NVMe stripes with 1-layer `.coli_pairs` Markov lookahead prefetching.

## Essential Commands

```bash
nala-colibri-rs doctor --lang en
nala-colibri-rs plan glm-5.3 --eco-thermal
nala-colibri-rs decide --prompt "Approve verified [Dok-COLIBRI-01, S. 4]" --options "approve,defer,reject" --evidence-lock
nala-colibri-rs serve --bind 127.0.0.1:8000
nala-colibri-rs --bench --iters 10000
```

## Defensive Security Hardening (`SEC-COLIBRI-01..04`)
1. **`SEC-COLIBRI-01`**: Strict `10 MiB` body cap (`HTTP 413 Payload Too Large`) + loopback `Origin` CSRF guard.
2. **`SEC-COLIBRI-02`**: Single-resolution IP-pinned SSRF blocker (prevents `TTL=0` DNS-rebinding TOCTOU and blocks loopback, RFC1918, `169.254.169.254`, `100.64.0.0/10`, and IPv6 ULA).
3. **`SEC-COLIBRI-03`**: Constant-time `HMAC-SHA256` cluster registration & heartbeat verification.
4. **`SEC-COLIBRI-04`**: `64 MiB` Safetensors header ceiling with `checked_mul` shape overflow guards and canonical symlink-escape prevention.

## License
Licensed under the **Apache License, Version 2.0** ([LICENSE](../LICENSE), [NOTICE](../NOTICE), [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)).
