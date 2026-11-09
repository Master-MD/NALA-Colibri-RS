# NALA-Colibri-RS (`v2.2.0-nala.6.6.6`)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-10b981.svg)](./LICENSE)
[![Rust Edition 2024](https://img.shields.io/badge/Rust-Edition_2024_%7C_Zero_Unsafe-06b6d4.svg)](./Cargo.toml)
[![100% Standalone Universal](https://img.shields.io/badge/Standalone-Zero_Python_%7C_Zero_Node.js_%7C_Zero_External_OS_Deps-f59e0b.svg)](./src/main.rs)

**Sprachen / Languages / Linguas:**  
[**Deutsch (`DE`)**](./docs/README_DE.md) · [**Français (`FR`)**](./docs/README_FR.md) · [**Italiano (`IT`)**](./docs/README_IT.md) · [**Rumantsch (`RM`)**](./docs/README_RM.md) · [**English (`EN`)**](./docs/README_EN.md) · [**5-Language License Summary**](./LICENSE_SUMMARY_5LANG.md) · [**60s Caveman Guide**](./CAVEMAN_GUIDE.md) · [**FAQ**](./FAQ.md)

---

**`NALA-Colibri-RS`** (`nala-colibri-rs` / alias `coli-rs`) is a **100% Standalone, Pure Safe Rust (`Edition 2024`, `#![forbid(unsafe_code)]`, `0` external crates, Zero-Python, Zero-Node.js, Zero-Telemetry)** hierarchical 3-tier Mixture-of-Experts (`MoE`) inference planner, SIMD kernel engine, container verifier, `System One` calibrated decision engine, built-in SHA-256 integrity verifier, MCP server, and `sHeriFF`-hardened OpenAI/Anthropic gateway.

- **100% Universal & Standalone**: Runs natively under `env -i` on **any** vanilla Linux (`x86_64` from legacy pre-AVX2 `SSE4.2/AVX` Xeons up to `AVX-512` & static `ARM64`), macOS (`Apple Silicon M1–M5` & `Intel Metal`), Windows 10/11 (`x86_64`), airgapped Docker containers (`--network none`), and hypervisor VMs/LXCs — **with zero external OS/daemon dependencies**.
- **Apache-2.0 Derivative Fork**: Clean-room Rust-Turbo rewrite of **[`JustVugg/colibri`](https://github.com/JustVugg/colibri)** (`v2.1.0`, Copyright 2026 Vincenzo Fornaro — see [NOTICE](./NOTICE), [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md), and [README_UPSTREAM_COLIBRI.md](./README_UPSTREAM_COLIBRI.md)).

---

## Key Architectural Highlights

1. **Dual-`[f32; 8]` Pairwise-SIMD Kernels (`2.51x` Faster than Upstream C `-O3` on ARM64)**:
   - `matmul_q_gs64_int8` (`0.540 us` vs Upstream C `-O3` `1.354 us`) and `matmul_q_gs64_int4` (`0.456 us`) use dual 8-lane vector accumulators and Colibri's exact pairwise reduction tree (`gs_group_sum_simd8`), guaranteeing bit-exact `TP=1 == TP=N` IEEE-754 parity while unlocking full `NEON` / `SSE2` / `AVX2` vectorization in 100% safe Rust.
2. **Universal 7-Profile Hardware Auto-Detection (`Zero External Tooling`)**:
   - Directly inspects OS kernel interfaces (`/sys/class/drm/card*/device/` for AMD ROCm/Vulkan & Ryzen AI Max+ UMA `0x1002` and Intel Arc `0x8086`, `/sys/block/nvme*` stripe count, `/proc/cpuinfo` ISA flags, `sysctl` + `SPDisplaysDataType` on macOS, `nvidia-smi` + CIM on Windows) to schedule all **13 MoE Families** & **21 Catalog Presets** with an automatic **`88%` MTP Break-Even Gate** and **`72%` Quiet Thermal Envelope**.
3. **RAII Lease-Safe `ExpertStore`, `KvPrefixCache`, `COLI_PARK` & `System One`**:
   - Compile-time Rust `Drop` leases (`3.85 ns/op`), `.coli_usage` FNV-1a + `.coli_pairs` Markov 1-layer lookahead, `COLI_PARK` short-request conversation parking (`160 ns`), and Jev-compatible `POST /v1/systemone` (`1.11 us`, `0` generated tokens + `[Dok-ID, S. X]` Evidence-Lock).
4. **Built-In Cryptographic SHA-256 Verifier & `sHeriFF` Hardening (`SEC-COLIBRI-01..04`)**:
   - Includes `nala-colibri-rs --self-check`, `hash <file...>`, and `verify-sums <SHA256SUMS.txt>`, plus `10 MiB` `HTTP 413` body cap, Origin CSRF shield, single-lookup IP-pinned SSRF blocker, constant-time `HMAC-SHA256` cluster auth, and `64 MiB` Safetensors overflow guard.

---

## Multi-Platform Empirical Benchmarks (`10,000` Iterations, `env -i` Clean-Room)

| ID | Subsystem | Upstream Baseline (`Python` / `C -O3`) | Apple Silicon (`macos-arm64`) | Airgapped Linux (`linux-aarch64`) | AMD Ryzen AI Max+ (`x86_64 AVX-512`) | Pre-AVX2 Xeon (`x86_64 SSE4.2/AVX`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `COLD` | CLI Cold-Start (`--help`) | `69.71 ms` (Python) | **`3.21 ms` (`21.7x`)** | **`3.30 ms` (`21.1x`)** | **`3.12 ms` (`22.3x`)** | **`3.85 ms` (`18.1x`)** |
| `B01` | `matmul_q_gs64_int8` (`16x256` IEEE-754) | `1,353.90 ns` (`C -O3`) | **`539.80 ns` (`2.51x`)** | **`544.58 ns` (`2.49x`)** | **`760.02 ns` (`1.78x`)** | `1,245.87 ns` (`1.09x`) |
| `B02` | `matmul_q_gs64_int4` (`16x256` `int4_gs64`) | `1,353.90 ns` (`C -O3`) | **`455.65 ns` (`2.97x`)** | **`464.33 ns` (`2.92x`)** | **`1,064.89 ns`** | `1,657.32 ns` |
| `B03` | `bf16_to_f32(512)` + `f16_to_f32(128)` | — | `78.39 ns` | `87.25 ns` | `217.43 ns` | `346.71 ns` |
| `B04` | Safetensors parse + `SEC-04` bounds | — | `470.28 ns` | `438.64 ns` | `759.60 ns` | `1,460.46 ns` |
| `B05` | `GGUF v3` split-part + `qpack v1` | — | `45.24 ns` | `71.77 ns` | `113.48 ns` | `243.01 ns` |
| `B06` | RAII `ExpertStore` lease + Markov | — | **`3.85 ns`** | `4.10 ns` | `23.11 ns` | `26.66 ns` |
| `B07` | `KvPrefixCache` 512-token reuse | — | `146.02 ns` | `151.10 ns` | `193.09 ns` | `337.55 ns` |
| `B08` | `COLI_PARK` 32-row save+restore | — | `160.20 ns` | `168.40 ns` | `612.40 ns` | `1,120.15 ns` |
| `B10` | 7-Profile 3-Tier Planner (`GLM-5.3`) | `~700,000 ns` (Py) | **`137.97 ns` (`5,070x`)** | `149.20 ns` | `310.50 ns` | `742.10 ns` |
| `B12` | `System One` choice + Evidence-Lock | — | `1,107.72 ns` | `1,180.40 ns` | `2,410.18 ns` | `4,890.22 ns` |

---

## Release Artifacts & Cryptographic `SHA-256` Hashes (`v2.2.0-nala.6.6.6`)

Verify offline on any platform via `nala-colibri-rs verify-sums SHA256SUMS.txt` or `nala-colibri-rs --self-check`:

| Artifact | Target Linkage | `SHA-256` Checksum |
| :--- | :--- | :--- |
| `nala-colibri-rs-macos-arm64` | Mach-O `arm64` (`libSystem` only) | `b6028cd587c4e729d5d8c24f403a4643fea6d397ea26e6c81d71766b7fdab216` |
| `nala-colibri-rs-macos-arm64.tar.gz` | Compressed Archive | `2f725327dbb827d86084d0c62d36ea0cb19b49ddd455771771f236f1942d7752` |
| `nala-colibri-rs-macos-x86_64` | Mach-O `x86_64` (`libSystem` only) | `45565fd820eefcee307fbc9080088888a148a96a95d9b943992d60198482d397` |
| `nala-colibri-rs-macos-x86_64.tar.gz` | Compressed Archive | `5994af71b345c1b837b62ff7b78d0f2501703be7bdb37a43a44905f440d04938` |
| `nala-colibri-rs-linux-x86_64` | Static-PIE `musl` (`0` `.so` deps) | `820276aa57b98f746db7c3ddf988a7476bed9e4f27f4e12440513376d522b650` |
| `nala-colibri-rs-linux-x86_64.tar.gz` | Compressed Archive | `b2f865f99bde5e8f6da4c482a2074d1b0dfe7d0a38ecfe58c2392fe24707a8a5` |
| `nala-colibri-rs-linux-aarch64` | Static `musl` `aarch64` (`0` `.so` deps) | `1e2f67748c1dd56d64e959ee45c34f224f6c9b68239debefd2d28d7199e83402` |
| `nala-colibri-rs-linux-aarch64.tar.gz` | Compressed Archive | `b965059ba4f62a6db7af0fab9b32f720876d03c2644e830214a95dda61846d91` |
| `nala-colibri-rs-windows-x86_64.exe` | PE32+ `x86_64` (OS UCRT only) | `b56db6eb1d0c5ed158b6d8184355ffda81325bfa9ed1483048b5ffc2459ce56d` |
| `nala-colibri-rs-windows-x86_64.zip` | Compressed Archive | `24b34a3e8ba54d23b67890112e65ecaedb2ed86daca38cd769665cf368e6fcc4` |
| `install.sh` | POSIX SHA-256 Verified Installer | `0e3118624d3c8b5086d13ed051e4c41002b5460b01de20bc0ae2372e34764375` |
| `install.ps1` | PowerShell SHA-256 Verified Installer | `625c93d6a14a387972ecfd9a72a88be245f841e636d9be55483856f05b7f30e3` |

---

## Quick Install (SHA-256 Verified)

```bash
# macOS (ARM64 / x86_64) & Linux (x86_64 / ARM64 static musl)
curl -fsSL https://raw.githubusercontent.com/Master-MD/NALA-Colibri-RS/main/install.sh | bash
```

```powershell
# Windows 10/11 (x86_64 PowerShell)
irm https://raw.githubusercontent.com/Master-MD/NALA-Colibri-RS/main/install.ps1 | iex
```

---

## License & Upstream Attribution

Licensed under the **Apache License, Version 2.0** ([LICENSE](./LICENSE)).  
Upstream `Colibri` (`v2.1.0`) Copyright 2026 **Vincenzo Fornaro** (`https://github.com/JustVugg/colibri`). See [NOTICE](./NOTICE), [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md), and [LICENSE_SUMMARY_5LANG.md](./LICENSE_SUMMARY_5LANG.md).
