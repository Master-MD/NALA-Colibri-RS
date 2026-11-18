# NALA-Colibri-RS (`v2.5.0-nala.6.6.6`)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-10b981.svg)](./LICENSE)
[![Rust Edition 2024](https://img.shields.io/badge/Rust-Edition_2024_%7C_Zero_Unsafe-06b6d4.svg)](./Cargo.toml)
[![100% Standalone Universal](https://img.shields.io/badge/Standalone-Zero_Python_%7C_Zero_Node.js_%7C_0_Crates-f59e0b.svg)](./src/main.rs)

**Sprachen / Languages / Linguas:**  
[**Deutsch (`DE`)**](./docs/README_DE.md) · [**Français (`FR`)**](./docs/README_FR.md) · [**Italiano (`IT`)**](./docs/README_IT.md) · [**Rumantsch (`RM`)**](./docs/README_RM.md) · [**English (`EN`)**](./docs/README_EN.md) · [**5-Language License Summary**](./LICENSE_SUMMARY_5LANG.md) · [**60s Caveman Guide**](./CAVEMAN_GUIDE.md) · [**FAQ**](./FAQ.md)

---

**`NALA-Colibri-RS`** (`nala-colibri-rs` / alias `coli-rs`) is a **100% Standalone, Pure Safe Rust (`Edition 2024`, `#![forbid(unsafe_code)]`, `0` external crates, Zero-Python, Zero-Node.js, Zero-Telemetry)** hierarchical 3-tier Mixture-of-Experts (`MoE`) inference planner, SIMD kernel engine, `WEX4`/`GGUF`/`Safetensors`/`qpack` verifier, `System One` calibrated decision engine, built-in SHA-256 & collaborator security verifier, MCP server, and `sHeriFF`-hardened OpenAI/Anthropic gateway.

- **100% Universal & Standalone**: Runs natively under `env -i` on **any** vanilla Linux (`x86_64` from legacy pre-AVX2 `SSE4.2/AVX1` servers up to `AVX-512` & static `ARM64`), macOS (`ARM64` & `Intel Metal`), Windows 10/11 (`x86_64`), airgapped containers (`--network none`), and KVM/LXC hypervisors — **with zero external OS/daemon dependencies**.
- **Apache-2.0 Derivative Fork**: Clean-room Rust-Turbo rewrite of **[`JustVugg/colibri`](https://github.com/JustVugg/colibri)** (`v2.1.0+`, Copyright 2026 Vincenzo Fornaro — see [NOTICE](./NOTICE), [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md), and [README_UPSTREAM_COLIBRI.md](./README_UPSTREAM_COLIBRI.md)).

---

## What's New in `v2.5.0-nala.6.6.6`

0. **Wave-4 Performance & Vault Suite (`RSI-COLIBRI-013..016`)**:
   - **`RSI-COLIBRI-013` (`RegisteredNvmePageRing` QD=8 + `F_NOCACHE` — Issue `#441`)**: Pre-registered 4 KiB-aligned (`[u64; 512]`) 8-slot descriptor ring with strict 16 MiB per-slot byte quota and RAII `NvmeRingSlotLease`.
   - **`RSI-COLIBRI-014` (DietGPU-Style SIMD rANS `BF16` Exponent-Plane Codec `COLI_RNS` — Issue `#594`)**: Lossless exponent-plane compression verified by embedded 64-bit `FNV-1a` checksum.
   - **`RSI-COLIBRI-015` (Routing-Aware 16-Node `NMED` Speculative Tree — Issue `#689`)**: Vetoes cold-NVMe draft branches dynamically with `FP32` `total_cmp` tie-breaking and automatic `KvPrefixCache` taint-reset on rollback.
   - **`RSI-COLIBRI-016` (Persistent `HMAC-SHA256` Authenticated `COLI_PARK` Vault `COLI_PK1`)**: Instant (`< 4 ms`) restart resume of parked MLA sessions with constant-time `HMAC-SHA256` and session-ID isolation. (Community & Hybrid Release)

1. **Upstream Top-Issue Bug-Killer Suite (`#2903`, `#2900`, `#2869`, `#2882`)**:
   - **Fixes `#2903` (13.4 GB MTP Rollback Snap OOM)**: Explicitly budgets progressive MTP rollback snapshot slots (`mtp_rollback_snap_gb = 13.4 GB` on `>= 250B` MoEs) and auto-gates MTP `OFF` before 48 GB hosts hit OOM.
   - **Fixes `#2900` (8 GB Discrete GPU Context Stall)**: Clamps automatic `--ctx` strictly from free GPU VRAM (`<= 35%` VRAM ceiling) on `<= 10 GB` discrete GPUs instead of host RAM, eliminating 1-token stalls.
   - **Fixes `#2869` (Client Disconnect KV-Slot Deadlock)**: Wraps every HTTP generation request in a compile-time Rust RAII `SlotLeaseGuard<'a>` whose `Drop` implementation frees the KV slot immediately on mid-stream TCP disconnect.
   - **Addresses `#2882` (Pre-AVX2 `SSE4.2/AVX1` Server Boost)**: Uses dual-`[f32; 8]` SIMD accumulators free of `libm fmaf` scalar traps.
2. **New `WEX4` / `WNR3` Compressed Expert Container & `AMD Quark MXFP4 (fmt 7)` Support (`#2884..#2897` & `#594`)**:
   - Native header & record verification for Colibri `WEX4` (`RAW` & `WNR3_FAST_SIMD`) with `COLI_WEX_VERIFY` SHA-256 attestation and catalog support for `amd/GLM-5.3-Flash-Quark-MXFP4` (`22` presets across 13 families).
3. **Debian/Ubuntu `.deb` Packages (`amd64` & `arm64` — Issue `#1852`)**:
   - Official zero-dependency `nala-colibri-rs_2.5.0_amd64.deb` and `nala-colibri-rs_2.5.0_arm64.deb` packages alongside standalone binaries for macOS, Linux, and Windows.
4. **Bidirectional Public/Internal Sync & `sHeriFF` Collaborator Code-Gate (`sync-gate`)**:
   - `nala-colibri-rs sync-gate --audit-diff <patch_file>` (`RSI-COLIBRI-012`) audits unified GitHub collaborator PR diffs across all 10 `sHeriFF` & outbound symmetry gates (`< 1.2 ms`), alongside `WNR3_FAST_SIMD` block decompression (`RSI-COLIBRI-009`), `SharedSystemPrefixDedup` (`RSI-COLIBRI-010`), and 2-Step Markov Lookahead (`RSI-COLIBRI-011`).
   - `nala-colibri-rs sync-gate --direction from-public` audits external PRs/patches across 6 deterministic AST gates (`#![forbid(unsafe_code)]`, zero external crate dependencies, no `build.rs` hooks, no shell/reverse-TCP backdoors, no Trojan-Source BiDi characters, and Merkle `SHA-256` attestation).
   - `nala-colibri-rs sync-gate --direction to-public` verifies zero private IPs, zero local paths, and zero internal topology fingerprints before publishing.

---

## Multi-Platform Empirical Benchmarks (`10,000` Iterations, `env -i` Clean-Room)

| ID | Subsystem | Upstream Baseline (`Python` / `C -O3`) | ARM64 UMA Workstation (`macos-arm64`) | ARM64 Static Linux (`linux-aarch64`) | x86_64 AVX-512 UMA Node (`linux-x86_64`) | x86_64 Legacy AVX1/SSE4.2 (`linux-x86_64`) |
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

## Release Artifacts & Cryptographic `SHA-256` Hashes (`v2.5.0-nala.6.6.6`)

Verify offline on any platform via `nala-colibri-rs verify-sums SHA256SUMS.txt` or `nala-colibri-rs --self-check`:

| Artifact | Target Linkage / Format | `SHA-256` Checksum |
| :--- | :--- | :--- |
| `nala-colibri-rs-macos-arm64` | Mach-O `arm64` (`libSystem` only) | `50c34a8b2d287308396b58d10087e7d628bd6976771aa9a9253732327837dcfb` |
| `nala-colibri-rs-macos-arm64.tar.gz` | Compressed Archive | `34eaba2d0ab27766ce2a3f49829d7d81106201cbf539e31fa1b75b92a511d461` |
| `nala-colibri-rs-macos-x86_64` | Mach-O `x86_64` (`libSystem` only) | `f8f2c2173b527895327476428eb26b128db8b735799624bbfeddaea9dfd1b38d` |
| `nala-colibri-rs-macos-x86_64.tar.gz` | Compressed Archive | `2e1e8f55301442626e40a8ff09d2d3a60b78422007ee1b02df558bebfd5d4d29` |
| `nala-colibri-rs-linux-x86_64` | Static-PIE `musl` (`0` `.so` deps) | `978b29342177132b76100f978967b74e8bf2e603f560a476b1a258fb2cd96839` |
| `nala-colibri-rs-linux-x86_64.tar.gz` | Compressed Archive | `81d88b80086dcd3051998611dbc392f86f92052a3608a3663fce982b5197a66d` |
| `nala-colibri-rs-linux-aarch64` | Static `musl` `aarch64` (`0` `.so` deps) | `7a140612b847a4df18d447e9627a0208feeba2a15b933d81752342ebb282ad1c` |
| `nala-colibri-rs-linux-aarch64.tar.gz` | Compressed Archive | `e56cb6861a52faa02c455cfcc205d1aeec3fb744e681bc6818e855f9acd0314e` |
| `nala-colibri-rs_2.5.0_amd64.deb` | Debian/Ubuntu Package (`amd64`) | `03653aa624bebcae394a5e36bb53e991956833025eef99a1bf4dde49860bb2d9` |
| `nala-colibri-rs_2.5.0_arm64.deb` | Debian/Ubuntu Package (`arm64`) | `a563b6b415e51f89c0543f74f75985db0f2a517557847bd18098a2a5c6f919ad` |
| `nala-colibri-rs-windows-x86_64.exe` | PE32+ `x86_64` (OS UCRT only) | `7a0c0e54184af4a8aa76709f0683020cb94d45b502d7ffda1390e6a2ef9aead1` |
| `nala-colibri-rs-windows-x86_64.zip` | Compressed Archive | `2aa4bfbcca3b666c920ad504a461ba09dccedd15be4f11cfa0d67f58b8abc5b5` |
| `install.sh` | POSIX SHA-256 Verified Installer | `c2d8696cd99ad7d677c2aea4dd432fd09417bf769354c54480e25cdc93776246` |
| `install.ps1` | PowerShell SHA-256 Verified Installer | `8b818db8b78323e55afc2ab3519d120c663b9fb88914f3b9883db99a5a99c0d4` |

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
Upstream `Colibri` (`v2.1.0+`) Copyright 2026 **Vincenzo Fornaro** (`https://github.com/JustVugg/colibri`). See [NOTICE](./NOTICE), [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md), and [LICENSE_SUMMARY_5LANG.md](./LICENSE_SUMMARY_5LANG.md).
