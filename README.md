# NALA-Colibri-RS (`v2.4.0-nala.6.6.6`)

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

## What's New in `v2.4.0-nala.6.6.6` (Community & Hybrid Release)

1. **Upstream Top-Issue Bug-Killer Suite (`#2903`, `#2900`, `#2869`, `#2882`)**:
   - **Fixes `#2903` (13.4 GB MTP Rollback Snap OOM)**: Explicitly budgets progressive MTP rollback snapshot slots (`mtp_rollback_snap_gb = 13.4 GB` on `>= 250B` MoEs) and auto-gates MTP `OFF` before 48 GB hosts hit OOM.
   - **Fixes `#2900` (8 GB Discrete GPU Context Stall)**: Clamps automatic `--ctx` strictly from free GPU VRAM (`<= 35%` VRAM ceiling) on `<= 10 GB` discrete GPUs instead of host RAM, eliminating 1-token stalls.
   - **Fixes `#2869` (Client Disconnect KV-Slot Deadlock)**: Wraps every HTTP generation request in a compile-time Rust RAII `SlotLeaseGuard<'a>` whose `Drop` implementation frees the KV slot immediately on mid-stream TCP disconnect.
   - **Addresses `#2882` (Pre-AVX2 `SSE4.2/AVX1` Server Boost)**: Uses dual-`[f32; 8]` SIMD accumulators free of `libm fmaf` scalar traps.
2. **New `WEX4` / `WNR3` Compressed Expert Container & `AMD Quark MXFP4 (fmt 7)` Support (`#2884..#2897` & `#594`)**:
   - Native header & record verification for Colibri `WEX4` (`RAW` & `WNR3_FAST_SIMD`) with `COLI_WEX_VERIFY` SHA-256 attestation and catalog support for `amd/GLM-5.3-Flash-Quark-MXFP4` (`22` presets across 13 families).
3. **Debian/Ubuntu `.deb` Packages (`amd64` & `arm64` — Issue `#1852`)**:
   - Official zero-dependency `nala-colibri-rs_2.4.0_amd64.deb` and `nala-colibri-rs_2.4.0_arm64.deb` packages alongside standalone binaries for macOS, Linux, and Windows.
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

## Release Artifacts & Cryptographic `SHA-256` Hashes (`v2.4.0-nala.6.6.6`)

Verify offline on any platform via `nala-colibri-rs verify-sums SHA256SUMS.txt` or `nala-colibri-rs --self-check`:

| Artifact | Target Linkage / Format | `SHA-256` Checksum |
| :--- | :--- | :--- |
| `nala-colibri-rs-macos-arm64` | Mach-O `arm64` (`libSystem` only) | `be57f74bc2adef651a0f57f4ac3ca780321702f1d6ae605fbba9e7ff7097dd6c` |
| `nala-colibri-rs-macos-arm64.tar.gz` | Compressed Archive | `462087c01147cfcafb8741dec85e4d64b4289316c51e0b14494fb264d226d706` |
| `nala-colibri-rs-macos-x86_64` | Mach-O `x86_64` (`libSystem` only) | `331267cf313930a632c29515ec8391712467fc3368f10e4713abb657b007f5c7` |
| `nala-colibri-rs-macos-x86_64.tar.gz` | Compressed Archive | `35c985cf95fcd9835504b8f0af5c81892fae077e267ef787a122ddad2ac02373` |
| `nala-colibri-rs-linux-x86_64` | Static-PIE `musl` (`0` `.so` deps) | `bba7ab3f945ffee30181aab315a339c008cc08de6c74d14993097b6ff21bc8e3` |
| `nala-colibri-rs-linux-x86_64.tar.gz` | Compressed Archive | `3d69d1cd96e6649e9b26f4c02985601f2f6299aee25a5e5f9647f89791983c2a` |
| `nala-colibri-rs-linux-aarch64` | Static `musl` `aarch64` (`0` `.so` deps) | `36ae2528cd5de195430e9a0cf4b816753800bcc26df49f4bb8ae023f19ee5b1e` |
| `nala-colibri-rs-linux-aarch64.tar.gz` | Compressed Archive | `2a58d97753a83cccca1773488d41a7acaee50f55d5dd6fba4370e2673bf0f1c4` |
| `nala-colibri-rs_2.4.0_amd64.deb` | Debian/Ubuntu Package (`amd64`) | `8cb03b712eab4771f7d39a8f00972a8faf180374f1bc25c0955c08a944c1ef5e` |
| `nala-colibri-rs_2.4.0_arm64.deb` | Debian/Ubuntu Package (`arm64`) | `908f2403afcc166d2689f57db13304bed5667f187dabe02e574e9a73ebac5bb1` |
| `nala-colibri-rs-windows-x86_64.exe` | PE32+ `x86_64` (OS UCRT only) | `8246c34ee8939357402b16a2c78d54664c2b320f8f812fd9d53bc4d7567a4e0c` |
| `nala-colibri-rs-windows-x86_64.zip` | Compressed Archive | `3aa8834aad3ba364957f6b8f95b82c07308977eb860b8db3784b85937f1067f8` |
| `install.sh` | POSIX SHA-256 Verified Installer | `3a3885d1a2b91f860f5137f4183c82a2fa88d3061f15d7a68bd87bd007dc0987` |
| `install.ps1` | PowerShell SHA-256 Verified Installer | `6217bd5df1e538193705ab78eb5e31c181ac653654a72fa80186afd0d3f23ca2` |

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
