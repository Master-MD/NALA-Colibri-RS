# NALA-Colibri-RS (`v3.0.0-nala.6.6.6`)

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

## What's New in `v3.0.0-nala.6.6.6`

0000. **Wave-8 Gated-DeltaNet SIMD Kernel, Pinned Shared-Expert Fast-Path, Zero-Alloc JSON Grammar & Per-Tenant Burst Limiter (`RSI-COLIBRI-029..032`)**:
   - **`RSI-COLIBRI-029` (`deltanet_step_simd8`)**: Fused `O(1)` constant-memory beta-gated Householder delta rule `S_t = alpha*S_{t-1} + beta*(v - alpha*(S_{t-1}*k_hat))*k_hat^T` + query read-out in a single L1-resident row pass with spectral stability & `NaN`/`Inf` guards.
   - **`RSI-COLIBRI-030` (`SharedExpertFastPath`)**: Immutable L2-pinned fast-path bypass for always-active shared MoE experts (`-100%` LRU/atomic lookup overhead).
   - **`RSI-COLIBRI-031` (`JsonGrammarStateMask`)**: Stack-allocated pushdown automaton (`depth <= 32`) guaranteeing 100% valid JSON on structured outputs.
   - **`RSI-COLIBRI-032` (`LoopbackTokenBucketLimiter` / `SEC-COLIBRI-07`)**: Per-tenant token-bucket burst & rate limiter protecting the 16 concurrent KV slots.

000. **Wave-7 Block-Sparse NSA/DSA Lightning Indexer, Zero-Alloc Stack SSE Streamer, Sigmoid No-Aux MoE Router & Tensor Canary Guard (`RSI-COLIBRI-025..028`)**:
   - **`RSI-COLIBRI-025` (`nsa_topk_blocks`)**: Hardware-aligned 64-token block importance scorer for 128k Native/Dynamic Sparse Attention (`DeepSeek-V4.1` & `MiMo-V2.6-Pro`) with strict causal mask boundary enforcement (`b * 64 <= current_pos`) and deterministic `FP32 total_cmp` tie-breaking.
   - **`RSI-COLIBRI-026` (`ZeroAllocSseFrame`)**: Fixed `[u8; 512]` stack-allocated OpenAI & Anthropic SSE token chunk formatter (`0` per-token heap allocations).
   - **`RSI-COLIBRI-027` (`route_topk_sigmoid_noaux`)**: Auxiliary-loss-free sigmoid affinity MoE router with per-expert bias clamping.
   - **`RSI-COLIBRI-028` (`verify_tensor_canaries`)**: `< 180 us` head/mid/tail 64-byte `FNV-1a` tensor canary integrity verifier.

00. **Wave-6 MLA `57x` Fused RoPE Kernel, Multi-Shard Merkle Verifier, Evidence-Pinned Context Governor & Anti-Replay Ring (`RSI-COLIBRI-021..024`)**:
   - **`RSI-COLIBRI-021` (`matmul_mla_latent_57x`)**: Fused Low-Rank Compressed KV-Latent (`c_KV`) SIMD up-projection + register-resident 2D RoPE phase rotation (`TP=1/2/4/8` bit-identical).
   - **`RSI-COLIBRI-022` (`verify_multi_shard_continuity`)**: Multi-shard split GGUF/WEX4 continuity (`00001-of-0000N..0000N-of-0000N`), 64-byte alignment & Merkle `SHA-256` pre-flight check.
   - **`RSI-COLIBRI-023` (`trim_context_with_evidence_pinning`)**: Dynamic middle-out context budget governor that unconditionally pins `[Dok-ID, S. X]` Evidence-Lock citations.
   - **`RSI-COLIBRI-024` (`ClusterAntiReplayRing` / `SEC-COLIBRI-06`)**: 64-entry sliding-window nonce + `±60s` timestamp drift + constant-time `HMAC-SHA256` cluster anti-replay guard.

0. **Wave-5 Quark MXFP4 SIMD, Thermal Balancer, Batch System One & SLSA SBOM Suite (`RSI-COLIBRI-017..020`)**:
   - **`RSI-COLIBRI-017` (`matmul_q_mxfp4_fmt7` — Issue `#2895`)**: 4-row interleaved OCP `MXFP4` (`E2M1` LUT + 1-cycle `E8M0` shared exponent shift) GEMV with order-preserving `gs_group_sum_simd8` tree reduction (`TP=1/2/4/8` bit-identical).
   - **`RSI-COLIBRI-018` (`AdaptiveNvmeStripeBalancer`)**: `62°C..68°C` thermal hysteresis + EMA latency stripe balancer across dual NVMe drives.
   - **`RSI-COLIBRI-019` (`POST /v1/systemone/batch`)**: Zero-token multi-hypothesis batch evaluator with per-hypothesis `[Dok-ID, S. X]` Evidence-Lock verification.
   - **`RSI-COLIBRI-020` (`nala-colibri-rs provenance dist/SHA256SUMS.txt`)**: Deterministic SLSA v1.0 & CycloneDX Zero-Dependency SBOM generator signed with `HMAC-SHA256`.

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
   - Official zero-dependency `nala-colibri-rs_3.0.0_amd64.deb` and `nala-colibri-rs_3.0.0_arm64.deb` packages alongside standalone binaries for macOS, Linux, and Windows.
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

## Release Artifacts & Cryptographic `SHA-256` Hashes (`v3.0.0-nala.6.6.6`)

Verify offline on any platform via `nala-colibri-rs verify-sums SHA256SUMS.txt` or `nala-colibri-rs --self-check`:

| Artifact | Target Linkage / Format | `SHA-256` Checksum |
| :--- | :--- | :--- |
| `nala-colibri-rs-macos-arm64` | Mach-O `arm64` (`libSystem` only) | `c7832fdd36f3a82a072bfe3fda291ecd6cb867e64093ae6b640fd391deda346c` |
| `nala-colibri-rs-macos-arm64.tar.gz` | Compressed Archive | `27cc739a737f31b0a2bd7fcc7db2d6084f577cbe625e5313147f343e5eddfba1` |
| `nala-colibri-rs-macos-x86_64` | Mach-O `x86_64` (`libSystem` only) | `274cc60cac927af2ddc0c689af8a8d92f30fe8c6025637c1035e728cfd758955` |
| `nala-colibri-rs-macos-x86_64.tar.gz` | Compressed Archive | `4fcdf70a3d5c93397d0b8980651530e299a4092d8fed063d4f0d02c7545ab941` |
| `nala-colibri-rs-linux-x86_64` | Static-PIE `musl` (`0` `.so` deps) | `eb257973590bfa3ea1d9776216051250f44faa8822aa77da296a9534b397720e` |
| `nala-colibri-rs-linux-x86_64.tar.gz` | Compressed Archive | `cfb6bfd2cd8a246b51318402df136e544e39a97442e686b8012e596550ac4ede` |
| `nala-colibri-rs-linux-aarch64` | Static `musl` `aarch64` (`0` `.so` deps) | `924af3e14ab6392984859111ddaf55991d58a2137bd80830b238e5e6b11b0cb7` |
| `nala-colibri-rs-linux-aarch64.tar.gz` | Compressed Archive | `dec6408aa055b755ed082b5c81cfcd294255c921345e5b6960829ee5b08fb4ca` |
| `nala-colibri-rs_3.0.0_amd64.deb` | Debian/Ubuntu Package (`amd64`) | `d1639529734662d0373fb845ee9f7771e9317a27ade1360068c7de93a331446c` |
| `nala-colibri-rs_3.0.0_arm64.deb` | Debian/Ubuntu Package (`arm64`) | `770f593d2cc4d41efcfb7738e5823da485926fef1df36e1c32b3f19c655fc3a9` |
| `nala-colibri-rs-windows-x86_64.exe` | PE32+ `x86_64` (OS UCRT only) | `7e74313c41b54095030582a2b60a1a124b9854fa94d485bf1aa11f7bb6f38dca` |
| `nala-colibri-rs-windows-x86_64.zip` | Compressed Archive | `a125cad8c6f5445c809b54e87afdd93bf3ca2597d78e103875b096d50d8e8dc8` |
| `install.sh` | POSIX SHA-256 Verified Installer | `683abaa145ae5c6c1b7fb647abc6d341e3c13b6153bb28532bcd5a2e90463753` |
| `install.ps1` | PowerShell SHA-256 Verified Installer | `2d30f1ff00ef515274b33471d1c9380b1756dddc796b1f32df6ff921783609dc` |

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
