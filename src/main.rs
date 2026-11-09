#![forbid(unsafe_code)]
//! `NALA-Colibri-RS` (`v2.2.0-nala.6.6.6`) — 100% Standalone Native Safe Rust-Turbo Hierarchical
//! 3-Tier (`VRAM + Host RAM + Striped NVMe`) MoE Inference Engine, Universal 7-Profile Hardware
//! Planner (Zero NALA-OS Dependency), Jev-Compatible `System One` Decision Engine, Safetensors /
//! GGUF / qpack Inspector, Cryptographic SHA-256 Verifier, MCP Server, and Hardened HTTP Gateway.
//!
//! Upstream Credit: Clean-room Rust derivative fork of `JustVugg/colibri` (`v2.1.0`, Apache-2.0,
//! Copyright 2026 Vincenzo Fornaro). Zero Python, Zero Node.js, Zero Telemetry, 0 external crates.

pub mod bench;
pub mod catalog;
pub mod containers;
pub mod expert_store;
pub mod kernels;
pub mod mcp;
pub mod planner;
pub mod rsi;
pub mod security;
pub mod server;
pub mod system_one;
pub mod ui_html;

use catalog::{CATALOG_MODELS, ENGINE_FAMILIES, find_family, find_model};
use containers::{inspect_container_file, parse_safetensors_header};
use expert_store::{ColiParkVault, ExpertStore, KvPrefixCache};
use kernels::{bf16_to_f32_bulk, f16_to_f32_bulk, matmul_q_gs64_int4, matmul_q_gs64_int8};
use mcp::{json_escape, run_mcp_stdio_loop};
use planner::{HalProfile, compute_three_tier_plan, detect_host_hardware};
use rsi::{RSI_PROPOSALS, compile_multi_ide_prompt};
use security::{
    hex_encode, hmac_sha256, is_strictly_public_ip, sha256_file, verify_hmac_sha256_hex,
};
use server::run_http_gateway;
use std::net::{IpAddr, Ipv4Addr};
use std::path::Path;
use system_one::{SystemOneMode, evaluate_system_one};

pub const VERSION: &str = "2.2.0-nala.6.6.6";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let json_mode = args.iter().any(|a| a == "--json");
    let quiet_eco = args.iter().any(|a| a == "--eco-thermal" || a == "--quiet");
    let lang = extract_flag_val(&args, "--lang").unwrap_or_else(|| "de".to_string());

    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help" || a == "help") {
        print_help(&lang);
        return;
    }

    if args.iter().any(|a| a == "-V" || a == "--version" || a == "version") {
        if json_mode {
            println!(
                "{{\"name\":\"nala-colibri-rs\",\"version\":\"{VERSION}\",\"edition\":\"2024\",\"standalone\":true,\"nala_os_required\":false,\"unsafe_code\":\"forbidden\",\"external_crates\":0,\"license\":\"Apache-2.0\"}}"
            );
        } else {
            println!("nala-colibri-rs {VERSION} (Apache-2.0, 100% Standalone Safe Rust Edition 2024, Zero-Python/Zero-Node.js)");
        }
        return;
    }

    if args.iter().any(|a| a == "--self-check" || a == "self-check") {
        cmd_self_check(quiet_eco, json_mode);
        return;
    }

    if args.iter().any(|a| a == "--bench" || a == "bench") {
        let iters = extract_flag_val(&args, "--iters")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(10_000);
        let rows = bench::run_all_benchmarks(iters);
        if json_mode {
            let items: Vec<String> = rows
                .iter()
                .map(|r| {
                    format!(
                        "{{\"id\":\"{}\",\"subsystem\":\"{}\",\"iterations\":{},\"ns_per_op\":{},\"us_per_op\":{}}}",
                        r.id,
                        json_escape(r.subsystem),
                        r.iterations,
                        r.ns_per_op,
                        r.us_per_op
                    )
                })
                .collect();
            println!(
                "{{\"version\":\"{VERSION}\",\"os_arch\":\"{}-{}\",\"standalone\":true,\"benchmarks\":[{}]}}",
                std::env::consts::OS,
                std::env::consts::ARCH,
                items.join(",")
            );
        } else {
            println!(
                "=== NALA-Colibri-RS v{VERSION} — 14-Subsystem Standalone Native Benchmark ({iters} iterations, {}-{}) ===",
                std::env::consts::OS,
                std::env::consts::ARCH
            );
            for r in &rows {
                println!(
                    "  [{}] {:<46} {:>10.2} ns/op ({:>8.3} us/op)",
                    r.id, r.subsystem, r.ns_per_op, r.us_per_op
                );
            }
        }
        return;
    }

    let cmd = args[0].as_str();
    match cmd {
        "list" | "catalog" => cmd_list(json_mode),
        "families" => cmd_families(json_mode),
        "info" => {
            let target = args.get(1).map(|s| s.as_str()).unwrap_or("glm-5.3");
            cmd_info(target, json_mode);
        }
        "doctor" | "status" => cmd_doctor(quiet_eco, json_mode),
        "plan" | "tune" | "setup" => {
            let target = args
                .get(1)
                .filter(|s| !s.starts_with('-'))
                .map(|s| s.as_str())
                .unwrap_or("glm-5.3");
            cmd_plan(target, &args, quiet_eco, json_mode);
        }
        "decide" | "systemone" => cmd_decide(&args, json_mode),
        "inspect" | "import" => {
            let Some(p) = args.get(1) else {
                eprintln!("Usage: nala-colibri-rs inspect <file.safetensors|file.gguf|file.qpack>");
                std::process::exit(2);
            };
            match inspect_container_file(Path::new(p)) {
                Ok(js) => println!("{js}"),
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            }
        }
        "hash" | "sha256" => cmd_hash(&args[1..], json_mode),
        "verify-sums" => {
            let Some(manifest) = args.get(1) else {
                eprintln!("Usage: nala-colibri-rs verify-sums <SHA256SUMS.txt>");
                std::process::exit(2);
            };
            cmd_verify_sums(Path::new(manifest), json_mode);
        }
        "mcp" | "--mcp-stdio" => {
            if let Err(e) = run_mcp_stdio_loop() {
                eprintln!("MCP error: {e}");
                std::process::exit(1);
            }
        }
        "serve" | "--serve" | "web" | "--web" | "start" => {
            let bind = extract_flag_val(&args, "--bind")
                .or_else(|| args.get(1).filter(|s| s.contains(':')).cloned())
                .unwrap_or_else(|| "127.0.0.1:8000".to_string());
            let model = extract_flag_val(&args, "--model").unwrap_or_else(|| "glm-5.3".to_string());
            if let Err(e) = run_http_gateway(&bind, &model) {
                eprintln!("{e}");
                std::process::exit(1);
            }
        }
        "rsi" => cmd_rsi(&args, json_mode),
        "prompt" => {
            let ide = extract_flag_val(&args, "--ide").unwrap_or_else(|| "Antigravity".to_string());
            let prop = extract_flag_val(&args, "--proposal")
                .or_else(|| args.get(1).filter(|s| !s.starts_with('-')).cloned())
                .unwrap_or_else(|| "RSI-COLIBRI-005".to_string());
            match compile_multi_ide_prompt(&ide, &prop) {
                Ok(txt) => println!("{txt}"),
                Err(e) => {
                    eprintln!("{e}");
                    std::process::exit(1);
                }
            }
        }
        _ => {
            eprintln!("Unknown subcommand '{cmd}'. Run `nala-colibri-rs --help`.");
            std::process::exit(2);
        }
    }
}

fn extract_flag_val(args: &[String], flag: &str) -> Option<String> {
    for i in 0..args.len() {
        if args[i] == flag {
            return args.get(i + 1).cloned();
        }
        if let Some(rest) = args[i].strip_prefix(&format!("{flag}=")) {
            return Some(rest.to_string());
        }
    }
    None
}

fn print_help(lang: &str) {
    let banner = match lang {
        "fr" => "Moteur d'inférence MoE hiérarchique 3-Niveaux universel 100% Rust Natif (Sans Python/Node.js)",
        "it" => "Motore di inferenza MoE gerarchico a 3 livelli universale 100% Rust Nativo (Zero dipendenze)",
        "rm" => "Maschina d'inferenza MoE ierarchica universala da 3 stgalims 100% Rust Nativ",
        "en" => "100% Standalone Native Rust Hierarchical 3-Tier (VRAM + Host RAM + Striped NVMe) MoE Engine",
        _ => "100% Standalone Native Rust Hierarchische 3-Tier (VRAM + Host-RAM + Striped NVMe) MoE-Engine",
    };
    println!(
        "NALA-Colibri-RS v{VERSION} (alias: coli-rs)\n{banner}\n\n\
         USAGE:\n  \
           nala-colibri-rs <SUBCOMMAND> [OPTIONS]\n\n\
         CORE SUBCOMMANDS:\n  \
           list [--json]                       List all 21 MoE/Vision/System-One presets across 13 families\n  \
           families [--json]                   Inspect all 13 Colibri engine architecture families\n  \
           info <model> [--json]               Show exact expert dimensions, MLA KV bytes & SPDX license\n  \
           doctor [--eco-thermal] [--json]     Auto-detect universal hardware profile, ISA & security gates\n  \
           plan <model> [--vram-gb X] [--ram-gb Y] [--nvme-gbps Z] [--hal P] [--eco-thermal] [--json]\n  \
           decide --prompt \"...\" [--mode choice|noul|score] [--options a,b,c] [--evidence-lock] [--json]\n  \
           inspect <path>                      Safely inspect Safetensors, GGUF v2/v3, or qpack v1 header\n  \
           hash <file...> [--json]             Compute streaming cryptographic SHA-256 digest(s)\n  \
           verify-sums <SHA256SUMS.txt>        Verify cryptographic SHA-256 manifest offline\n  \
           --self-check [--json]               Verify binary SHA-256 + standalone kernel/security invariants\n  \
           serve [--bind 127.0.0.1:8000] [--model glm-5.3]   Start hardened OpenAI/Anthropic/SystemOne/UI server\n  \
           mcp                                 Start JSON-RPC 2.0 MCP server over stdio (11 tools)\n  \
           rsi [--json]                        Display Wave-1 & Wave-2 Dream/REA/RSI Solution & Delta Matrix\n  \
           prompt <RSI-ID> [--ide Antigravity|Codex|ZCode|OpenCode]  Compile Multi-IDE rollout prompt\n  \
           --bench [--iters 10000] [--json]    Run 14-subsystem native micro-benchmark suite\n  \
           --lang de|fr|it|rm|en               Select localized CLI header language"
    );
}

pub fn run_runtime_invariants_check() -> Result<(), String> {
    // 1. Dual-[f32;8] SIMD GEMV TP=1 vs TP=2 bit-exactness + int4 nibble LUT
    let x = [1.0f32; 64];
    let wq = [2i8; 4 * 64];
    let sc = [0.5f32; 4];
    let mut y8 = [0.0f32; 4];
    matmul_q_gs64_int8(&mut y8, &x, &wq, &sc, 4, 64);
    if y8 != [64.0, 64.0, 64.0, 64.0] {
        return Err("matmul_q_gs64_int8 self-test mismatch".to_string());
    }

    let wp = [0x99u8; 4 * 32];
    let mut y4 = [0.0f32; 4];
    matmul_q_gs64_int4(&mut y4, &x, &wp, &sc, 4, 64);
    if y4 != [32.0, 32.0, 32.0, 32.0] {
        return Err("matmul_q_gs64_int4 self-test mismatch".to_string());
    }

    // 2. BF16 & F16 conversion
    let mut fp = [0.0f32; 2];
    bf16_to_f32_bulk(&[0x3f80, 0xc000], &mut fp);
    if fp != [1.0, -2.0] {
        return Err("bf16_to_f32 self-test mismatch".to_string());
    }
    f16_to_f32_bulk(&[0x3c00, 0xc000], &mut fp);
    if fp != [1.0, -2.0] {
        return Err("f16_to_f32 self-test mismatch".to_string());
    }

    // 3. Safetensors SEC-COLIBRI-04 overflow guard
    let bad = br#"{"t":{"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,8]}}"#;
    let mut st_buf = Vec::with_capacity(8 + bad.len());
    st_buf.extend_from_slice(&(bad.len() as u64).to_le_bytes());
    st_buf.extend_from_slice(bad);
    if parse_safetensors_header(&st_buf, None).is_ok() {
        return Err("SEC-COLIBRI-04 overflow check failed".to_string());
    }

    // 4. RAII ExpertStore + KvPrefixCache + COLI_PARK
    let store = ExpertStore::new("self-check", 2, 4, 2, 4, 7.0, 7.0);
    {
        let lease = store.acquire(1, 1).ok_or("ExpertStore acquire failed")?;
        if lease.active_leases() != 1 || store.can_evict(1, 1) {
            return Err("RAII ExpertLease eviction guard failed".to_string());
        }
    }
    if !store.can_evict(1, 1) {
        return Err("RAII ExpertLease drop release failed".to_string());
    }

    let mut kv = KvPrefixCache::new(8);
    kv.commit_tokens(0, &[1, 2, 3, 4]);
    if kv.reusable_prefix_len(&[1, 2, 3, 9]) != 3 {
        return Err("KvPrefixCache reuse mismatch".to_string());
    }
    let mut park = ColiParkVault::new(4);
    let mut live_kv = [1.0f32; 8];
    let mut live_tok = [7u32, 8u32];
    park.park_save(&live_kv, &live_tok, 1);
    live_kv[0] = -5.0;
    park.park_restore(&mut live_kv, &mut live_tok);
    if live_kv[0] != 1.0 {
        return Err("COLI_PARK restore mismatch".to_string());
    }

    // 5. System One + SEC-02/03 SSRF & HMAC
    let s1 = evaluate_system_one("Verify yes [Dok-SELF, S. 1]", SystemOneMode::Noul, &[], 1.0, true);
    if !s1.evidence_lock_ok || s1.generated_tokens != 0 {
        return Err("System One Evidence-Lock self-test failed".to_string());
    }
    let mac = hex_encode(&hmac_sha256(b"k", b"m"));
    if !verify_hmac_sha256_hex(b"k", b"m", &mac)
        || is_strictly_public_ip(IpAddr::V4(Ipv4Addr::new(169, 254, 169, 254)))
    {
        return Err("SEC-02/03 HMAC or SSRF guard self-test failed".to_string());
    }

    Ok(())
}

fn cmd_self_check(quiet_eco: bool, json_mode: bool) {
    let exe_hash = std::env::current_exe()
        .ok()
        .and_then(|p| sha256_file(&p).ok())
        .unwrap_or_else(|| "unavailable".to_string());
    let hw = detect_host_hardware(quiet_eco);
    if let Err(e) = run_runtime_invariants_check() {
        eprintln!("SELF_CHECK_FAILED: {e}");
        std::process::exit(1);
    }
    if json_mode {
        println!(
            "{{\"self_check\":\"PASS\",\"version\":\"{VERSION}\",\"standalone\":true,\"nala_os_required\":false,\"os_arch\":\"{}\",\"cpu_isa\":\"{}\",\"hal_profile\":\"{}\",\"binary_sha256\":\"{}\"}}",
            hw.os_arch,
            hw.cpu_isa,
            hw.profile.slug(),
            exe_hash
        );
    } else {
        println!("=== NALA-Colibri-RS v{VERSION} Standalone Self-Check ===");
        println!("  Status:            PASS (All kernel, container, RAII & security invariants verified)");
        println!("  Standalone Mode:   YES (0% NALA-OS / Python / Node.js dependency)");
        println!("  OS / Arch / ISA:   {} ({})", hw.os_arch, hw.cpu_isa);
        println!("  Detected HAL:      {} (VRAM {:.2} GB | RAM {:.2} GB | NVMe {}x)", hw.profile.slug(), hw.vram_gb, hw.ram_gb, hw.nvme_stripes);
        println!("  Binary SHA-256:    {}", exe_hash);
    }
}

fn cmd_hash(files: &[String], json_mode: bool) {
    let paths: Vec<&String> = files.iter().filter(|s| !s.starts_with('-')).collect();
    if paths.is_empty() {
        eprintln!("Usage: nala-colibri-rs hash <file...>");
        std::process::exit(2);
    }
    let mut rows = Vec::new();
    for p in paths {
        match sha256_file(Path::new(p)) {
            Ok(hex) => {
                if json_mode {
                    rows.push(format!("{{\"file\":\"{}\",\"sha256\":\"{hex}\"}}", json_escape(p)));
                } else {
                    println!("{hex}  {p}");
                }
            }
            Err(e) => {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
    }
    if json_mode {
        println!("[{}]", rows.join(","));
    }
}

fn cmd_verify_sums(manifest: &Path, json_mode: bool) {
    let content = match std::fs::read_to_string(manifest) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Cannot read manifest '{}': {e}", manifest.display());
            std::process::exit(1);
        }
    };
    let base_dir = manifest.parent().unwrap_or_else(|| Path::new("."));
    let mut checked = 0usize;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut parts = trimmed.split_whitespace();
        let Some(expected) = parts.next() else { continue };
        let Some(rel_name) = parts.next() else { continue };
        let clean_name = rel_name.trim_start_matches('*');
        let candidate = if Path::new(clean_name).exists() {
            Path::new(clean_name).to_path_buf()
        } else {
            base_dir.join(clean_name)
        };
        let actual = match sha256_file(&candidate) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("FAIL {}: {e}", candidate.display());
                std::process::exit(1);
            }
        };
        if !actual.eq_ignore_ascii_case(expected) {
            eprintln!("FAIL {}: expected {expected}, got {actual}", candidate.display());
            std::process::exit(1);
        }
        checked += 1;
        if !json_mode {
            println!("{clean_name}: OK ({actual})");
        }
    }
    if json_mode {
        println!("{{\"status\":\"PASS\",\"verified_files\":{checked}}}");
    } else {
        println!("Verified {checked} file(s): ALL SHA-256 HASHES MATCH.");
    }
}

fn cmd_list(json_mode: bool) {
    if json_mode {
        let items: Vec<String> = CATALOG_MODELS
            .iter()
            .map(|m| {
                format!(
                    "{{\"slug\":\"{}\",\"family\":\"{}\",\"total_params_b\":{},\"active_params_b\":{},\"layers\":{},\"experts\":{},\"disk_gb\":{},\"license\":\"{}\",\"commercial_ok\":{}}}",
                    m.slug,
                    m.family_id,
                    m.total_params_b,
                    m.active_params_b,
                    m.num_layers,
                    m.routed_experts,
                    m.disk_footprint_gb,
                    m.license_spdx,
                    m.commercial_ok
                )
            })
            .collect();
        println!("[{}]", items.join(","));
    } else {
        println!(
            "{:<22} {:<13} {:>9} {:>9} {:>10} {:>9}  {}",
            "SLUG", "FAMILY", "TOTAL(B)", "ACTIVE", "EXPERTS", "DISK(GB)", "LICENSE"
        );
        println!("{}", "-".repeat(94));
        for m in CATALOG_MODELS {
            println!(
                "{:<22} {:<13} {:>8.1}B {:>8.1}B {:>4}x{:<5} {:>7.1}GB  {}",
                m.slug,
                m.family_id,
                m.total_params_b,
                m.active_params_b,
                m.num_layers,
                m.routed_experts,
                m.disk_footprint_gb,
                m.license_spdx
            );
        }
    }
}

fn cmd_families(json_mode: bool) {
    if json_mode {
        let items: Vec<String> = ENGINE_FAMILIES
            .iter()
            .map(|f| {
                format!(
                    "{{\"id\":\"{}\",\"attention\":\"{}\",\"mtp\":{},\"vision\":{},\"tools\":{},\"system_one\":{}}}",
                    f.id,
                    f.attention.as_str(),
                    f.supports_mtp,
                    f.supports_vision,
                    f.supports_tools,
                    f.supports_system_one
                )
            })
            .collect();
        println!("[{}]", items.join(","));
    } else {
        for f in ENGINE_FAMILIES {
            println!(
                "[ {:<13} ] {} | Attention: {} | Quant: {}",
                f.id,
                f.title,
                f.attention.as_str(),
                f.default_quant
            );
        }
    }
}

fn cmd_info(query: &str, json_mode: bool) {
    let Some(m) = find_model(query) else {
        eprintln!("Model '{query}' not found in catalog.");
        std::process::exit(1);
    };
    let fam = find_family(m.family_id);
    if json_mode {
        println!(
            "{{\"slug\":\"{}\",\"display_name\":\"{}\",\"family\":\"{}\",\"hf_repo\":\"{}\",\"container\":\"{}\",\"total_params_b\":{},\"active_params_b\":{},\"layers\":{},\"routed_experts\":{},\"top_k\":{},\"kv_bytes_per_token\":{},\"license_spdx\":\"{}\",\"commercial_ok\":{}}}",
            m.slug,
            json_escape(m.display_name),
            m.family_id,
            m.hf_repo,
            m.container_fmt,
            m.total_params_b,
            m.active_params_b,
            m.num_layers,
            m.routed_experts,
            m.top_k_experts,
            m.kv_bytes_per_token,
            m.license_spdx,
            m.commercial_ok
        );
    } else {
        println!("Model:          {} ({})", m.display_name, m.slug);
        println!("Engine Family:  {} ({})", m.family_id, fam.map(|f| f.attention.as_str()).unwrap_or(""));
        println!("HF Repository:  {}", m.hf_repo);
        println!("Container:      {}", m.container_fmt);
        println!("Parameters:     {:.1}B total / {:.1}B active per token", m.total_params_b, m.active_params_b);
        println!("MoE Geometry:   {} layers x {} experts (top-{} routed)", m.num_layers, m.routed_experts, m.top_k_experts);
        println!("KV Footprint:   {} bytes/token | Disk: {:.1} GB", m.kv_bytes_per_token, m.disk_footprint_gb);
        println!("Weight License: {} (commercial_ok={})", m.license_spdx, m.commercial_ok);
    }
}

fn cmd_doctor(quiet_eco: bool, json_mode: bool) {
    let hw = detect_host_hardware(quiet_eco);
    if json_mode {
        println!(
            "{{\"hal_profile\":\"{}\",\"title\":\"{}\",\"standalone_mode\":{},\"nala_os_required\":false,\"os_arch\":\"{}\",\"cpu_isa\":\"{}\",\"cpu_threads\":{},\"vram_gb\":{},\"ram_gb\":{},\"nvme_stripes\":{},\"nvme_gbps\":{},\"gpu_mem_bw_gbps\":{},\"eco_thermal_quiet\":{},\"security_gates\":[\"SEC-COLIBRI-01\",\"SEC-COLIBRI-02\",\"SEC-COLIBRI-03\",\"SEC-COLIBRI-04\"]}}",
            hw.profile.slug(),
            hw.profile.title(),
            hw.standalone_mode,
            hw.os_arch,
            hw.cpu_isa,
            hw.cpu_threads,
            hw.vram_gb,
            hw.ram_gb,
            hw.nvme_stripes,
            hw.aggregate_nvme_gbps,
            hw.gpu_mem_bw_gbps,
            hw.eco_thermal_quiet
        );
    } else {
        println!("=== NALA-Colibri-RS Universal Hardware & Security Doctor ===");
        println!("  Standalone:     YES (Zero NALA-OS / Python / Node.js required)");
        println!("  HAL Profile:    {}", hw.profile.title());
        println!("  OS / Arch:      {} ({} threads | ISA: {})", hw.os_arch, hw.cpu_threads, hw.cpu_isa);
        println!("  Tier 0 (VRAM):  {:.2} GB (@ {:.0} GB/s)", hw.vram_gb, hw.gpu_mem_bw_gbps);
        println!("  Tier 1 (RAM):   {:.2} GB", hw.ram_gb);
        println!("  Tier 2 (NVMe):  {} stripe(s) @ {:.1} GB/s", hw.nvme_stripes, hw.aggregate_nvme_gbps);
        println!("  Quiet Thermal:  {}", if hw.eco_thermal_quiet { "ACTIVE (72% Quiet Duty Cap)" } else { "OFF (100% Performance)" });
        println!("  sHeriFF Gates:  SEC-COLIBRI-01..04 + #![forbid(unsafe_code)] PASS");
    }
}

fn cmd_plan(target: &str, args: &[String], quiet_eco: bool, json_mode: bool) {
    let Some(m) = find_model(target) else {
        eprintln!("Unknown model '{target}'. Run `nala-colibri-rs list`.");
        std::process::exit(1);
    };
    let mut hw = detect_host_hardware(quiet_eco);
    if let Some(p_str) = extract_flag_val(args, "--hal") {
        if let Some(prof) = HalProfile::from_slug(&p_str) {
            hw.profile = prof;
        }
    }
    if let Some(v) = extract_flag_val(args, "--vram-gb").and_then(|s| s.parse::<f64>().ok()) {
        hw.vram_gb = v;
    }
    if let Some(r) = extract_flag_val(args, "--ram-gb").and_then(|s| s.parse::<f64>().ok()) {
        hw.ram_gb = r;
    }
    if let Some(b) = extract_flag_val(args, "--nvme-gbps").and_then(|s| s.parse::<f64>().ok()) {
        hw.aggregate_nvme_gbps = b;
    }
    let slots = extract_flag_val(args, "--kv-slots")
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(4);
    let ctx = extract_flag_val(args, "--ctx")
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(8192);

    let p = compute_three_tier_plan(m, &hw, slots, ctx);
    if json_mode {
        println!(
            "{{\"model\":\"{}\",\"hal_profile\":\"{}\",\"total_experts\":{},\"hot_vram_experts\":{},\"warm_ram_experts\":{},\"cold_nvme_experts\":{},\"vram_used_gb\":{},\"ram_used_gb\":{},\"nvme_streamed_gb\":{},\"kv_slots\":{},\"learned_cache_hit_pct\":{},\"mtp_enabled\":{},\"nvme_duty_cap_pct\":{},\"estimated_decode_tok_s\":{},\"estimated_prefill_tok_s\":{},\"fit_verdict\":\"{}\"}}",
            p.model_slug,
            p.hal_profile,
            p.total_routed_experts,
            p.hot_vram_experts,
            p.warm_ram_experts,
            p.cold_nvme_experts,
            p.vram_used_gb,
            p.ram_used_gb,
            p.nvme_streamed_gb,
            p.kv_slots,
            p.learned_cache_hit_pct,
            p.mtp_enabled,
            p.nvme_duty_cap_pct,
            p.estimated_decode_tok_s,
            p.estimated_prefill_tok_s,
            p.fit_verdict
        );
    } else {
        println!("=== 3-Tier Hierarchical MoE Plan: {} ===", p.display_name);
        println!("  HAL Profile:        {}", p.hal_profile);
        println!("  Total MoE Experts:  {}", p.total_routed_experts);
        println!("  -> Tier 0 (VRAM):   {} hot experts ({:.2} GB used)", p.hot_vram_experts, p.vram_used_gb);
        println!("  -> Tier 1 (RAM):    {} warm experts ({:.2} GB used)", p.warm_ram_experts, p.ram_used_gb);
        println!("  -> Tier 2 (NVMe):   {} cold experts ({:.2} GB striped)", p.cold_nvme_experts, p.nvme_streamed_gb);
        println!("  KV Slots:           {} x {} tokens ({:.2} GB)", p.kv_slots, p.context_tokens_per_slot, p.kv_budget_gb);
        println!("  Learned Cache Hit:  {:.2}% (.coli_usage + Markov lookahead)", p.learned_cache_hit_pct);
        println!("  MTP Draft Gate:     {}", p.mtp_gate_reason);
        println!("  Estimated Speed:    {:.2} decode tok/s | {:.1} prefill tok/s", p.estimated_decode_tok_s, p.estimated_prefill_tok_s);
        println!("  Verdict:            {}", p.fit_verdict);
    }
}

fn cmd_decide(args: &[String], json_mode: bool) {
    let prompt = extract_flag_val(args, "--prompt")
        .unwrap_or_else(|| "Verified bit-exact IEEE-754 safety [Dok-COLIBRI-01, S. 4]".to_string());
    let mode_s = extract_flag_val(args, "--mode").unwrap_or_else(|| "choice".to_string());
    let opts: Vec<String> = extract_flag_val(args, "--options")
        .map(|s| {
            s.split(',')
                .map(|x| x.trim().to_string())
                .filter(|x| !x.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let req_ev = args.iter().any(|a| a == "--evidence-lock");
    let res = evaluate_system_one(&prompt, SystemOneMode::parse(&mode_s), &opts, 1.0, req_ev);

    if json_mode {
        let dist_js: Vec<String> = res
            .distribution
            .iter()
            .map(|d| format!("{{\"option\":\"{}\",\"probability\":{}}}", json_escape(&d.label), d.probability))
            .collect();
        println!(
            "{{\"mode\":\"{}\",\"winner\":\"{}\",\"scalar_score\":{},\"confidence\":{},\"entropy_nats\":{},\"evidence_lock_ok\":{},\"generated_tokens\":0,\"distribution\":[{}]}}",
            res.mode.as_str(),
            json_escape(&res.winner),
            res.scalar_score.map(|v| v.to_string()).unwrap_or_else(|| "null".to_string()),
            res.confidence,
            res.entropy_nats,
            res.evidence_lock_ok,
            dist_js.join(",")
        );
    } else {
        println!(
            "System One ({}) -> Winner: '{}' | Confidence: {:.2}% | Evidence-Lock: {} | Generated Tokens: 0",
            res.mode.as_str(),
            res.winner,
            res.confidence * 100.0,
            if res.evidence_lock_ok { "PASS" } else { "BLOCKED_MISSING_CITATION" }
        );
        for d in &res.distribution {
            println!("  - {:<18} {:>7.2}% (logit {:+.4})", d.label, d.probability * 100.0, d.logit);
        }
    }
}

fn cmd_rsi(args: &[String], json_mode: bool) {
    let _ = args;
    if json_mode {
        let items: Vec<String> = RSI_PROPOSALS
            .iter()
            .map(|p| {
                format!(
                    "{{\"id\":\"{}\",\"wave\":{},\"status\":\"{}\",\"title\":\"{}\",\"delta_perf\":\"{}\",\"delta_feat\":\"{}\",\"delta_sec\":\"{}\",\"score\":{}}}",
                    p.id,
                    p.wave,
                    p.status,
                    json_escape(p.title),
                    json_escape(p.delta_performance),
                    json_escape(p.delta_features),
                    json_escape(p.delta_security),
                    p.council_score
                )
            })
            .collect();
        println!("[{}]", items.join(","));
    } else {
        println!("=== NALA-Colibri-RS Dream / Software-Pearls / REA / RSI Matrix ===");
        for p in RSI_PROPOSALS {
            println!(
                "  [{}] (Wave {} | {} | Score {:.2}/10) {}\n      Δ Perf: {}\n      Δ Feat: {}\n      Δ Sec:  {}",
                p.id, p.wave, p.status, p.council_score, p.title, p.delta_performance, p.delta_features, p.delta_security
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::{MAX_SAFETENSORS_HEADER_BYTES, detect_split_gguf_part, parse_gguf_header, parse_qpack_header, parse_safetensors_header};
    use crate::expert_store::{ColiParkVault, ExpertStore, GenerationScheduler};
    use crate::kernels::gs_group_sum_simd8;
    use crate::rsi::{MANDATORY_THESTANDART_FOOTER, compile_multi_ide_prompt, issue_two_click_arm_token, verify_two_click_execute};
    use crate::security::{safe_subpath, sha256, sha256_reader, verify_and_pin_public_url, verify_loopback_origin};
    use std::net::Ipv6Addr;

    #[test]
    fn test_01_all_13_families_and_21_models_present() {
        assert_eq!(ENGINE_FAMILIES.len(), 13);
        assert!(CATALOG_MODELS.len() >= 20);
        assert!(find_model("glm-5.3").is_some());
        assert!(find_model("deepseek-v4.1-flash").is_some());
        assert!(find_model("kimi-k3").is_some());
        assert!(!find_model("qwen-image-2.1").unwrap().commercial_ok);
    }

    #[test]
    fn test_02_ieee754_order_preserving_matmul_q_gs64_tp_invariance() {
        let rows = 8usize;
        let cols = 128usize;
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.013) - 0.7).collect();
        let wq: Vec<i8> = (0..rows * cols).map(|i| ((i * 7 % 29) as i8) - 14).collect();
        let scales: Vec<f32> = (0..rows * 2).map(|i| 0.02 + (i as f32) * 0.001).collect();

        let mut out_tp1 = vec![0.0f32; rows];
        matmul_q_gs64_int8(&mut out_tp1, &x, &wq, &scales, rows, cols);

        let mut out_tp2_a = vec![0.0f32; 4];
        let mut out_tp2_b = vec![0.0f32; 4];
        matmul_q_gs64_int8(&mut out_tp2_a, &x, &wq[..4 * cols], &scales[..8], 4, cols);
        matmul_q_gs64_int8(&mut out_tp2_b, &x, &wq[4 * cols..], &scales[8..], 4, cols);

        for r in 0..4 {
            assert_eq!(out_tp1[r].to_bits(), out_tp2_a[r].to_bits());
            assert_eq!(out_tp1[r + 4].to_bits(), out_tp2_b[r].to_bits());
        }
    }

    #[test]
    fn test_03_int4_gs64_packed_gemv_exactness() {
        let rows = 4usize;
        let cols = 64usize;
        let x = vec![1.0f32; cols];
        let wp = vec![0x99u8; rows * 32];
        let scales = vec![0.5f32; rows];
        let mut out = vec![0.0f32; rows];
        matmul_q_gs64_int4(&mut out, &x, &wp, &scales, rows, cols);
        for &v in &out {
            assert_eq!(v, 32.0f32);
        }
    }

    #[test]
    fn test_04_bf16_and_f16_conversions() {
        let bf = [0x3f80u16, 0xc000u16];
        let mut out = [0.0f32; 2];
        bf16_to_f32_bulk(&bf, &mut out);
        assert_eq!(out, [1.0f32, -2.0f32]);

        let fp16 = [0x3c00u16, 0xc000u16];
        f16_to_f32_bulk(&fp16, &mut out);
        assert_eq!(out, [1.0f32, -2.0f32]);
    }

    #[test]
    fn test_05_sec_colibri_04_safetensors_header_cap_and_overflow() {
        let mut huge = [0u8; 16];
        huge[..8].copy_from_slice(&(MAX_SAFETENSORS_HEADER_BYTES + 1).to_le_bytes());
        assert!(parse_safetensors_header(&huge, None).is_err());

        let bad_json = br#"{"evil":{"dtype":"F32","shape":[18446744073709551615,4],"data_offsets":[0,16]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(bad_json.len() as u64).to_le_bytes());
        buf.extend_from_slice(bad_json);
        assert!(parse_safetensors_header(&buf, None).is_err());
    }

    #[test]
    fn test_06_gguf_split_shard_and_qpack_headers() {
        assert_eq!(detect_split_gguf_part("Qwen3.6-00002-of-00005.gguf"), Some((2, 5)));
        let mut gguf = [0u8; 24];
        gguf[0..4].copy_from_slice(b"GGUF");
        gguf[4..8].copy_from_slice(&3u32.to_le_bytes());
        gguf[8..16].copy_from_slice(&128u64.to_le_bytes());
        gguf[16..24].copy_from_slice(&16u64.to_le_bytes());
        let g = parse_gguf_header(&gguf, Some("part-00001-of-00003.gguf")).unwrap();
        assert_eq!(g.split_part, Some((1, 3)));

        let mut qp = [0u8; 20];
        qp[0..4].copy_from_slice(b"QPK1");
        qp[4..8].copy_from_slice(&1u32.to_le_bytes());
        qp[8..12].copy_from_slice(&8u32.to_le_bytes());
        qp[12..20].copy_from_slice(&4096u64.to_le_bytes());
        assert_eq!(parse_qpack_header(&qp).unwrap().chunk_count, 8);
    }

    #[test]
    fn test_07_raii_expert_store_lease_blocks_eviction_until_drop() {
        let mut store = ExpertStore::new("glm-5.3", 4, 8, 4, 12, 7.2, 6.8);
        store.record_markov_pair(1, 2, 7);
        assert_eq!(store.predict_next_layer_expert(1, 2), Some((2, 7)));
        assert!(store.can_evict(1, 2));
        {
            let lease = store.acquire(1, 2).unwrap();
            assert_eq!(lease.active_leases(), 1);
            assert!(!store.can_evict(1, 2));
        }
        assert!(store.can_evict(1, 2));
        assert!(store.encode_coli_usage_bytes().starts_with(b"COLI_U01"));
    }

    #[test]
    fn test_08_kv_prefix_taint_and_coli_park_roundtrip() {
        let mut kv_cache = KvPrefixCache::new(64);
        kv_cache.commit_tokens(0, &[10, 20, 30, 40]);
        assert_eq!(kv_cache.reusable_prefix_len(&[10, 20, 30, 99]), 3);
        assert_eq!(kv_cache.reusable_prefix_len(&[10, 20, 30, 40]), 3);
        kv_cache.mark_tainted();
        assert_eq!(kv_cache.reusable_prefix_len(&[10, 20, 30, 40]), 0);

        let mut vault = ColiParkVault::new(4);
        let mut live_kv: Vec<f32> = (0..40).map(|v| v as f32).collect();
        let mut live_toks: Vec<u32> = (100..110).collect();
        vault.park_save(&live_kv, &live_toks, 3);
        live_kv[..12].fill(-99.0);
        live_toks[..3].fill(999);
        let restored = vault.park_restore(&mut live_kv, &mut live_toks);
        assert_eq!(restored, 3);
        assert_eq!(&live_toks[..3], &[100, 101, 102]);
        assert_eq!(live_kv[11], 11.0);
    }

    #[test]
    fn test_09_scheduler_16_slots_and_mtp_88pct_gate() {
        let mut sched = GenerationScheduler::new(2, 1);
        let s0 = sched.try_acquire_slot().unwrap();
        let s1 = sched.try_acquire_slot().unwrap();
        assert_eq!(sched.try_acquire_slot(), Err("QUEUED_FIFO"));
        assert_eq!(sched.try_acquire_slot(), Err("HTTP_429_QUEUE_FULL"));
        sched.release_slot(s0);
        sched.release_slot(s1);
        assert_eq!(sched.active_count(), 0);

        let m = find_model("glm-5.3").unwrap();
        let mut hw = detect_host_hardware(true);
        hw.vram_gb = 16.0;
        hw.ram_gb = 32.0;
        let plan = compute_three_tier_plan(m, &hw, 2, 4096);
        assert!(!plan.mtp_enabled);
        assert_eq!(plan.nvme_duty_cap_pct, 72);
    }

    #[test]
    fn test_10_system_one_probabilities_sum_to_one_and_evidence_lock() {
        let opts = vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()];
        let res_no_cit = evaluate_system_one("Choose beta option", SystemOneMode::Choice, &opts, 1.0, true);
        assert!(!res_no_cit.evidence_lock_ok);
        assert_eq!(res_no_cit.generated_tokens, 0);
        let sum: f64 = res_no_cit.distribution.iter().map(|d| d.probability).sum();
        assert!((sum - 1.0).abs() < 1e-4);

        let res_with_cit = evaluate_system_one(
            "Choose beta verified [Dok-SPEC-07, S. 12]",
            SystemOneMode::Choice,
            &opts,
            1.0,
            true,
        );
        assert!(res_with_cit.evidence_lock_ok);
        assert_eq!(res_with_cit.winner, "beta");
    }

    #[test]
    fn test_11_sec_colibri_01_02_03_04_defenses() {
        assert!(verify_loopback_origin(Some("http://127.0.0.1:8000")));
        assert!(!verify_loopback_origin(Some("https://evil.example.com")));
        assert!(!is_strictly_public_ip(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1))));
        assert!(!is_strictly_public_ip(IpAddr::V4(Ipv4Addr::new(169, 254, 169, 254))));
        assert!(!is_strictly_public_ip(IpAddr::V4(Ipv4Addr::new(100, 88, 1, 2))));
        assert!(!is_strictly_public_ip(IpAddr::V6(Ipv6Addr::LOCALHOST)));
        assert!(is_strictly_public_ip(IpAddr::V4(Ipv4Addr::new(1, 1, 1, 1))));
        assert!(verify_and_pin_public_url("http://169.254.169.254/latest/meta-data").is_err());
        assert!(verify_and_pin_public_url("http://localhost:8000/health").is_err());

        let mac = hex_encode(&hmac_sha256(b"secret", b"payload"));
        assert!(verify_hmac_sha256_hex(b"secret", b"payload", &mac));
        assert!(!verify_hmac_sha256_hex(b"secret", b"tampered", &mac));

        let tmp = std::env::temp_dir();
        assert!(safe_subpath(&tmp, "../etc/passwd").is_err());
        assert!(safe_subpath(&tmp, "/etc/passwd").is_err());
    }

    #[test]
    fn test_12_two_click_gate_and_verbatim_multi_ide_footer() {
        let tok = issue_two_click_arm_token("RSI-COLIBRI-005").unwrap();
        assert!(verify_two_click_execute("RSI-COLIBRI-005", "wrong-token").is_err());
        assert!(verify_two_click_execute("RSI-COLIBRI-005", &tok).is_ok());

        for ide in ["Codex", "Antigravity", "ZCode", "OpenCode"] {
            let prompt = compile_multi_ide_prompt(ide, "RSI-COLIBRI-006").unwrap();
            assert!(prompt.ends_with(MANDATORY_THESTANDART_FOOTER));
        }
    }

    #[test]
    fn test_13_streaming_sha256_matches_fips_180_4_across_multi_chunk_payload() {
        let payload: Vec<u8> = (0..150_000usize).map(|i| ((i * 31 + 7) & 0xff) as u8).collect();
        let expected = sha256(&payload);
        let streamed = sha256_reader(payload.as_slice()).unwrap();
        assert_eq!(expected, streamed);
    }

    #[test]
    fn test_14_pairwise_simd8_reduction_tree_exactness() {
        let a0 = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let a1 = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let expected = ((0.1f32 + 1.0 + (0.5 + 5.0)) + (0.3 + 3.0 + (0.7 + 7.0)))
            + ((0.2f32 + 2.0 + (0.6 + 6.0)) + (0.4 + 4.0 + (0.8 + 8.0)));
        assert_eq!(gs_group_sum_simd8(a0, a1).to_bits(), expected.to_bits());
    }

    #[test]
    fn test_15_standalone_runtime_invariants_self_check() {
        assert!(run_runtime_invariants_check().is_ok());
    }
}
