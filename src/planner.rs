//! `RSI-COLIBRI-001` & `RSI-COLIBRI-004`: Universal 100% Standalone 7-Profile Hardware Detector
//! & 3-Tier (VRAM + Host RAM + Striped NVMe) Resource Planner.
//!
//! Zero NALA-OS dependency: Uses standard OS kernel sysfs/sysctl/CIM/SMI interfaces across
//! Linux (`x86_64` SSE4.2/AVX/AVX2/AVX-512 & `aarch64`), macOS (`Apple Silicon M1-M5` & `Intel Metal`),
//! Windows 11 (`x86_64`), Docker containers, and KVM/QEMU hypervisor VMs/LXCs.

use crate::catalog::{CatalogModel, find_family};
use std::process::Command;
use std::sync::OnceLock;

static CACHED_BASE_HARDWARE: OnceLock<HardwareEnvelope> = OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalProfile {
    AppleSiliconUma,
    MacOsIntelMetal,
    NvidiaDgxSparkUma,
    NvidiaCudaDiscrete,
    AmdRocmVulkan,
    IntelArcSyclNpu,
    UniversalCpuHybrid,
}

impl HalProfile {
    pub const fn slug(self) -> &'static str {
        match self {
            Self::AppleSiliconUma => "apple-silicon-uma-metal",
            Self::MacOsIntelMetal => "macos-intel-metal-avx2",
            Self::NvidiaDgxSparkUma => "nvidia-dgx-spark-gb10-uma",
            Self::NvidiaCudaDiscrete => "nvidia-multi-single-cuda",
            Self::AmdRocmVulkan => "amd-rocm-vulkan-uma",
            Self::IntelArcSyclNpu => "intel-arc-sycl-npu",
            Self::UniversalCpuHybrid => "universal-cpu-hybrid",
        }
    }

    pub const fn title(self) -> &'static str {
        match self {
            Self::AppleSiliconUma => "Profile 1: Apple Silicon M1-M5 UMA (Metal + F_NOCACHE)",
            Self::MacOsIntelMetal => "Profile 2: macOS Intel x86_64 (Metal dGPU + SSE4.2/AVX2)",
            Self::NvidiaDgxSparkUma => "Profile 3: NVIDIA DGX/RTX Spark GB10 128GB UMA",
            Self::NvidiaCudaDiscrete => "Profile 4: NVIDIA Discrete Multi/Single CUDA",
            Self::AmdRocmVulkan => "Profile 5: AMD ROCm/Vulkan dGPU & Ryzen AI Max+ UMA",
            Self::IntelArcSyclNpu => "Profile 6: Intel Arc / Core Ultra SYCL + NPU",
            Self::UniversalCpuHybrid => "Profile 7: Universal ARM64/x86_64 CPU + NVMe Streamer",
        }
    }

    pub fn from_slug(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "1" | "apple" | "metal" | "uma" | "apple-silicon-uma-metal" => Some(Self::AppleSiliconUma),
            "2" | "macos-intel" | "macos-intel-metal-avx2" => Some(Self::MacOsIntelMetal),
            "3" | "spark" | "dgx-spark" | "nvidia-dgx-spark-gb10-uma" => Some(Self::NvidiaDgxSparkUma),
            "4" | "cuda" | "nvidia" | "nvidia-multi-single-cuda" => Some(Self::NvidiaCudaDiscrete),
            "5" | "amd" | "rocm" | "vulkan" | "amd-rocm-vulkan-uma" => Some(Self::AmdRocmVulkan),
            "6" | "intel-arc" | "sycl" | "intel-arc-sycl-npu" => Some(Self::IntelArcSyclNpu),
            "7" | "cpu" | "universal" | "universal-cpu-hybrid" => Some(Self::UniversalCpuHybrid),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HardwareEnvelope {
    pub profile: HalProfile,
    pub os_arch: String,
    pub cpu_isa: String,
    pub cpu_threads: usize,
    pub vram_gb: f64,
    pub ram_gb: f64,
    pub nvme_stripes: u32,
    pub aggregate_nvme_gbps: f64,
    pub gpu_mem_bw_gbps: f64,
    pub resizable_bar: bool,
    pub eco_thermal_quiet: bool,
    pub standalone_mode: bool,
}

#[derive(Debug, Clone)]
pub struct ThreeTierPlan {
    pub model_slug: String,
    pub display_name: String,
    pub family_id: String,
    pub hal_profile: String,
    pub total_routed_experts: u64,
    pub hot_vram_experts: u64,
    pub warm_ram_experts: u64,
    pub cold_nvme_experts: u64,
    pub vram_used_gb: f64,
    pub ram_used_gb: f64,
    pub nvme_streamed_gb: f64,
    pub kv_slots: u32,
    pub context_tokens_per_slot: u32,
    pub kv_budget_gb: f64,
    pub learned_cache_hit_pct: f64,
    pub mtp_enabled: bool,
    pub mtp_gate_reason: String,
    pub nvme_duty_cap_pct: u8,
    pub estimated_decode_tok_s: f64,
    pub estimated_prefill_tok_s: f64,
    pub fit_verdict: &'static str,
}

pub fn detect_host_hardware(eco_thermal_quiet: bool) -> HardwareEnvelope {
    let mut env = CACHED_BASE_HARDWARE
        .get_or_init(probe_host_hardware_once)
        .clone();
    env.eco_thermal_quiet = eco_thermal_quiet;
    env
}

fn probe_host_hardware_once() -> HardwareEnvelope {
    let os_arch = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);
    let cpu_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);

    if std::env::consts::OS == "macos" {
        let mem_bytes = sysctl_u64("hw.memsize").unwrap_or(32 * 1024 * 1024 * 1024);
        let total_gb = (mem_bytes as f64) / 1_073_741_824.0;
        let brand = sysctl_str("machdep.cpu.brand_string").unwrap_or_default().to_ascii_lowercase();
        if std::env::consts::ARCH == "aarch64" {
            let gpu_wired = (total_gb * 0.72).max(4.0);
            let host_warm = (total_gb * 0.20).max(2.0);
            let bw = if brand.contains("ultra") {
                819.0
            } else if brand.contains("max") {
                546.0
            } else if brand.contains("pro") {
                273.0
            } else {
                120.0
            };
            return HardwareEnvelope {
                profile: HalProfile::AppleSiliconUma,
                os_arch,
                cpu_isa: "arm64-neon-fp16-dotprod".to_string(),
                cpu_threads,
                vram_gb: round2(gpu_wired),
                ram_gb: round2(host_warm),
                nvme_stripes: 1,
                aggregate_nvme_gbps: 7.2,
                gpu_mem_bw_gbps: bw,
                resizable_bar: true,
                eco_thermal_quiet: false,
                standalone_mode: true,
            };
        }
        let dgpu_vram = detect_macos_metal_vram_gb().unwrap_or(8.0);
        let isa = detect_macos_x86_isa();
        return HardwareEnvelope {
            profile: HalProfile::MacOsIntelMetal,
            os_arch,
            cpu_isa: isa,
            cpu_threads,
            vram_gb: round2(dgpu_vram),
            ram_gb: round2((total_gb * 0.80).max(8.0)),
            nvme_stripes: 1,
            aggregate_nvme_gbps: 3.5,
            gpu_mem_bw_gbps: if dgpu_vram >= 15.0 { 1024.0 } else { 256.0 },
            resizable_bar: dgpu_vram >= 15.0,
            eco_thermal_quiet: false,
            standalone_mode: true,
        };
    }

    if std::env::consts::OS == "linux" {
        let ram_gb = linux_memtotal_gb().unwrap_or(64.0);
        let cpu_isa = detect_linux_cpu_isa();
        let nvme_count = detect_linux_nvme_count().max(1);
        let nvme_bw = if nvme_count >= 2 { 13.6 } else { 7.0 };

        // 1) NVIDIA CUDA / DGX Spark via nvidia-smi
        if let Some((gpu_count, total_vram_gb, is_gb10_spark)) = detect_nvidia_smi() {
            let profile = if is_gb10_spark {
                HalProfile::NvidiaDgxSparkUma
            } else {
                HalProfile::NvidiaCudaDiscrete
            };
            return HardwareEnvelope {
                profile,
                os_arch,
                cpu_isa,
                cpu_threads,
                vram_gb: round2(total_vram_gb),
                ram_gb: round2((ram_gb * 0.82).max(4.0)),
                nvme_stripes: nvme_count.max(gpu_count.min(2)),
                aggregate_nvme_gbps: nvme_bw,
                gpu_mem_bw_gbps: if is_gb10_spark { 273.0 } else { 896.0 },
                resizable_bar: true,
                eco_thermal_quiet: false,
                standalone_mode: true,
            };
        }

        // 2) Linux DRM sysfs (/sys/class/drm/card*/device/) for AMD ROCm/Vulkan (0x1002) & Intel Arc (0x8086)
        if let Some((profile, vram_gb, bw_gbps)) = detect_linux_drm_gpu(ram_gb) {
            let host_ram_rem = if vram_gb > ram_gb * 0.5 {
                // Unified memory APU (e.g., AMD Ryzen AI Max+ 395 w/ Radeon 8060S)
                (ram_gb - vram_gb - 4.0).max(8.0)
            } else {
                (ram_gb * 0.82).max(4.0)
            };
            return HardwareEnvelope {
                profile,
                os_arch,
                cpu_isa,
                cpu_threads,
                vram_gb: round2(vram_gb),
                ram_gb: round2(host_ram_rem),
                nvme_stripes: nvme_count,
                aggregate_nvme_gbps: nvme_bw,
                gpu_mem_bw_gbps: bw_gbps,
                resizable_bar: true,
                eco_thermal_quiet: false,
                standalone_mode: true,
            };
        }

        return HardwareEnvelope {
            profile: HalProfile::UniversalCpuHybrid,
            os_arch,
            cpu_isa,
            cpu_threads,
            vram_gb: 0.0,
            ram_gb: round2((ram_gb * 0.82).max(4.0)),
            nvme_stripes: nvme_count,
            aggregate_nvme_gbps: nvme_bw,
            gpu_mem_bw_gbps: 96.0,
            resizable_bar: false,
            eco_thermal_quiet: false,
            standalone_mode: true,
        };
    }

    // Windows / Other POSIX Universal Detection
    let win_ram = detect_windows_ram_gb().unwrap_or(32.0);
    if let Some((gpu_count, vram_gb, _)) = detect_nvidia_smi() {
        return HardwareEnvelope {
            profile: HalProfile::NvidiaCudaDiscrete,
            os_arch,
            cpu_isa: "x86_64-sse4.2-avx2".to_string(),
            cpu_threads,
            vram_gb: round2(vram_gb),
            ram_gb: round2((win_ram * 0.80).max(4.0)),
            nvme_stripes: gpu_count.clamp(1, 2),
            aggregate_nvme_gbps: 6.8,
            gpu_mem_bw_gbps: 448.0,
            resizable_bar: true,
            eco_thermal_quiet: false,
            standalone_mode: true,
        };
    }

    HardwareEnvelope {
        profile: HalProfile::UniversalCpuHybrid,
        os_arch,
        cpu_isa: format!("{}-baseline-simd", std::env::consts::ARCH),
        cpu_threads,
        vram_gb: 0.0,
        ram_gb: round2((win_ram * 0.80).max(8.0)),
        nvme_stripes: 1,
        aggregate_nvme_gbps: 6.0,
        gpu_mem_bw_gbps: 96.0,
        resizable_bar: false,
        eco_thermal_quiet: false,
        standalone_mode: true,
    }
}

fn sysctl_u64(key: &str) -> Option<u64> {
    let out = Command::new("/usr/sbin/sysctl")
        .arg("-n")
        .arg(key)
        .output()
        .or_else(|_| Command::new("sysctl").arg("-n").arg(key).output())
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout).trim().parse::<u64>().ok()
}

fn sysctl_str(key: &str) -> Option<String> {
    let out = Command::new("/usr/sbin/sysctl")
        .arg("-n")
        .arg(key)
        .output()
        .or_else(|_| Command::new("sysctl").arg("-n").arg(key).output())
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn detect_macos_x86_isa() -> String {
    let feats = sysctl_str("machdep.cpu.features")
        .unwrap_or_default()
        .to_ascii_lowercase();
    let leaf7 = sysctl_str("machdep.cpu.leaf7_features")
        .unwrap_or_default()
        .to_ascii_lowercase();
    if leaf7.contains("avx2") {
        "x86_64-sse4.2-avx-avx2".to_string()
    } else if feats.contains("avx1.0") || feats.contains("avx") {
        "x86_64-sse4.2-avx".to_string()
    } else {
        "x86_64-sse4.2".to_string()
    }
}

fn detect_macos_metal_vram_gb() -> Option<f64> {
    let out = Command::new("/usr/sbin/system_profiler")
        .arg("SPDisplaysDataType")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let txt = String::from_utf8_lossy(&out.stdout);
    let mut best_gb = 0.0f64;
    for line in txt.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("VRAM") {
            if let Some((_, rhs)) = trimmed.split_once(':') {
                let parts: Vec<&str> = rhs.split_whitespace().collect();
                if let Some(num) = parts.first().and_then(|s| s.parse::<f64>().ok()) {
                    let unit = parts.get(1).copied().unwrap_or("GB").to_ascii_uppercase();
                    let gb = if unit.contains("MB") { num / 1024.0 } else { num };
                    if gb > best_gb {
                        best_gb = gb;
                    }
                }
            }
        }
    }
    if best_gb > 0.0 { Some(best_gb) } else { None }
}

fn linux_memtotal_gb() -> Option<f64> {
    let txt = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in txt.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb_str = rest.split_whitespace().next()?;
            let kb = kb_str.parse::<f64>().ok()?;
            return Some(kb / (1024.0 * 1024.0));
        }
    }
    None
}

fn detect_linux_cpu_isa() -> String {
    if std::env::consts::ARCH == "aarch64" {
        return "arm64-neon-fp16".to_string();
    }
    let Ok(info) = std::fs::read_to_string("/proc/cpuinfo") else {
        return format!("{}-simd", std::env::consts::ARCH);
    };
    let lower = info.to_ascii_lowercase();
    if lower.contains("avx512f") {
        "x86_64-avx2-avx512".to_string()
    } else if lower.contains("avx2") {
        "x86_64-sse4.2-avx2-fma".to_string()
    } else if lower.contains(" avx ") {
        "x86_64-sse4.2-avx".to_string()
    } else if lower.contains("sse4_2") {
        "x86_64-sse4.2".to_string()
    } else {
        "x86_64-sse2".to_string()
    }
}

fn detect_linux_nvme_count() -> u32 {
    let Ok(rd) = std::fs::read_dir("/sys/block") else {
        return 1;
    };
    let mut cnt = 0u32;
    for entry in rd.flatten() {
        if let Some(name) = entry.file_name().to_str() {
            if name.starts_with("nvme") {
                cnt += 1;
            }
        }
    }
    cnt.max(1)
}

fn detect_linux_drm_gpu(host_ram_gb: f64) -> Option<(HalProfile, f64, f64)> {
    let rd = std::fs::read_dir("/sys/class/drm").ok()?;
    let mut amd_vram_bytes = 0u64;
    let mut amd_gtt_bytes = 0u64;
    let mut found_amd = false;
    let mut found_intel_gpu = false;

    for entry in rd.flatten() {
        let name = entry.file_name();
        let Some(ns) = name.to_str() else { continue };
        if !ns.starts_with("card") || ns.contains('-') {
            continue;
        }
        let dev_dir = entry.path().join("device");
        let vendor = std::fs::read_to_string(dev_dir.join("vendor"))
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        if vendor == "0x1002" {
            found_amd = true;
            let vram = std::fs::read_to_string(dev_dir.join("mem_info_vram_total"))
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(0);
            let gtt = std::fs::read_to_string(dev_dir.join("mem_info_gtt_total"))
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(0);
            amd_vram_bytes = amd_vram_bytes.max(vram);
            amd_gtt_bytes = amd_gtt_bytes.max(gtt);
        } else if vendor == "0x8086" {
            found_intel_gpu = true;
        }
    }

    if found_amd {
        let vram_gb = (amd_vram_bytes as f64) / 1_073_741_824.0;
        let gtt_gb = (amd_gtt_bytes as f64) / 1_073_741_824.0;
        // APU UMA (e.g., Ryzen AI Max+ 395 Radeon 8060S where GTT/UMA pool >= 32 GB)
        if vram_gb < 8.0 && (gtt_gb >= 24.0 || host_ram_gb >= 60.0) {
            let uma_pool = if gtt_gb >= 24.0 { gtt_gb } else { host_ram_gb * 0.72 };
            return Some((HalProfile::AmdRocmVulkan, uma_pool, 256.0));
        }
        return Some((HalProfile::AmdRocmVulkan, vram_gb.max(8.0), 512.0));
    }
    if found_intel_gpu && host_ram_gb >= 16.0 {
        return Some((HalProfile::IntelArcSyclNpu, (host_ram_gb * 0.45).min(16.0), 256.0));
    }
    None
}

fn detect_nvidia_smi() -> Option<(u32, f64, bool)> {
    let candidates = ["nvidia-smi", "/usr/bin/nvidia-smi", "C:\\Windows\\System32\\nvidia-smi.exe"];
    for bin in candidates {
        let Ok(out) = Command::new(bin)
            .args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
            .output()
        else {
            continue;
        };
        if !out.status.success() {
            continue;
        }
        let stdout = String::from_utf8_lossy(&out.stdout);
        let mut count = 0u32;
        let mut sum_mib = 0.0f64;
        let mut is_spark = false;
        for line in stdout.lines() {
            let mut parts = line.split(',');
            let name = parts.next().unwrap_or("").trim().to_ascii_lowercase();
            if name.is_empty() {
                continue;
            }
            if name.contains("gb10") || name.contains("spark") {
                is_spark = true;
            }
            let mib = parts
                .next()
                .and_then(|s| s.trim().parse::<f64>().ok())
                .unwrap_or(0.0);
            count += 1;
            sum_mib += mib;
        }
        if count > 0 {
            return Some((count, sum_mib / 1024.0, is_spark));
        }
    }
    None
}

fn detect_windows_ram_gb() -> Option<f64> {
    if std::env::consts::OS != "windows" {
        return None;
    }
    let out = Command::new("powershell")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let bytes = String::from_utf8_lossy(&out.stdout).trim().parse::<f64>().ok()?;
    Some(bytes / 1_073_741_824.0)
}

/// Compute a deterministic 3-tier `VRAM -> Host RAM -> Striped NVMe` placement & throughput plan.
pub fn compute_three_tier_plan(
    model: &CatalogModel,
    hw: &HardwareEnvelope,
    kv_slots: u32,
    ctx_per_slot: u32,
) -> ThreeTierPlan {
    let slots = kv_slots.clamp(1, 16);
    let ctx = ctx_per_slot.max(512);
    let kv_bytes_total = (model.kv_bytes_per_token as f64) * (slots as f64) * (ctx as f64);
    let kv_gb = kv_bytes_total / 1_073_741_824.0;

    let total_experts = (model.num_layers as u64) * (model.routed_experts as u64);
    let expert_gb = (model.expert_bytes_int4 as f64) / 1_073_741_824.0;

    let fixed_gpu_need = model.shared_trunk_gb + kv_gb;
    let free_vram_for_experts = (hw.vram_gb - fixed_gpu_need - 0.75).max(0.0);
    let trunk_spill_ram = (fixed_gpu_need - hw.vram_gb.max(0.0)).max(0.0);
    let free_ram_for_experts = (hw.ram_gb - trunk_spill_ram - 2.0).max(0.0);

    let (hot_vram, warm_ram, cold_nvme) = if model.routed_experts <= 1 || expert_gb <= 0.0 {
        (total_experts, 0, 0)
    } else {
        let cap_vram = ((free_vram_for_experts / expert_gb).floor() as u64).min(total_experts);
        let rem_after_vram = total_experts - cap_vram;
        let cap_ram = ((free_ram_for_experts / expert_gb).floor() as u64).min(rem_after_vram);
        let rem_nvme = rem_after_vram - cap_ram;
        (cap_vram, cap_ram, rem_nvme)
    };

    let resident_frac = if total_experts == 0 {
        1.0
    } else {
        ((hot_vram + warm_ram) as f64) / (total_experts as f64)
    };
    let learned_hit = if resident_frac >= 0.999 {
        1.0
    } else {
        (resident_frac.powf(0.58) * 1.06).clamp(0.05, 0.995)
    };

    let fam = find_family(model.family_id);
    let fam_supports_mtp = fam.map(|f| f.supports_mtp).unwrap_or(false);
    let (mtp_enabled, mtp_reason) = if !fam_supports_mtp {
        (false, "Family uses single-head decoding (no MTP draft head)".to_string())
    } else if learned_hit >= 0.88 {
        (
            true,
            format!(
                "ACTIVE: resident expert hit {:.1}% >= 88.0% MTP break-even gate (+24% tok/s boost)",
                learned_hit * 100.0
            ),
        )
    } else {
        (
            false,
            format!(
                "AUTO-GATED OFF: expert hit {:.1}% < 88.0% threshold (prevents -32% cold-SSD draft thrash)",
                learned_hit * 100.0
            ),
        )
    };

    let duty_cap_pct: u8 = if hw.eco_thermal_quiet { 72 } else { 100 };
    let duty_scale = (duty_cap_pct as f64) / 100.0;

    let active_expert_bytes_per_tok =
        (model.num_layers as f64) * (model.top_k_experts as f64) * (model.expert_bytes_int4 as f64);
    let active_gb_per_tok =
        (model.shared_trunk_gb * 0.55) + (active_expert_bytes_per_tok / 1_073_741_824.0);

    let gpu_bw_gbps = hw.gpu_mem_bw_gbps.max(96.0);
    let ram_bw_gbps = match hw.profile {
        HalProfile::AppleSiliconUma => gpu_bw_gbps,
        HalProfile::NvidiaDgxSparkUma => 273.0,
        HalProfile::AmdRocmVulkan if hw.gpu_mem_bw_gbps <= 300.0 => 256.0,
        _ => 92.0,
    };
    let nvme_bw_gbps = (hw.aggregate_nvme_gbps * duty_scale).max(1.5);

    let vram_frac = if total_experts == 0 {
        1.0
    } else {
        (hot_vram as f64) / (total_experts as f64)
    };
    let ram_frac = (learned_hit - vram_frac).max(0.0);
    let nvme_frac = (1.0 - learned_hit).max(0.0);

    let step_sec = (active_gb_per_tok * vram_frac) / gpu_bw_gbps
        + (active_gb_per_tok * ram_frac) / ram_bw_gbps
        + (active_gb_per_tok * nvme_frac) / nvme_bw_gbps
        + 0.0012;
    let mtp_mult = if mtp_enabled { 1.24 } else { 1.0 };
    let decode_tok_s = ((1.0 / step_sec) * mtp_mult * duty_scale).clamp(0.8, 360.0);
    let prefill_tok_s = (decode_tok_s * 14.5).clamp(18.0, 4400.0);

    let fit_verdict = if cold_nvme == 0 && trunk_spill_ram == 0.0 {
        "TIER_0_1_ALL_RESIDENT (Zero NVMe Cold Reads)"
    } else if learned_hit >= 0.70 {
        "TIER_2_HYBRID_SWEETSPOT (Hot VRAM + Warm RAM + Striped NVMe)"
    } else if hw.ram_gb + hw.vram_gb >= model.shared_trunk_gb + 2.0 {
        "TIER_2_NVME_STREAMING (Frontier Scale on Local SSD)"
    } else {
        "INSUFFICIENT_TRUNK_RAM (Increase RAM or choose Flash/REAP preset)"
    };

    ThreeTierPlan {
        model_slug: model.slug.to_string(),
        display_name: model.display_name.to_string(),
        family_id: model.family_id.to_string(),
        hal_profile: hw.profile.slug().to_string(),
        total_routed_experts: total_experts,
        hot_vram_experts: hot_vram,
        warm_ram_experts: warm_ram,
        cold_nvme_experts: cold_nvme,
        vram_used_gb: round2((fixed_gpu_need + (hot_vram as f64) * expert_gb).min(hw.vram_gb)),
        ram_used_gb: round2(trunk_spill_ram + (warm_ram as f64) * expert_gb),
        nvme_streamed_gb: round2((cold_nvme as f64) * expert_gb),
        kv_slots: slots,
        context_tokens_per_slot: ctx,
        kv_budget_gb: round2(kv_gb),
        learned_cache_hit_pct: round2(learned_hit * 100.0),
        mtp_enabled,
        mtp_gate_reason: mtp_reason,
        nvme_duty_cap_pct: duty_cap_pct,
        estimated_decode_tok_s: round2(decode_tok_s),
        estimated_prefill_tok_s: round2(prefill_tok_s),
        fit_verdict,
    }
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}
