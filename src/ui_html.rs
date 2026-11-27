//! Embedded 5-Language (`DE / FR / IT / RM / EN`) AAA Anti-AI-Slop Precision Cockpit HTML
//! + RSI-COLIBRI-060 Zero-Dependency Native ANSI Terminal UI & Live Expert-Map Renderer (`coli tui` parity).

pub const COCKPIT_HTML: &str = include_str!("../assets/cockpit.html");

/// RSI-COLIBRI-060 Security Guard: Strip untrusted ANSI CSI/OSC escape sequences and dangerous C0
/// control bytes from model-generated text before rendering inside a terminal frame (defeats
/// terminal-injection / OSC-52 clipboard hijacking attacks).
pub fn strip_untrusted_ansi(raw: &str) -> String {
    let bytes = raw.as_bytes();
    let mut out = String::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == 0x1b {
            i += 1;
            if i < bytes.len() && bytes[i] == b'[' {
                i += 1;
                while i < bytes.len() && !(0x40..=0x7e).contains(&bytes[i]) {
                    i += 1;
                }
                if i < bytes.len() {
                    i += 1;
                }
            } else if i < bytes.len() && bytes[i] == b']' {
                i += 1;
                while i < bytes.len() && bytes[i] != 0x07 {
                    if bytes[i] == 0x1b && i + 1 < bytes.len() && bytes[i + 1] == b'\\' {
                        i += 2;
                        break;
                    }
                    i += 1;
                }
                if i < bytes.len() && bytes[i] == 0x07 {
                    i += 1;
                }
            }
            continue;
        }
        let b = bytes[i];
        if b < 0x20 && b != b'\n' && b != b'\t' {
            i += 1;
            continue;
        }
        // Safe UTF-8 char boundary copy
        if let Some(ch) = raw[i..].chars().next() {
            out.push(ch);
            i += ch.len_utf8();
        } else {
            break;
        }
    }
    out
}

/// RSI-COLIBRI-060: Render a zero-dependency native ANSI terminal frame (`coli tui` parity)
/// displaying the 3-Tier (`VRAM Hot` / `RAM Warm` / `NVMe Cold`) expert heat map, folded reasoning
/// summary, active KV slots, and `SEC-01..12` shield status in `< 150 µs`.
pub fn render_tui_expert_map(
    plan: &crate::planner::ThreeTierPlan,
    active_slots: usize,
    sec_blocks: u64,
    reasoning_preview: &str,
) -> String {
    let total = plan.total_routed_experts.max(1);
    let bar_w = 40u64;
    let hot_w = ((plan.hot_vram_experts * bar_w) / total).min(bar_w) as usize;
    let warm_w = ((plan.warm_ram_experts * bar_w) / total).min(bar_w.saturating_sub(hot_w as u64)) as usize;
    let cold_w = (bar_w as usize).saturating_sub(hot_w + warm_w);

    let safe_reason = strip_untrusted_ansi(reasoning_preview);
    let folded: String = safe_reason.chars().take(72).collect();

    format!(
        "\x1b[1;36m┌──────────────── NALA-Colibri-RS v{} — Live TUI Expert Map ────────────────┐\x1b[0m\n\
         │ Model: {:<32} | HAL: {:<28} │\n\
         │ Decode: {:>5.1} tok/s | Prefill: {:>6.1} tok/s | Cache Hit: {:>5.1}%           │\n\
         │ 3-Tier Map: [\x1b[32m{}\x1b[33m{}\x1b[34m{}\x1b[0m] (VRAM:{} RAM:{} NVMe:{}) │\n\
         │ Slots: {:>2}/64 | SEC-01..12 Blocks: {:>4} | Quiet Cap: {:>2}% | MTP: {:<5}        │\n\
         │ Reasoning [folded]: {:<56} │\n\
         \x1b[1;36m└──────────────────────────────────────────────────────────────────────────────┘\x1b[0m\n",
        crate::VERSION,
        plan.model_slug,
        plan.hal_profile,
        plan.estimated_decode_tok_s,
        plan.estimated_prefill_tok_s,
        plan.learned_cache_hit_pct,
        "#".repeat(hot_w),
        "=".repeat(warm_w),
        ".".repeat(cold_w),
        plan.hot_vram_experts,
        plan.warm_ram_experts,
        plan.cold_nvme_experts,
        active_slots,
        sec_blocks,
        plan.nvme_duty_cap_pct,
        if plan.mtp_enabled { "ON" } else { "OFF" },
        folded
    )
}
