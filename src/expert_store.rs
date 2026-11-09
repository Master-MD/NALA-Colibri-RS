//! `RSI-COLIBRI-002`: Zero-Lock RAII Lease-Safe 3-Tier `ExpertStore`, `.coli_usage` FNV-1a Heat Tracker,
//! `.coli_pairs` 2-Step Markov Prefetcher, Weighted Dual-NVMe Stripe Mapper, `KvPrefixCache`,
//! `ColiParkVault` (`COLI_PARK` Short-Request Conversation Parking), and 16-Slot `GenerationScheduler`.
//!
//! Eliminates C manual `release()` use-after-free / double-release bugs via Rust RAII `Drop` leases.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, Ordering};

pub const FNV1A_64_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
pub const FNV1A_64_PRIME: u64 = 0x0000_0100_0000_01b3;

#[inline]
pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h = FNV1A_64_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV1A_64_PRIME);
    }
    h
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTier {
    HotVram,
    WarmHostRam,
    ColdStripedNvme { stripe_idx: u8 },
}

impl MemoryTier {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HotVram => "VRAM_HOT",
            Self::WarmHostRam => "HOST_RAM_WARM",
            Self::ColdStripedNvme { .. } => "NVME_COLD_STRIPED",
        }
    }
}

pub struct ExpertSlot {
    pub layer: u16,
    pub expert_id: u16,
    pub tier: MemoryTier,
    pub active_leases: AtomicU32,
    pub hit_count: AtomicU32,
    pub pinned_hot: bool,
}

/// Compile-time safe RAII lease over an MoE expert slot (`expert_store.h` replacement).
/// While `ExpertLease<'a>` is alive, the slot cannot be evicted from RAM/VRAM (`active_leases > 0`).
pub struct ExpertLease<'a> {
    slot: &'a ExpertSlot,
}

impl<'a> ExpertLease<'a> {
    #[inline]
    pub fn tier(&self) -> MemoryTier {
        self.slot.tier
    }

    #[inline]
    pub fn active_leases(&self) -> u32 {
        self.slot.active_leases.load(Ordering::Acquire)
    }
}

impl Drop for ExpertLease<'_> {
    #[inline]
    fn drop(&mut self) {
        self.slot.active_leases.fetch_sub(1, Ordering::Release);
    }
}

/// 3-Tier Hierarchical MoE Expert Store + `.coli_usage` & `.coli_pairs` Markov Lookahead Engine.
pub struct ExpertStore {
    pub model_signature_fnv1a: u64,
    pub num_layers: u16,
    pub num_experts: u16,
    pub nvme_stripe_weights: [u16; 2], // e.g., [65, 35] proportional to SSD sequential read speed
    slots: Vec<ExpertSlot>,
    /// Compact 2-step Markov transition table: `(layer, src_expert) -> predicted_next_layer_expert`
    markov_next: Vec<u16>,
}

impl ExpertStore {
    pub fn new(
        model_id: &str,
        num_layers: u16,
        num_experts: u16,
        hot_vram_budget: usize,
        warm_ram_budget: usize,
        stripe0_gbps: f64,
        stripe1_gbps: f64,
    ) -> Self {
        let sig = fnv1a_64(model_id.as_bytes());
        let total = (num_layers as usize) * (num_experts as usize);
        let sum_bw = (stripe0_gbps + stripe1_gbps).max(0.1);
        let w0 = ((stripe0_gbps / sum_bw) * 100.0).round().clamp(1.0, 100.0) as u16;
        let w1 = 100u16.saturating_sub(w0);

        let mut slots = Vec::with_capacity(total);
        for idx in 0..total {
            let layer = (idx / (num_experts as usize)) as u16;
            let expert_id = (idx % (num_experts as usize)) as u16;
            let (tier, pinned) = if idx < hot_vram_budget {
                (MemoryTier::HotVram, true)
            } else if idx < hot_vram_budget.saturating_add(warm_ram_budget) {
                (MemoryTier::WarmHostRam, false)
            } else {
                // Weighted dual-NVMe stripe assignment proportional to measured SSD bandwidth
                let bucket = ((idx as u64).wrapping_mul(37) % 100) as u16;
                let stripe_idx = if w1 == 0 || bucket < w0 { 0 } else { 1 };
                (MemoryTier::ColdStripedNvme { stripe_idx }, false)
            };
            slots.push(ExpertSlot {
                layer,
                expert_id,
                tier,
                active_leases: AtomicU32::new(0),
                hit_count: AtomicU32::new(0),
                pinned_hot: pinned,
            });
        }

        Self {
            model_signature_fnv1a: sig,
            num_layers,
            num_experts,
            nvme_stripe_weights: [w0, w1],
            slots,
            markov_next: vec![0u16; total],
        }
    }

    #[inline]
    fn flat_index(&self, layer: u16, expert_id: u16) -> Option<usize> {
        if layer >= self.num_layers || expert_id >= self.num_experts {
            return None;
        }
        Some((layer as usize) * (self.num_experts as usize) + (expert_id as usize))
    }

    /// Acquire an RAII lease on `(layer, expert_id)` (`route -> union -> place` step) and record `.coli_usage` heat.
    #[inline]
    pub fn acquire(&self, layer: u16, expert_id: u16) -> Option<ExpertLease<'_>> {
        let idx = self.flat_index(layer, expert_id)?;
        let slot = &self.slots[idx];
        slot.active_leases.fetch_add(1, Ordering::AcqRel);
        slot.hit_count.fetch_add(1, Ordering::Relaxed);
        Some(ExpertLease { slot })
    }

    /// Check whether `(layer, expert_id)` can be safely evicted right now (`!pinned && active_leases == 0`).
    #[inline]
    pub fn can_evict(&self, layer: u16, expert_id: u16) -> bool {
        let Some(idx) = self.flat_index(layer, expert_id) else {
            return false;
        };
        let slot = &self.slots[idx];
        !slot.pinned_hot && slot.active_leases.load(Ordering::Acquire) == 0
    }

    /// Record cross-layer expert firing transition `(layer, cur_e) -> (layer+1, next_e)` (`.coli_pairs` learn step).
    #[inline]
    pub fn record_markov_pair(&mut self, layer: u16, cur_expert: u16, next_expert: u16) {
        if layer + 1 < self.num_layers && next_expert < self.num_experts {
            if let Some(idx) = self.flat_index(layer, cur_expert) {
                self.markov_next[idx] = next_expert;
            }
        }
    }

    /// Predict the next layer's hot expert during current-layer attention (`overlap` lookahead step).
    #[inline]
    pub fn predict_next_layer_expert(&self, layer: u16, cur_expert: u16) -> Option<(u16, u16)> {
        if layer + 1 >= self.num_layers {
            return None;
        }
        let idx = self.flat_index(layer, cur_expert)?;
        Some((layer + 1, self.markov_next[idx]))
    }

    /// Serialize `.coli_usage` header (`COLI_U01` + FNV-1a 64-bit model signature + u16 dims + u32 hit counts).
    pub fn encode_coli_usage_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(20 + self.slots.len() * 4);
        out.extend_from_slice(b"COLI_U01");
        out.extend_from_slice(&self.model_signature_fnv1a.to_le_bytes());
        out.extend_from_slice(&self.num_layers.to_le_bytes());
        out.extend_from_slice(&self.num_experts.to_le_bytes());
        for s in &self.slots {
            out.extend_from_slice(&s.hit_count.load(Ordering::Relaxed).to_le_bytes());
        }
        out
    }
}

/// Exact Token-Prefix KV Reuse Tracker (`kv_prefix.h` parity) with `taint` rollback safety guard.
#[derive(Debug, Clone)]
pub struct KvPrefixCache {
    fed_tokens: Vec<u32>,
    tainted: bool,
}

impl KvPrefixCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            fed_tokens: Vec::with_capacity(capacity),
            tainted: false,
        }
    }

    /// Compute how many leading prompt tokens match the live KV cache (keeping >= 1 token to re-feed for logits).
    pub fn reusable_prefix_len(&self, prompt_tokens: &[u32]) -> usize {
        if self.tainted || prompt_tokens.len() <= 1 || self.fed_tokens.is_empty() {
            return 0;
        }
        let max_reuse = prompt_tokens.len() - 1;
        let limit = self.fed_tokens.len().min(max_reuse);
        let mut matched = 0usize;
        while matched < limit && self.fed_tokens[matched] == prompt_tokens[matched] {
            matched += 1;
        }
        matched
    }

    pub fn commit_tokens(&mut self, start_pos: usize, tokens: &[u32]) {
        if self.tainted {
            self.fed_tokens.clear();
            self.tainted = false;
        }
        if start_pos < self.fed_tokens.len() {
            self.fed_tokens.truncate(start_pos);
        }
        self.fed_tokens.extend_from_slice(tokens);
    }

    /// Mark KV state tainted after speculative MTP rollback or non-causal overwrite (forces clean rebuild).
    pub fn mark_tainted(&mut self) {
        self.tainted = true;
    }

    pub fn is_tainted(&self) -> bool {
        self.tainted
    }
}

/// `COLI_PARK` Short-Request Conversation Parking Vault (`park.h` parity).
/// Saves only the first `n` KV rows (`n = short_prompt_tokens + max_tokens`) of a long 32k+ chat
/// before running a brief side-request (e.g., chat title / tool router), then restores in place.
#[derive(Debug, Clone)]
pub struct ColiParkVault {
    row_width_f32: usize,
    saved_prefix_tokens: Vec<u32>,
    saved_kv_rows: Vec<f32>,
}

impl ColiParkVault {
    pub fn new(row_width_f32: usize) -> Self {
        Self {
            row_width_f32,
            saved_prefix_tokens: Vec::new(),
            saved_kv_rows: Vec::new(),
        }
    }

    pub fn park_save(&mut self, live_kv: &[f32], live_tokens: &[u32], overwrite_rows: usize) {
        let rows = overwrite_rows.min(live_tokens.len());
        let floats = rows.saturating_mul(self.row_width_f32).min(live_kv.len());
        self.saved_prefix_tokens.clear();
        self.saved_prefix_tokens.extend_from_slice(&live_tokens[..rows]);
        self.saved_kv_rows.clear();
        self.saved_kv_rows.extend_from_slice(&live_kv[..floats]);
    }

    pub fn park_restore(&mut self, live_kv: &mut [f32], live_tokens: &mut [u32]) -> usize {
        let rows = self.saved_prefix_tokens.len().min(live_tokens.len());
        let floats = self.saved_kv_rows.len().min(live_kv.len());
        live_tokens[..rows].copy_from_slice(&self.saved_prefix_tokens[..rows]);
        live_kv[..floats].copy_from_slice(&self.saved_kv_rows[..floats]);
        self.saved_prefix_tokens.clear();
        self.saved_kv_rows.clear();
        rows
    }
}

/// Bounded 16-Slot Concurrent KV Admission Scheduler (`GenerationScheduler` parity).
pub struct GenerationScheduler {
    max_slots: usize,
    free_slots: VecDeque<usize>,
    wait_queue_limit: usize,
    queued: usize,
}

impl GenerationScheduler {
    pub fn new(max_slots: usize, wait_queue_limit: usize) -> Self {
        let clamped = max_slots.clamp(1, 16);
        let mut free = VecDeque::with_capacity(clamped);
        for s in 0..clamped {
            free.push_back(s);
        }
        Self {
            max_slots: clamped,
            free_slots: free,
            wait_queue_limit,
            queued: 0,
        }
    }

    pub fn try_acquire_slot(&mut self) -> Result<usize, &'static str> {
        if let Some(slot) = self.free_slots.pop_front() {
            Ok(slot)
        } else if self.queued >= self.wait_queue_limit {
            Err("HTTP_429_QUEUE_FULL")
        } else {
            self.queued += 1;
            Err("QUEUED_FIFO")
        }
    }

    pub fn release_slot(&mut self, slot: usize) {
        if slot < self.max_slots && !self.free_slots.contains(&slot) {
            if self.queued > 0 {
                self.queued -= 1;
            }
            self.free_slots.push_back(slot);
        }
    }

    pub fn active_count(&self) -> usize {
        self.max_slots - self.free_slots.len()
    }
}
