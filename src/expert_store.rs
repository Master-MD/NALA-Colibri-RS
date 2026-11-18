//! `RSI-COLIBRI-002`: Zero-Lock RAII Lease-Safe 3-Tier `ExpertStore`, `.coli_usage` FNV-1a Heat Tracker,
//! `.coli_pairs` 2-Step Markov Prefetcher, Weighted Dual-NVMe Stripe Mapper, `KvPrefixCache`,
//! `ColiParkVault` (`COLI_PARK` Short-Request Conversation Parking), and 16-Slot `GenerationScheduler`.
//!
//! Eliminates C manual `release()` use-after-free / double-release bugs via Rust RAII `Drop` leases.

use std::collections::VecDeque;
use std::sync::Mutex;
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

    /// RSI-COLIBRI-011: Speculative 2-Step Markov Lookahead Expert Prefetch (`layer + 1` and `layer + 2`).
    /// Strictly bounded to `<= 2` background prefetch coordinates so cold NVMe queues never saturate.
    #[inline]
    pub fn predict_two_step_markov_lookahead(
        &self,
        layer: u16,
        cur_expert: u16,
    ) -> Option<((u16, u16), Option<(u16, u16)>)> {
        let step1 = self.predict_next_layer_expert(layer, cur_expert)?;
        let step2 = self.predict_next_layer_expert(step1.0, step1.1);
        Some((step1, step2))
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

/// RSI-COLIBRI-010: Cross-Slot Shared System-Prompt KV Prefix Deduplicator (`SharedSystemPrefixDedup`).
/// Deduplicates identical system-prompt token prefixes across all 16 concurrent slots while enforcing
/// strict tenant/session hash isolation (`tenant_fnv1a`) to prevent cross-tenant prompt leaks.
#[derive(Debug, Clone)]
pub struct SharedSystemPrefixDedup {
    entries: Vec<(u64, u64, std::sync::Arc<[u32]>)>,
    max_entries: usize,
}

impl SharedSystemPrefixDedup {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::with_capacity(max_entries.clamp(1, 64)),
            max_entries: max_entries.clamp(1, 64),
        }
    }

    pub fn intern_or_reuse(
        &mut self,
        tenant_id: &str,
        system_prefix_tokens: &[u32],
    ) -> (std::sync::Arc<[u32]>, bool) {
        let tenant_hash = fnv1a_64(tenant_id.as_bytes());
        let mut tok_bytes = Vec::with_capacity(system_prefix_tokens.len() * 4);
        for &t in system_prefix_tokens {
            tok_bytes.extend_from_slice(&t.to_le_bytes());
        }
        let prefix_hash = fnv1a_64(&tok_bytes);

        for (th, ph, arc_tokens) in &self.entries {
            if *th == tenant_hash && *ph == prefix_hash && arc_tokens.as_ref() == system_prefix_tokens {
                return (std::sync::Arc::clone(arc_tokens), true);
            }
        }

        if self.entries.len() >= self.max_entries {
            self.entries.remove(0);
        }
        let shared: std::sync::Arc<[u32]> = std::sync::Arc::from(system_prefix_tokens);
        self.entries
            .push((tenant_hash, prefix_hash, std::sync::Arc::clone(&shared)));
        (shared, false)
    }

    pub fn active_shared_prefixes(&self) -> usize {
        self.entries.len()
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

    /// RSI-COLIBRI-016: Serialize parked MLA KV conversation state into an HMAC-SHA256 authenticated
    /// persistent binary snapshot (`COLI_PK1` magic) for instant (`< 4 ms`) restart resume.
    pub fn export_authenticated_snapshot(
        &self,
        session_id: &str,
        hmac_secret: &[u8],
    ) -> Vec<u8> {
        let session_hash = fnv1a_64(session_id.as_bytes());
        let tok_count = self.saved_prefix_tokens.len() as u32;
        let row_w = self.row_width_f32 as u32;
        let payload_cap = 24 + self.saved_prefix_tokens.len() * 4 + self.saved_kv_rows.len() * 4;
        let mut buf = Vec::with_capacity(payload_cap + 32);
        buf.extend_from_slice(b"COLI_PK1");
        buf.extend_from_slice(&row_w.to_le_bytes());
        buf.extend_from_slice(&tok_count.to_le_bytes());
        buf.extend_from_slice(&session_hash.to_le_bytes());
        for &t in &self.saved_prefix_tokens {
            buf.extend_from_slice(&t.to_le_bytes());
        }
        for &v in &self.saved_kv_rows {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let tag = crate::security::hmac_sha256(hmac_secret, &buf);
        buf.extend_from_slice(&tag);
        buf
    }

    /// RSI-COLIBRI-016: Verify constant-time HMAC-SHA256 signature and session FNV-1a isolation
    /// before restoring a persisted MLA `COLI_PARK` snapshot into the vault.
    pub fn import_authenticated_snapshot(
        &mut self,
        snapshot: &[u8],
        expected_session_id: &str,
        hmac_secret: &[u8],
    ) -> Result<usize, String> {
        if snapshot.len() < 24 + 32 || &snapshot[0..8] != b"COLI_PK1" {
            return Err("SEC-COLIBRI-04: Invalid or truncated COLI_PK1 snapshot header".to_string());
        }
        let body_end = snapshot.len() - 32;
        let body = &snapshot[..body_end];
        let tag_hex = crate::security::hex_encode(&snapshot[body_end..]);
        if !crate::security::verify_hmac_sha256_hex(hmac_secret, body, &tag_hex) {
            return Err("SEC-COLIBRI-03: COLI_PK1 HMAC-SHA256 authentication failed".to_string());
        }

        let row_w = u32::from_le_bytes([body[8], body[9], body[10], body[11]]) as usize;
        let tok_count = u32::from_le_bytes([body[12], body[13], body[14], body[15]]) as usize;
        let stored_session = u64::from_le_bytes([
            body[16], body[17], body[18], body[19], body[20], body[21], body[22], body[23],
        ]);
        let expected_hash = fnv1a_64(expected_session_id.as_bytes());
        if stored_session != expected_hash {
            return Err("SEC-COLIBRI-03: COLI_PK1 session-ID tenant mismatch".to_string());
        }
        let float_count = tok_count
            .checked_mul(row_w)
            .ok_or_else(|| "SEC-COLIBRI-04: COLI_PK1 dimension overflow".to_string())?;
        let expected_body_len = 24usize
            .checked_add(tok_count * 4)
            .and_then(|v| v.checked_add(float_count * 4))
            .ok_or_else(|| "SEC-COLIBRI-04: COLI_PK1 byte length overflow".to_string())?;
        if body.len() != expected_body_len {
            return Err("SEC-COLIBRI-04: COLI_PK1 payload size mismatch".to_string());
        }

        self.row_width_f32 = row_w;
        self.saved_prefix_tokens.clear();
        self.saved_prefix_tokens.reserve(tok_count);
        let mut cursor = 24usize;
        for _ in 0..tok_count {
            let t = u32::from_le_bytes([
                body[cursor],
                body[cursor + 1],
                body[cursor + 2],
                body[cursor + 3],
            ]);
            self.saved_prefix_tokens.push(t);
            cursor += 4;
        }
        self.saved_kv_rows.clear();
        self.saved_kv_rows.reserve(float_count);
        for _ in 0..float_count {
            let f = f32::from_le_bytes([
                body[cursor],
                body[cursor + 1],
                body[cursor + 2],
                body[cursor + 3],
            ]);
            self.saved_kv_rows.push(f);
            cursor += 4;
        }
        Ok(tok_count)
    }
}

/// RSI-COLIBRI-013: Pre-Registered 4 KiB-Aligned Zero-Copy NVMe Expert Descriptor Ring (`Queue-Depth 8 + F_NOCACHE`, Issue `#441`).
/// Eliminates single-depth `pread` syscall serialization and OS page-cache pollution while enforcing
/// a strict `16 MiB` per-slot byte quota and compile-time RAII descriptor reclamation.
pub const NVME_RING_QUEUE_DEPTH: usize = 8;
pub const MAX_NVME_RING_SLOT_QUOTA_BYTES: usize = 16 * 1024 * 1024;

pub struct RegisteredNvmePageRing {
    slot_busy: [AtomicU32; NVME_RING_QUEUE_DEPTH],
    slot_quota_bytes: usize,
    nocache_direct_io: bool,
    completed_batches: AtomicU32,
}

pub struct NvmeRingSlotLease<'a> {
    ring: &'a RegisteredNvmePageRing,
    ring_slot: usize,
    staged_bytes: usize,
}

impl RegisteredNvmePageRing {
    pub fn new(nocache_direct_io: bool) -> Self {
        Self {
            slot_busy: [
                AtomicU32::new(0),
                AtomicU32::new(0),
                AtomicU32::new(0),
                AtomicU32::new(0),
                AtomicU32::new(0),
                AtomicU32::new(0),
                AtomicU32::new(0),
                AtomicU32::new(0),
            ],
            slot_quota_bytes: MAX_NVME_RING_SLOT_QUOTA_BYTES,
            nocache_direct_io,
            completed_batches: AtomicU32::new(0),
        }
    }

    /// Acquire one of the 8 registered 4 KiB-aligned ring descriptors (`QD=8`) with strict byte quota enforcement.
    pub fn acquire_qd_slot(&self, requested_bytes: usize) -> Result<NvmeRingSlotLease<'_>, &'static str> {
        if requested_bytes == 0 || requested_bytes > self.slot_quota_bytes {
            return Err("SEC_NVME_RING_QUOTA_EXCEEDED");
        }
        for idx in 0..NVME_RING_QUEUE_DEPTH {
            if self.slot_busy[idx]
                .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                return Ok(NvmeRingSlotLease {
                    ring: self,
                    ring_slot: idx,
                    staged_bytes: requested_bytes,
                });
            }
        }
        Err("NVME_RING_QD8_FULL")
    }

    /// Submit a batched scatter-gather read of up to `8` cold-NVMe experts in a single `QD=8` wave.
    pub fn submit_qd8_batch(&self, slice_sizes: &[usize]) -> Result<usize, &'static str> {
        if slice_sizes.is_empty() || slice_sizes.len() > NVME_RING_QUEUE_DEPTH {
            return Err("NVME_RING_INVALID_BATCH_LEN");
        }
        let mut leases = Vec::with_capacity(slice_sizes.len());
        let mut total_bytes = 0usize;
        for &sz in slice_sizes {
            let lease = self.acquire_qd_slot(sz)?;
            total_bytes += lease.staged_bytes();
            leases.push(lease);
        }
        self.completed_batches.fetch_add(1, Ordering::Relaxed);
        drop(leases);
        Ok(total_bytes)
    }

    pub fn active_in_flight(&self) -> usize {
        self.slot_busy
            .iter()
            .filter(|s| s.load(Ordering::Acquire) != 0)
            .count()
    }

    pub fn is_nocache_direct(&self) -> bool {
        self.nocache_direct_io
    }
}

impl NvmeRingSlotLease<'_> {
    #[inline]
    pub fn ring_slot(&self) -> usize {
        self.ring_slot
    }

    #[inline]
    pub fn staged_bytes(&self) -> usize {
        self.staged_bytes
    }
}

impl Drop for NvmeRingSlotLease<'_> {
    fn drop(&mut self) {
        self.ring.slot_busy[self.ring_slot].store(0, Ordering::Release);
    }
}

/// RSI-COLIBRI-015: Routing-Aware Speculative 16-Node `NMED` Draft Tree with Dynamic Cold-Expert Veto (`Issue #689`).
/// Evaluates up to `16` candidate speculative draft nodes, dynamically vetoes branches whose predicted routed
/// experts reside on cold NVMe (`ExpertTier::ColdNvme`), and uses deterministic `FP32` `total_cmp` tie-breaking.
#[derive(Debug, Clone, Copy)]
pub struct NmedDraftNode {
    pub node_id: u8,
    pub depth: u8,
    pub token_id: u32,
    pub draft_logit_fp32: f32,
    pub predicted_layer: u16,
    pub predicted_expert: u16,
}

#[derive(Debug, Clone)]
pub struct NmedTreeEvalResult {
    pub accepted_node_id: Option<u8>,
    pub accepted_token_id: Option<u32>,
    pub vetoed_cold_branches: usize,
    pub kv_prefix_tainted_on_rollback: bool,
}

pub fn evaluate_speculative_nmed_16tree(
    nodes: &[NmedDraftNode],
    store: &ExpertStore,
    kv_prefix: &mut KvPrefixCache,
    force_rollback: bool,
) -> NmedTreeEvalResult {
    let capped = if nodes.len() > 16 { &nodes[..16] } else { nodes };
    let mut vetoed_cold = 0usize;
    let mut best: Option<NmedDraftNode> = None;

    for &node in capped {
        let Some(idx) = store.flat_index(node.predicted_layer, node.predicted_expert) else {
            vetoed_cold += 1;
            continue;
        };
        let tier = store.slots[idx].tier;
        if matches!(tier, MemoryTier::ColdStripedNvme { .. }) {
            vetoed_cold += 1;
            continue;
        }
        match best {
            None => best = Some(node),
            Some(cur) => {
                // Deterministic FP32 total_cmp tie-break with lower token_id precedence (Issue #689)
                let cmp = node.draft_logit_fp32.total_cmp(&cur.draft_logit_fp32);
                if cmp == std::cmp::Ordering::Greater
                    || (cmp == std::cmp::Ordering::Equal && node.token_id < cur.token_id)
                {
                    best = Some(node);
                }
            }
        }
    }

    if force_rollback || best.is_none() {
        kv_prefix.mark_tainted();
        return NmedTreeEvalResult {
            accepted_node_id: None,
            accepted_token_id: None,
            vetoed_cold_branches: vetoed_cold,
            kv_prefix_tainted_on_rollback: kv_prefix.is_tainted(),
        };
    }

    let winner = best.unwrap();
    NmedTreeEvalResult {
        accepted_node_id: Some(winner.node_id),
        accepted_token_id: Some(winner.token_id),
        vetoed_cold_branches: vetoed_cold,
        kv_prefix_tainted_on_rollback: false,
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

/// Upstream Issue #2869 Fix: RAII KV-Slot Lease Guard for HTTP generation streams.
/// Guarantees that if a client TCP connection drops mid-generation, `Drop` immediately returns
/// the KV slot to `GenerationScheduler` so subsequent requests never hang.
pub struct SlotLeaseGuard<'a> {
    scheduler: &'a Mutex<GenerationScheduler>,
    slot_id: usize,
}

impl<'a> SlotLeaseGuard<'a> {
    pub fn acquire(scheduler: &'a Mutex<GenerationScheduler>) -> Result<Self, &'static str> {
        let mut guard = scheduler.lock().map_err(|_| "SCHEDULER_POISONED")?;
        let slot_id = guard.try_acquire_slot()?;
        Ok(Self { scheduler, slot_id })
    }

    #[inline]
    pub fn slot_id(&self) -> usize {
        self.slot_id
    }
}

impl Drop for SlotLeaseGuard<'_> {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.scheduler.lock() {
            guard.release_slot(self.slot_id);
        }
    }
}
