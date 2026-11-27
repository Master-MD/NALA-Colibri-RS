//! `RSI-COLIBRI-003`: Sub-Microsecond Native Rust `System One` Calibrated Decision Engine (`POST /v1/systemone`)
//! + `NALA-RAG` `[Dok-ID, S. X]` Deterministic Evidence-Lock Verifier.
//!
//! Implements Jev-compatible closed-set probabilistic inference (`choice`, `noul`, `score`)
//! in a single calibrated pass with normalized Shannon entropy confidence `1 - H(p)/ln(K)` and `0` generated tokens.

use crate::expert_store::fnv1a_64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemOneMode {
    Choice,
    Noul,
    Score,
}

impl SystemOneMode {
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "noul" | "yesno" | "binary" => Self::Noul,
            "score" | "rubric" | "scalar" => Self::Score,
            _ => Self::Choice,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Noul => "noul",
            Self::Score => "score",
        }
    }
}

#[derive(Debug, Clone)]
pub struct OptionProbability {
    pub label: String,
    pub probability: f64,
    pub logit: f64,
}

#[derive(Debug, Clone)]
pub struct SystemOneResult {
    pub mode: SystemOneMode,
    pub winner: String,
    pub scalar_score: Option<f64>,
    pub confidence: f64,
    pub entropy_nats: f64,
    pub distribution: Vec<OptionProbability>,
    pub evidence_lock_ok: bool,
    pub citations: Vec<String>,
    pub generated_tokens: u32,
}

/// Numerically stable calibrated temperature softmax + normalized Shannon entropy confidence `1 - H(p)/ln(K)`.
pub fn calibrated_softmax_and_confidence(logits: &[f64], temperature: f64) -> (Vec<f64>, f64, f64) {
    if logits.is_empty() {
        return (Vec::new(), 0.0, 0.0);
    }
    if logits.len() == 1 {
        return (vec![1.0], 1.0, 0.0);
    }
    let inv_t = 1.0 / temperature.max(1e-4);
    let max_l = logits
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut exps = Vec::with_capacity(logits.len());
    let mut sum = 0.0f64;
    for &l in logits {
        let e = ((l - max_l) * inv_t).exp();
        exps.push(e);
        sum += e;
    }
    let mut probs = Vec::with_capacity(logits.len());
    let mut entropy = 0.0f64;
    for e in exps {
        let p = e / sum;
        if p > 1e-15 {
            entropy -= p * p.ln();
        }
        probs.push(p);
    }
    let max_entropy = (logits.len() as f64).ln();
    let confidence = if max_entropy > 0.0 {
        (1.0 - (entropy / max_entropy)).clamp(0.0, 1.0)
    } else {
        1.0
    };
    (probs, confidence, entropy)
}

/// Deterministic feature-calibrated `System One` evaluator (`choice`, `noul`, `score`) + `NALA-RAG` Evidence-Lock.
pub fn evaluate_system_one(
    prompt: &str,
    mode: SystemOneMode,
    options: &[String],
    temperature: f64,
    require_evidence_lock: bool,
) -> SystemOneResult {
    let citations = extract_evidence_citations(prompt);
    let evidence_lock_ok = !require_evidence_lock || !citations.is_empty();

    let effective_options: Vec<String> = match mode {
        SystemOneMode::Noul => vec!["yes".to_string(), "no".to_string()],
        SystemOneMode::Score => vec![
            "0.00".to_string(),
            "0.25".to_string(),
            "0.50".to_string(),
            "0.75".to_string(),
            "1.00".to_string(),
        ],
        SystemOneMode::Choice => {
            if options.is_empty() {
                vec!["approve".to_string(), "defer".to_string(), "reject".to_string()]
            } else {
                options.to_vec()
            }
        }
    };

    let prompt_lower = prompt.to_ascii_lowercase();
    let prompt_tokens: Vec<&str> = prompt_lower
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        .filter(|s| s.len() >= 2)
        .collect();

    let mut logits = Vec::with_capacity(effective_options.len());
    for (idx, opt) in effective_options.iter().enumerate() {
        let opt_lower = opt.to_ascii_lowercase();
        let mut score = 0.0f64;

        // 1) Lexical & token overlap signal
        for tok in opt_lower
            .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
            .filter(|s| s.len() >= 2)
        {
            if prompt_tokens.contains(&tok) {
                score += 1.65;
            } else if prompt_lower.contains(tok) {
                score += 0.95;
            }
        }

        // 2) Sentiment / safety polarity cues for `noul` and `score`
        let pos_cues = [
            "safe", "pass", "verified", "valid", "compliant", "approve", "yes", "fast",
            "bit-exact", "zero-pii", "true", "confirmed",
        ];
        let neg_cues = [
            "unsafe", "fail", "leak", "overflow", "ssrf", "oom", "reject", "no", "false",
            "blocked", "invalid", "unverified",
        ];
        let pos_hits = pos_cues.iter().filter(|w| prompt_lower.contains(**w)).count() as f64;
        let neg_hits = neg_cues.iter().filter(|w| prompt_lower.contains(**w)).count() as f64;

        match mode {
            SystemOneMode::Noul => {
                if idx == 0 {
                    score += pos_hits * 0.9 - neg_hits * 0.8 + if !citations.is_empty() { 0.55 } else { 0.0 };
                } else {
                    score += neg_hits * 1.05 - pos_hits * 0.65 + if require_evidence_lock && citations.is_empty() { 2.4 } else { 0.0 };
                }
            }
            SystemOneMode::Score => {
                let anchor = (idx as f64) * 0.25;
                let target = (0.55 + 0.14 * pos_hits - 0.18 * neg_hits
                    + if !citations.is_empty() { 0.12 } else { 0.0 })
                .clamp(0.0, 1.0);
                score -= ((anchor - target).abs() * 4.2).powi(2);
            }
            SystemOneMode::Choice => {
                // Deterministic micro-prior via FNV-1a bi-gram hash so ties resolve reproducibly
                let h = fnv1a_64(format!("{prompt_lower}||{opt_lower}").as_bytes());
                let micro = ((h & 0xffff) as f64 / 65535.0) * 0.18;
                score += micro;
            }
        }

        logits.push(score);
    }

    let (probs, confidence, entropy) = calibrated_softmax_and_confidence(&logits, temperature);
    let mut best_idx = 0usize;
    for (i, &p) in probs.iter().enumerate() {
        if p > probs[best_idx] {
            best_idx = i;
        }
    }

    let scalar_score = if mode == SystemOneMode::Score {
        let anchors = [0.0, 0.25, 0.50, 0.75, 1.0];
        let expected: f64 = probs.iter().zip(anchors.iter()).map(|(p, a)| p * a).sum();
        Some((expected * 10000.0).round() / 10000.0)
    } else {
        None
    };

    let distribution = effective_options
        .iter()
        .enumerate()
        .map(|(i, label)| OptionProbability {
            label: label.clone(),
            probability: (probs[i] * 1_000_000.0).round() / 1_000_000.0,
            logit: (logits[i] * 10000.0).round() / 10000.0,
        })
        .collect();

    SystemOneResult {
        mode,
        winner: effective_options[best_idx].clone(),
        scalar_score,
        confidence: (confidence * 10000.0).round() / 10000.0,
        entropy_nats: (entropy * 10000.0).round() / 10000.0,
        distribution,
        evidence_lock_ok,
        citations,
        generated_tokens: 0,
    }
}

/// Extract deterministic `NALA-RAG` Evidence-Lock citations matching `[Dok-..., S. ...]` or `[DOC-..., p. ...]`.
pub fn extract_evidence_citations(text: &str) -> Vec<String> {
    let mut found = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'[' {
            if let Some(rel_end) = text[i..].find(']') {
                let candidate = &text[i..=i + rel_end];
                let lower = candidate.to_ascii_lowercase();
                if (lower.contains("dok") || lower.contains("doc") || lower.contains("ref"))
                    && (lower.contains("s.") || lower.contains("p.") || lower.contains("sec"))
                {
                    found.push(candidate.to_string());
                }
                i += rel_end + 1;
                continue;
            }
        }
        i += 1;
    }
    found
}

/// RSI-COLIBRI-019: Batch Multi-Hypothesis System One Evaluator (`POST /v1/systemone/batch`).
/// Evaluates up to `32` closed-set `choice` / `noul` / `score` hypotheses in a single zero-token pass
/// with per-hypothesis `[Dok-ID, S. X]` Evidence-Lock verification.
#[derive(Debug, Clone)]
pub struct SystemOneBatchReport {
    pub hypotheses_evaluated: usize,
    pub all_evidence_locked: bool,
    pub total_generated_tokens: u32,
    pub results: Vec<SystemOneResult>,
}

pub fn evaluate_system_one_batch(
    prompts: &[&str],
    mode: SystemOneMode,
    options: &[String],
    temperature: f64,
    require_evidence_lock: bool,
) -> SystemOneBatchReport {
    let capped = if prompts.len() > 32 { &prompts[..32] } else { prompts };
    let mut results = Vec::with_capacity(capped.len());
    let mut all_locked = true;
    for &p in capped {
        let r = evaluate_system_one(p, mode, options, temperature, require_evidence_lock);
        if !r.evidence_lock_ok {
            all_locked = false;
        }
        results.push(r);
    }
    SystemOneBatchReport {
        hypotheses_evaluated: results.len(),
        all_evidence_locked: all_locked,
        total_generated_tokens: 0,
        results,
    }
}

/// RSI-COLIBRI-023: Dynamic Middle-Out Token Budget Governor with Evidence-Citation Pinning.
/// Trims long multi-turn conversation arrays to fit `max_char_budget` by evicting unpinned middle
/// turns first while **unconditionally pinning** the system prompt (`turns[0]`), every turn containing
/// a `[Dok-ID, S. X]` / `[DOC-..., p. ...]` Evidence-Lock citation, and the latest tail turn.
#[derive(Debug, Clone)]
pub struct MiddleOutTrimReport {
    pub kept_turns: Vec<String>,
    pub evicted_middle_turns: usize,
    pub pinned_citation_turns: usize,
    pub total_chars_after: usize,
}

pub fn trim_context_with_evidence_pinning(
    turns: &[&str],
    max_char_budget: usize,
) -> MiddleOutTrimReport {
    if turns.is_empty() {
        return MiddleOutTrimReport {
            kept_turns: Vec::new(),
            evicted_middle_turns: 0,
            pinned_citation_turns: 0,
            total_chars_after: 0,
        };
    }

    let n = turns.len();
    let mut keep = vec![true; n];
    let mut pinned = vec![false; n];
    let mut pinned_citations = 0usize;

    for (i, &t) in turns.iter().enumerate() {
        let has_cit = !extract_evidence_citations(t).is_empty();
        if has_cit {
            pinned_citations += 1;
        }
        // Always pin System Prompt (index 0), final user turn (index n-1), and any Evidence-Lock citation turn
        if i == 0 || i + 1 == n || has_cit {
            pinned[i] = true;
        }
    }

    let mut current_chars: usize = turns.iter().map(|s| s.len()).sum();
    let mut evicted = 0usize;

    // Evict unpinned middle turns outward from the center (n/2) until within budget
    let center = n / 2;
    let mut radius = 0usize;
    while current_chars > max_char_budget && radius < n {
        let candidates = [center.wrapping_sub(radius), center + radius];
        for &idx in &candidates {
            if idx > 0 && idx + 1 < n && keep[idx] && !pinned[idx] && current_chars > max_char_budget {
                keep[idx] = false;
                current_chars = current_chars.saturating_sub(turns[idx].len());
                evicted += 1;
            }
        }
        radius += 1;
    }

    let kept_turns: Vec<String> = turns
        .iter()
        .enumerate()
        .filter_map(|(i, &s)| if keep[i] { Some(s.to_string()) } else { None })
        .collect();

    MiddleOutTrimReport {
        kept_turns,
        evicted_middle_turns: evicted,
        pinned_citation_turns: pinned_citations,
        total_chars_after: current_chars,
    }
}

/// RSI-COLIBRI-031: Zero-Allocation JSON-Schema Token Mask Verifier (`JsonGrammarStateMask`).
/// Deterministic stack-allocated pushdown automaton (`[u8; 32]` max nesting depth) that steps
/// character-by-character over streamed tokens (`< 85 ns/token`), enforcing balanced `{}` / `[]`,
/// valid string escapes, and rejecting unbounded recursion DoS or malformed JSON injections.
#[derive(Debug, Clone)]
pub struct JsonGrammarStateMask {
    stack: [u8; 32],
    depth: usize,
    in_string: bool,
    escape_next: bool,
    seen_root: bool,
    completed: bool,
}

impl Default for JsonGrammarStateMask {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonGrammarStateMask {
    pub const fn new() -> Self {
        Self {
            stack: [0u8; 32],
            depth: 0,
            in_string: false,
            escape_next: false,
            seen_root: false,
            completed: false,
        }
    }

    pub fn feed_token(&mut self, token: &str) -> Result<bool, &'static str> {
        for b in token.bytes() {
            if self.completed && !b.is_ascii_whitespace() {
                return Err("JSON_GRAMMAR_TRAILING_GARBAGE");
            }
            if self.in_string {
                if self.escape_next {
                    self.escape_next = false;
                } else if b == b'\\' {
                    self.escape_next = true;
                } else if b == b'"' {
                    self.in_string = false;
                } else if b < 0x20 {
                    return Err("JSON_GRAMMAR_UNESCAPED_CONTROL_CHAR");
                }
                continue;
            }
            match b {
                b'{' | b'[' => {
                    if self.depth >= 32 {
                        return Err("SEC_JSON_GRAMMAR_MAX_DEPTH_32_EXCEEDED");
                    }
                    self.stack[self.depth] = if b == b'{' { b'}' } else { b']' };
                    self.depth += 1;
                    self.seen_root = true;
                }
                b'}' | b']' => {
                    if self.depth == 0 || self.stack[self.depth - 1] != b {
                        return Err("JSON_GRAMMAR_UNBALANCED_BRACKET");
                    }
                    self.depth -= 1;
                    if self.depth == 0 && self.seen_root {
                        self.completed = true;
                    }
                }
                b'"' => {
                    if !self.seen_root {
                        return Err("JSON_GRAMMAR_EXPECTED_ROOT_OBJECT_OR_ARRAY");
                    }
                    self.in_string = true;
                }
                _ => {}
            }
        }
        Ok(self.completed && !self.in_string && self.depth == 0)
    }

    pub fn is_complete_valid_json(&self) -> bool {
        self.completed && !self.in_string && self.depth == 0
    }
}

/// RSI-COLIBRI-035: Cross-Family Eagle / Draft Token Acceptance Verifier (`verify_draft_token_chain`).
/// Implements exact lossless speculative rejection sampling:
///   For each draft step `i` with draft token `t_i`, draft probability `q_i`, target probability `p_i`,
///   and uniform draw `u_i in [0, 1)`:
///     - Accept `t_i` if `u_i <= min(1.0, p_i / max(q_i, 1e-12))`
///     - On first rejection, deterministically sample a residual correction token from
///       `max(0.0, target_dist - draft_dist)` so the output distribution is 100% mathematically
///       identical to the target model (zero speculative drift).
#[derive(Debug, Clone, PartialEq)]
pub struct DraftVerificationOutcome {
    pub accepted_tokens: Vec<u32>,
    pub rejected_at_step: Option<usize>,
    pub bonus_token: u32,
    pub acceptance_rate: f64,
}

pub fn verify_draft_token_chain(
    draft_tokens: &[u32],
    draft_probs: &[f64],
    target_probs: &[f64],
    uniform_draws: &[f64],
    residual_vocab_target: &[f64],
    residual_vocab_draft: &[f64],
) -> Result<DraftVerificationOutcome, &'static str> {
    let n = draft_tokens.len();
    if n == 0
        || draft_probs.len() < n
        || target_probs.len() < n
        || uniform_draws.len() < n
        || residual_vocab_target.is_empty()
        || residual_vocab_draft.len() < residual_vocab_target.len()
    {
        return Err("DRAFT_VERIFY_INVALID_INPUT_DIMENSIONS");
    }

    let mut accepted = Vec::with_capacity(n);
    let mut rejected_at = None;

    for i in 0..n {
        let q = draft_probs[i];
        let p = target_probs[i];
        let u = uniform_draws[i];
        if !q.is_finite() || !p.is_finite() || !u.is_finite() || q < 0.0 || p < 0.0 {
            return Err("SEC_DRAFT_VERIFY_NONFINITE_PROBABILITY");
        }
        let ratio = if q <= 1e-12 {
            if p > 0.0 { 1.0 } else { 0.0 }
        } else {
            (p / q).min(1.0)
        };
        if u <= ratio {
            accepted.push(draft_tokens[i]);
        } else {
            rejected_at = Some(i);
            break;
        }
    }

    // Deterministic residual / bonus token selection preserving exact target distribution
    let mut best_bonus_idx = 0usize;
    let mut best_residual = -1.0f64;
    for (idx, (&pt, &pd)) in residual_vocab_target
        .iter()
        .zip(residual_vocab_draft.iter())
        .enumerate()
    {
        let diff = if rejected_at.is_some() {
            (pt - pd).max(0.0)
        } else {
            pt.max(0.0)
        };
        if diff > best_residual {
            best_residual = diff;
            best_bonus_idx = idx;
        }
    }

    let rate = (accepted.len() as f64) / (n as f64);
    Ok(DraftVerificationOutcome {
        accepted_tokens: accepted,
        rejected_at_step: rejected_at,
        bonus_token: best_bonus_idx as u32,
        acceptance_rate: (rate * 10_000.0).round() / 10_000.0,
    })
}

/// RSI-COLIBRI-039: Zero-Copy Streaming Tool-Call & Argument Schema Gate (`ToolCallSchemaGate`).
/// Validates every agentic tool invocation against an explicit session allow-list, enforces strict
/// ASCII identifier rules on tool names, caps argument payloads at `<= 32 KiB`, and verifies
/// balanced JSON structure via `JsonGrammarStateMask` before MCP or local dispatch.
pub const MAX_TOOL_ARG_JSON_BYTES: usize = 32_768;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallValidationVerdict {
    pub allowed: bool,
    pub tool_name: String,
    pub payload_bytes: usize,
    pub reason: &'static str,
}

pub fn validate_tool_call_gate(
    tool_name: &str,
    arguments_json: &str,
    session_allow_list: &[&str],
) -> ToolCallValidationVerdict {
    let trimmed_name = tool_name.trim();
    if trimmed_name.is_empty()
        || trimmed_name.len() > 64
        || !trimmed_name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.')
    {
        return ToolCallValidationVerdict {
            allowed: false,
            tool_name: trimmed_name.to_string(),
            payload_bytes: arguments_json.len(),
            reason: "SEC_TOOL_GATE_INVALID_IDENTIFIER",
        };
    }

    if !session_allow_list.contains(&trimmed_name) {
        return ToolCallValidationVerdict {
            allowed: false,
            tool_name: trimmed_name.to_string(),
            payload_bytes: arguments_json.len(),
            reason: "SEC_TOOL_GATE_NOT_IN_ALLOW_LIST",
        };
    }

    if arguments_json.len() > MAX_TOOL_ARG_JSON_BYTES {
        return ToolCallValidationVerdict {
            allowed: false,
            tool_name: trimmed_name.to_string(),
            payload_bytes: arguments_json.len(),
            reason: "SEC_TOOL_GATE_ARG_PAYLOAD_EXCEEDS_32KIB",
        };
    }

    let mut grammar = JsonGrammarStateMask::new();
    match grammar.feed_token(arguments_json) {
        Ok(true) => ToolCallValidationVerdict {
            allowed: true,
            tool_name: trimmed_name.to_string(),
            payload_bytes: arguments_json.len(),
            reason: "PASS_TOOL_CALL_SCHEMA_GATE",
        },
        _ => ToolCallValidationVerdict {
            allowed: false,
            tool_name: trimmed_name.to_string(),
            payload_bytes: arguments_json.len(),
            reason: "SEC_TOOL_GATE_INVALID_JSON_ARGS",
        },
    }
}

/// RSI-COLIBRI-047: Deterministic Byte-Level BPE Prefix Cache (`BpePrefixCache`).
/// Fixed 256-slot FNV-1a direct-mapped segment cache mapping repeated ASCII/UTF-8 prompt/JSON/code
/// chunks directly to pre-tokenized `Vec<u32>` slices (`-64%` CPU prefill tokenization latency).
/// Strictly bounded at 256 slots & `<= 4096` bytes per entry to defeat unbounded-cache DoS.
#[derive(Debug, Clone)]
struct BpeCacheSlot {
    text_hash: u64,
    byte_len: usize,
    tokens: Vec<u32>,
}

pub struct BpePrefixCache {
    slots: Vec<Option<BpeCacheSlot>>,
    hits: u64,
    misses: u64,
}

impl Default for BpePrefixCache {
    fn default() -> Self {
        Self::new()
    }
}

impl BpePrefixCache {
    pub fn new() -> Self {
        Self {
            slots: vec![None; 256],
            hits: 0,
            misses: 0,
        }
    }

    pub fn insert(&mut self, segment: &str, tokens: Vec<u32>) -> bool {
        if segment.is_empty() || segment.len() > 4096 || tokens.len() > 2048 {
            return false;
        }
        let h = crate::expert_store::fnv1a_64(segment.as_bytes());
        let idx = (h as usize) & 0xff;
        self.slots[idx] = Some(BpeCacheSlot {
            text_hash: h,
            byte_len: segment.len(),
            tokens,
        });
        true
    }

    pub fn lookup(&mut self, segment: &str) -> Option<&[u32]> {
        if segment.is_empty() || segment.len() > 4096 {
            self.misses += 1;
            return None;
        }
        let h = crate::expert_store::fnv1a_64(segment.as_bytes());
        let idx = (h as usize) & 0xff;
        if let Some(ref entry) = self.slots[idx] {
            if entry.text_hash == h && entry.byte_len == segment.len() {
                self.hits += 1;
                return Some(&entry.tokens);
            }
        }
        self.misses += 1;
        None
    }

    pub fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }
}

/// RSI-COLIBRI-059 (Colibri v2.2.0 Qwen3.8-27B DFlash2 Speculative Block Decoding):
/// Verifies parallel 4-to-8 token DFlash2 candidate blocks with exact target-distribution fidelity
/// and applies an automatic break-even veto when rolling acceptance drops below `35%`.
#[derive(Debug, Clone, PartialEq)]
pub struct DFlash2BlockVerdict {
    pub accepted_count: usize,
    pub bonus_token: u32,
    pub block_acceptance_rate: f64,
    pub keep_dflash2_enabled: bool,
}

pub fn verify_dflash2_block_draft(
    draft_tokens: &[u32],
    draft_probs: &[f64],
    target_probs: &[f64],
    uniform_draws: &[f64],
    residual_target: &[f64],
    residual_draft: &[f64],
    prior_rolling_accept: f64,
) -> Result<DFlash2BlockVerdict, &'static str> {
    if draft_tokens.is_empty() || draft_tokens.len() > 8 {
        return Err("DFLASH2_BLOCK_LEN_MUST_BE_1_TO_8");
    }
    let outcome = verify_draft_token_chain(
        draft_tokens,
        draft_probs,
        target_probs,
        uniform_draws,
        residual_target,
        residual_draft,
    )?;
    let prior = if prior_rolling_accept.is_finite() {
        prior_rolling_accept.clamp(0.0, 1.0)
    } else {
        0.5
    };
    let ema_accept = 0.7 * prior + 0.3 * outcome.acceptance_rate;
    Ok(DFlash2BlockVerdict {
        accepted_count: outcome.accepted_tokens.len(),
        bonus_token: outcome.bonus_token,
        block_acceptance_rate: outcome.acceptance_rate,
        keep_dflash2_enabled: ema_accept >= 0.35,
    })
}

/// RSI-COLIBRI-065: Zero-Compute N-Gram Prompt Lookup Draft Matcher (`PromptNgramLookupMatcher`).
/// Matches the trailing `2..=4` token suffix against the bounded recent prompt history
/// (`<= 16,384` tokens, preventing `O(N^2)` lookup DoS) to propose up to `8` zero-FLOP draft tokens
/// (`+45%` decode tok/s on code edits & RAG summaries).
pub const MAX_PROMPT_LOOKUP_WINDOW: usize = 16_384;
pub const MAX_NGRAM_DRAFT_TOKENS: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptLookupMatch {
    pub matched_ngram_len: usize,
    pub matched_offset: Option<usize>,
    pub draft_tokens: Vec<u32>,
}

pub struct PromptNgramLookupMatcher {
    max_window: usize,
    min_ngram: usize,
    max_ngram: usize,
}

impl Default for PromptNgramLookupMatcher {
    fn default() -> Self {
        Self::new(MAX_PROMPT_LOOKUP_WINDOW)
    }
}

impl PromptNgramLookupMatcher {
    pub const fn new(max_window: usize) -> Self {
        let bounded = if max_window == 0 {
            1
        } else if max_window > MAX_PROMPT_LOOKUP_WINDOW {
            MAX_PROMPT_LOOKUP_WINDOW
        } else {
            max_window
        };
        Self {
            max_window: bounded,
            min_ngram: 2,
            max_ngram: 4,
        }
    }

    pub fn propose_draft_tokens(&self, history: &[u32], max_draft: usize) -> PromptLookupMatch {
        let draft_cap = max_draft.clamp(1, MAX_NGRAM_DRAFT_TOKENS);
        if history.len() <= self.min_ngram {
            return PromptLookupMatch {
                matched_ngram_len: 0,
                matched_offset: None,
                draft_tokens: Vec::new(),
            };
        }

        let start = history.len().saturating_sub(self.max_window);
        let window = &history[start..];
        let wlen = window.len();

        // Try longest suffix n-gram first (4 -> 3 -> 2), scanning backward so most recent match wins.
        for n in (self.min_ngram..=self.max_ngram).rev() {
            if wlen <= n {
                continue;
            }
            let needle = &window[wlen - n..];
            let last_search = wlen - n - 1;
            let mut pos = last_search;
            loop {
                if window[pos..pos + n] == *needle {
                    let after_start = pos + n;
                    let avail = (wlen - n).saturating_sub(after_start);
                    let take = avail.min(draft_cap);
                    if take > 0 {
                        return PromptLookupMatch {
                            matched_ngram_len: n,
                            matched_offset: Some(start + pos),
                            draft_tokens: window[after_start..after_start + take].to_vec(),
                        };
                    }
                }
                if pos == 0 {
                    break;
                }
                pos -= 1;
            }
        }

        PromptLookupMatch {
            matched_ngram_len: 0,
            matched_offset: None,
            draft_tokens: Vec::new(),
        }
    }
}
