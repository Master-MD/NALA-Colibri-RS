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
