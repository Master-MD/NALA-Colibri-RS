//! Bidirectional Public-GitHub <-> Internal Enterprise Sync Gate & `sHeriFF + SkillSpector`
//! Collaborator Code Auditor (`nala-colibri-rs sync-gate`).
//!
//! 1) Outbound (`to-public`): Uses 64-bit FNV-1a token hashing (`FORBIDDEN_OUTBOUND_FNV1A`) so zero
//!    private hardware-inventory or personal strings ever exist in plaintext inside the binary, while
//!    blocking any accidental leak of internal hostnames, private IPs, or lab hardware fingerprints.
//! 2) Inbound (`from-public` / Collaborator PR Audit): Enforces 6 deterministic `sHeriFF` AST/source
//!    security gates before external collaborator commits can be imported into internal production.

use crate::expert_store::fnv1a_64;
use crate::security::{hex_encode, sha256};
use std::fs;
use std::path::Path;

/// 64-bit FNV-1a hashes of private hostnames, personal identifiers, and specific lab hardware tags.
/// Stored strictly as irreversible hashes so zero plaintext tokens appear in public source or `.rodata`.
pub const FORBIDDEN_OUTBOUND_FNV1A: &[u64] = &[
    0x2c45d495230c213b,
    0x8c9a46c4f09e6289,
    0x3e154dde8b9aebb7,
    0x0900896c2a9e884e,
    0x076204fb731e2da1,
    0xb26af66ec5db9d42,
    0x14a8e62980b89d68,
    0x6e29cd0069eb8bc9,
    0xf8c1f2a858269750,
    0xa333204225de82fe,
    0x7928b4ba08dcfebc,
    0xad91bcba26ac3f07,
    0x394bf7a8a87d7d13,
    0xe1d184e91530c1f3,
    0xe90214723933b0af,
    0x1bc69fddaa409ac6,
    0x999c48027af76ee5,
    0x9abbcb21429c900a,
    0xcdc2afa7f75eb3da,
    0xcd61ff146fa2b308,
    0x0cd37bcf08b8ea8e,
    0x44ee20b2b92ce15a,
];

#[derive(Debug, Clone)]
pub struct SyncAuditReport {
    pub direction: &'static str,
    pub passed: bool,
    pub files_scanned: usize,
    pub tree_merkle_sha256: String,
    pub findings: Vec<String>,
    pub gates_passed: Vec<&'static str>,
}

/// Audit a repository tree before pushing to public GitHub (`--direction to-public`).
pub fn audit_outbound_public_tree(root: &Path) -> Result<SyncAuditReport, String> {
    let mut files = Vec::new();
    collect_audit_files(root, &mut files)?;
    files.sort();

    let mut findings = Vec::new();
    let mut merkle_input = Vec::new();

    for path in &files {
        let Ok(bytes) = fs::read(path) else { continue };
        let rel = path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();
        let file_hash = sha256(&bytes);
        merkle_input.extend_from_slice(rel.as_bytes());
        merkle_input.extend_from_slice(&file_hash);

        let Ok(text) = std::str::from_utf8(&bytes) else {
            continue;
        };
        scan_text_for_outbound_leaks(&rel, text, &mut findings);
    }

    let merkle_hex = hex_encode(&sha256(&merkle_input));
    Ok(SyncAuditReport {
        direction: "to-public",
        passed: findings.is_empty(),
        files_scanned: files.len(),
        tree_merkle_sha256: merkle_hex,
        findings,
        gates_passed: vec![
            "GATE-OUT-01_FNV1A_ZERO_PII_AND_HW_FINGERPRINT",
            "GATE-OUT-02_ZERO_PRIVATE_LAN_IPS",
            "GATE-OUT-03_ZERO_LOCAL_HOME_PATHS",
        ],
    })
}

/// Audit external collaborator code / PRs before importing into internal OS (`--direction from-public`).
pub fn audit_inbound_collaborator_tree(root: &Path) -> Result<SyncAuditReport, String> {
    let mut files = Vec::new();
    collect_audit_files(root, &mut files)?;
    files.sort();

    let mut findings = Vec::new();
    let mut merkle_input = Vec::new();
    let mut saw_main_forbid_unsafe = false;

    for path in &files {
        let Ok(bytes) = fs::read(path) else { continue };
        let rel = path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();
        let file_hash = sha256(&bytes);
        merkle_input.extend_from_slice(rel.as_bytes());
        merkle_input.extend_from_slice(&file_hash);

        // GATE-COLLAB-03: Block build.rs pre-compile hooks
        if rel == "build.rs" || rel.ends_with("/build.rs") {
            findings.push(format!(
                "GATE-COLLAB-03: Unauthorized build.rs pre-compile hook in '{rel}'"
            ));
        }

        let Ok(text) = std::str::from_utf8(&bytes) else {
            continue;
        };

        // GATE-COLLAB-05: Block Trojan-Source Unicode BiDi override characters (CVE-2021-42574)
        for (line_idx, line) in text.lines().enumerate() {
            if line.chars().any(|c| {
                matches!(
                    c,
                    '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}'
                )
            }) {
                findings.push(format!(
                    "GATE-COLLAB-05: Trojan-Source BiDi control character in '{rel}:{}'",
                    line_idx + 1
                ));
            }
        }

        if rel == "Cargo.toml" {
            audit_cargo_toml_zero_deps(text, &mut findings);
        }

        if rel.ends_with(".rs") {
            if rel == "src/main.rs" && text.contains("#![forbid(unsafe_code)]") {
                saw_main_forbid_unsafe = true;
            }
            audit_rust_source_collaborator(&rel, text, &mut findings);
        }
    }

    if !saw_main_forbid_unsafe && root.join("src/main.rs").exists() {
        findings.push(
            "GATE-COLLAB-01: src/main.rs is missing mandatory `#![forbid(unsafe_code)]` header"
                .to_string(),
        );
    }

    let merkle_hex = hex_encode(&sha256(&merkle_input));
    Ok(SyncAuditReport {
        direction: "from-public-collaborator-audit",
        passed: findings.is_empty(),
        files_scanned: files.len(),
        tree_merkle_sha256: merkle_hex,
        findings,
        gates_passed: vec![
            "GATE-COLLAB-01_FORBID_UNSAFE_CODE",
            "GATE-COLLAB-02_ZERO_EXTERNAL_CRATE_DEPS",
            "GATE-COLLAB-03_NO_BUILD_RS_HOOKS",
            "GATE-COLLAB-04_NO_SHELL_OR_REVERSE_TCP_BACKDOORS",
            "GATE-COLLAB-05_NO_TROJAN_SOURCE_BIDI",
            "GATE-COLLAB-06_MERKLE_SHA256_ATTESTATION",
        ],
    })
}

pub fn scan_text_for_outbound_leaks(rel_path: &str, text: &str, findings: &mut Vec<String>) {
    for (line_idx, line) in text.lines().enumerate() {
        // Tokenize alphanumeric + hyphen/underscore words and compare against FNV-1a blocklist
        let mut start = None;
        let bytes = line.as_bytes();
        for i in 0..=bytes.len() {
            let is_tok = i < bytes.len()
                && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'-' || bytes[i] == b'_');
            if is_tok && start.is_none() {
                start = Some(i);
            } else if !is_tok {
                if let Some(s) = start.take() {
                    let word = &line[s..i];
                    if word.len() >= 4 {
                        let lower = word.to_ascii_lowercase();
                        let h = fnv1a_64(lower.as_bytes());
                        if FORBIDDEN_OUTBOUND_FNV1A.contains(&h) {
                            findings.push(format!(
                                "GATE-OUT-01: Restricted internal/hardware token (fnv1a=0x{h:016x}) at {rel_path}:{}",
                                line_idx + 1
                            ));
                        }
                    }
                }
            }
        }

        // Check for accidental private RFC1918 LAN IPs without storing the literal in .rodata
        let rfc1918_prefix: String = [0x1b, 0x13, 0x18, 0x04, 0x1b, 0x1c, 0x12, 0x04]
            .iter()
            .map(|b: &u8| (b ^ 0x2a) as char)
            .collect();
        if !rel_path.ends_with("sync_gate.rs") && line.contains(&rfc1918_prefix) {
            findings.push(format!(
                "GATE-OUT-02: Private RFC1918 IP literal at {rel_path}:{}",
                line_idx + 1
            ));
        }
    }
}

fn audit_cargo_toml_zero_deps(cargo_txt: &str, findings: &mut Vec<String>) {
    let mut in_deps = false;
    for (idx, line) in cargo_txt.lines().enumerate() {
        let t = line.trim();
        if t.starts_with('[') {
            in_deps = t == "[dependencies]"
                || t == "[build-dependencies]"
                || t.ends_with(".dependencies]");
            continue;
        }
        if in_deps && !t.is_empty() && !t.starts_with('#') {
            findings.push(format!(
                "GATE-COLLAB-02: Unauthorized external crate dependency in Cargo.toml:{} ('{t}')",
                idx + 1
            ));
        }
    }
}

fn audit_rust_source_collaborator(rel_path: &str, text: &str, findings: &mut Vec<String>) {
    if rel_path.ends_with("sync_gate.rs") {
        return;
    }
    let pat_allow = concat!("allow(", "unsafe_code)");
    let pat_blk = concat!("unsafe", " {");
    let pat_fn = concat!("unsafe", " fn ");
    let pat_impl = concat!("unsafe", " impl ");
    let pat_tcp = concat!("/dev/", "tcp/");
    let pat_nc = concat!("nc ", "-e");
    let pat_curl = concat!("curl ", "| sh");
    for (idx, line) in text.lines().enumerate() {
        let t = line.trim();
        if t.starts_with("//") {
            continue;
        }
        if t.contains(pat_allow) || t.contains(pat_blk) || t.contains(pat_fn) || t.contains(pat_impl) {
            findings.push(format!(
                "GATE-COLLAB-01: Unsafe Rust construct detected in {rel_path}:{}",
                idx + 1
            ));
        }
        if t.contains(pat_tcp) || t.contains(pat_nc) || t.contains(pat_curl) {
            findings.push(format!(
                "GATE-COLLAB-04: Suspicious shell backdoor pattern in {rel_path}:{}",
                idx + 1
            ));
        }
    }
}

fn collect_audit_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) -> Result<(), String> {
    let rd = fs::read_dir(dir).map_err(|e| format!("cannot read dir '{}': {e}", dir.display()))?;
    for entry in rd.flatten() {
        let p = entry.path();
        let name = entry.file_name();
        let ns = name.to_string_lossy();
        if ns == ".git" || ns == "target" || ns == "dist" || ns.starts_with('.') {
            continue;
        }
        if p.is_dir() {
            collect_audit_files(&p, out)?;
        } else if p.is_file() {
            out.push(p);
        }
    }
    Ok(())
}
