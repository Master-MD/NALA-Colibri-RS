# NALA-Colibri-RS (`v2.2.0-nala.6.6.6`) — Dokumentation auf Deutsch (`DE`)

[Zurück zum Haupt-README](../README.md) · [Français](./README_FR.md) · [Italiano](./README_IT.md) · [Rumantsch](./README_RM.md) · [English](./README_EN.md) · [5-Sprachen Lizenzübersicht](../LICENSE_SUMMARY_5LANG.md)

---

## Überblick

**`NALA-Colibri-RS`** (`nala-colibri-rs` / `coli-rs`) ist die **100% native Safe-Rust-Turbo-Neuimplementierung (`Edition 2024`, `#![forbid(unsafe_code)]`, `0` externe Crates, Zero-Python, Zero-Node.js)** von [`JustVugg/colibri`](https://github.com/JustVugg/colibri) (`v2.1.0`, Apache-2.0, Copyright 2026 Vincenzo Fornaro).

Grosse Mixture-of-Experts-Modelle (`GLM-5.3 744B`, `DeepSeek-V4.1-Flash 552B`, `Mistral-Large-4 1.02T`, `Kimi K3 2.8T`) aktivieren pro Token nur einen kleinen Bruchteil ihrer Experten. `NALA-Colibri-RS` verteilt diese deterministisch über drei Speicherschichten:
- **Tier 0 (`VRAM Hot`)**: Heisse Experten + Shared Trunk + komprimierter MLA-KV-Cache direkt auf der GPU.
- **Tier 1 (`Host RAM Warm`)**: Warme Experten im gepinnten Arbeitsspeicher mit `.coli_usage` FNV-1a-Hit-Zählern.
- **Tier 2 (`Dual-NVMe Striped Cold`)**: Kalte Experten über gewichtete NVMe-Stripes inkl. `.coli_pairs` 1-Layer-Markov-Lookahead.

## Wichtigste Befehle

```bash
# Kryptographischer Selbsttest & SHA-256 Prüfung (100% Standalone ohne externe Tools)
nala-colibri-rs --self-check
nala-colibri-rs verify-sums dist/SHA256SUMS.txt

# Hardware- & Sicherheits-Diagnose (7 universelle HAL-Profile + SEC-COLIBRI-01..04)
nala-colibri-rs doctor --lang de

# 3-Tier MoE Ressourcenplan inkl. 88% MTP Auto-Gate & Quiet-Envelope berechnen
nala-colibri-rs plan glm-5.3 --eco-thermal

# Jev-kompatible System-One-Entscheidung (0 generierte Tokens + Evidence-Lock)
nala-colibri-rs decide --prompt "Freigabe verifiziert [Dok-COLIBRI-01, S. 4]" --options "freigeben,pruefen,ablehnen" --evidence-lock

# Gehärtetes OpenAI / Anthropic / SystemOne / MCP & 5-Sprachen Web-Cockpit starten
nala-colibri-rs serve --bind 127.0.0.1:8000
```

## Sicherheits-Härtung (`sHeriFF SEC-COLIBRI-01..04`)
1. **`SEC-COLIBRI-01`**: Striktes `10 MiB` Request-Body-Limit (`HTTP 413`) + Loopback-`Origin`-CSRF-Schutz.
2. **`SEC-COLIBRI-02`**: IP-gepinnter SSRF-Schutz ohne doppelte DNS-Auflösung (verhindert `TTL=0` DNS-Rebinding sowie Zugriff auf `127.0.0.0/8`, RFC1918, `169.254.169.254` und `100.64.0.0/10`).
3. **`SEC-COLIBRI-03`**: Zeitkonstante `HMAC-SHA256`-Authentifizierung für `/v1/cluster/register` und `/v1/cluster/heartbeat`.
4. **`SEC-COLIBRI-04`**: `64 MiB` Safetensors-Header-Obergrenze mit `checked_mul` Überlaufprüfung und kanonischem Symlink-/Pfadschutz.

## Lizenz
**Apache License 2.0** ([LICENSE](../LICENSE), [NOTICE](../NOTICE), [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)).
