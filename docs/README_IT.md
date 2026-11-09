# NALA-Colibri-RS (`v2.2.0-nala.6.6.6`) — Documentazione in Italiano (`IT`)

[Torna al README principale](../README.md) · [Deutsch](./README_DE.md) · [Français](./README_FR.md) · [Rumantsch](./README_RM.md) · [English](./README_EN.md) · [Riepilogo Licenze (5 Lingue)](../LICENSE_SUMMARY_5LANG.md)

---

## Panoramica

**`NALA-Colibri-RS`** (`nala-colibri-rs` / `coli-rs`) è la reimplementazione **100% Rust nativo sicuro (`Edition 2024`, `#![forbid(unsafe_code)]`, zero crate esterni, senza Python né Node.js)** di [`JustVugg/colibri`](https://github.com/JustVugg/colibri) (`v2.1.0`, Apache-2.0, Copyright 2026 Vincenzo Fornaro).

Distribuisce i modelli Mixture-of-Experts (`GLM-5.3 744B`, `DeepSeek-V4.1-Flash 552B`, `Mistral-Large-4 1.02T`, `Kimi K3 2.8T`) su tre livelli gerarchici:
- **Livello 0 (`VRAM Hot`)**: Esperti caldi e cache KV MLA compressa `57x` sulla GPU.
- **Livello 1 (`Host RAM Warm`)**: Esperti tiepidi in RAM con statistiche `.coli_usage` FNV-1a.
- **Livello 2 (`Dual-NVMe Striped Cold`)**: Streaming da SSD NVMe con pre-lettura predittiva di Markov `.coli_pairs`.

## Comandi Principali

```bash
nala-colibri-rs doctor --lang it
nala-colibri-rs plan glm-5.3
nala-colibri-rs decide --prompt "Verifica completata [Dok-COLIBRI-01, S. 4]" --mode noul --evidence-lock
nala-colibri-rs serve --bind 127.0.0.1:8000
```

## Sicurezza Difensiva (`SEC-COLIBRI-01..04`)
- Tet massima richiesta `10 MiB` (`HTTP 413`) e filtro CSRF `Origin` (`SEC-COLIBRI-01`).
- Protezione SSRF con pinning diretto dell'IP pubblico contro DNS-rebinding (`SEC-COLIBRI-02`).
- Autenticazione cluster `HMAC-SHA256` a tempo costante (`SEC-COLIBRI-03`).
- Limite header Safetensors a `64 MiB` con controllo di overflow `checked_mul` (`SEC-COLIBRI-04`).

## Licenza
Rilasciato sotto **Apache License 2.0** ([LICENSE](../LICENSE), [NOTICE](../NOTICE)).
