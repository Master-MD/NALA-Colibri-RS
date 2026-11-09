# NALA-Colibri-RS (`v2.2.0-nala.6.6.6`) — Documentation en Français (`FR`)

[Retour au README principal](../README.md) · [Deutsch](./README_DE.md) · [Italiano](./README_IT.md) · [Rumantsch](./README_RM.md) · [English](./README_EN.md) · [Résumé des licences (5 langues)](../LICENSE_SUMMARY_5LANG.md)

---

## Présentation

**`NALA-Colibri-RS`** (`nala-colibri-rs` / `coli-rs`) est la réécriture **100 % Rust natif sûr (`Edition 2024`, `#![forbid(unsafe_code)]`, `0` crate externe, sans Python ni Node.js)** de [`JustVugg/colibri`](https://github.com/JustVugg/colibri) (`v2.1.0`, Apache-2.0, Copyright 2026 Vincenzo Fornaro).

Les grands modèles Mixture-of-Experts (`GLM-5.3 744B`, `DeepSeek-V4.1-Flash 552B`, `Kimi K3 2.8T`) sont orchestrés sur trois niveaux de mémoire :
- **Niveau 0 (`VRAM Hot`)** : Experts chauds + tronc partagé + cache KV latent MLA compressé `57x`.
- **Niveau 1 (`Host RAM Warm`)** : Experts tièdes en mémoire vive avec compteurs FNV-1a `.coli_usage`.
- **Niveau 2 (`Striped NVMe Cold`)** : Experts froids diffusés depuis un ou deux SSD NVMe avec prélecture de Markov `.coli_pairs`.

## Commandes Principales

```bash
nala-colibri-rs doctor --lang fr
nala-colibri-rs plan deepseek-v4.1-flash
nala-colibri-rs decide --prompt "Validation conforme [Dok-COLIBRI-01, S. 4]" --mode noul --evidence-lock
nala-colibri-rs serve --bind 127.0.0.1:8000
```

## Durcissement de Sécurité (`SEC-COLIBRI-01..04`)
- Limite stricte de `10 Mio` (`HTTP 413`) et protection CSRF `Origin` (`SEC-COLIBRI-01`).
- Blocage SSRF avec épinglage d'adresse IP résolue une seule fois contre le DNS-rebinding (`SEC-COLIBRI-02`).
- Authentification de cluster `HMAC-SHA256` en temps constant (`SEC-COLIBRI-03`).
- Plafond d'en-tête Safetensors à `64 Mio` avec arithmétique vérifiée `checked_mul` et protection symlink (`SEC-COLIBRI-04`).

## Licence
Distribué sous **Apache License 2.0** ([LICENSE](../LICENSE), [NOTICE](../NOTICE)).
