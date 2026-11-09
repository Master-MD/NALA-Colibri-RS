# NALA-Colibri-RS (`v2.2.0-nala.6.6.6`) — Documentaziun en Rumantsch Grischun (`RM`)

[Enavos al README principal](../README.md) · [Deutsch](./README_DE.md) · [Français](./README_FR.md) · [Italiano](./README_IT.md) · [English](./README_EN.md) · [Survista da las licenzas (5 linguas)](../LICENSE_SUMMARY_5LANG.md)

---

## Survista

**`NALA-Colibri-RS`** (`nala-colibri-rs` / `coli-rs`) è ina reimplementaziun **100 % Rust nativ segir (`Edition 2024`, `#![forbid(unsafe_code)]`, `0` crates externas, senza Python ni Node.js)** da [`JustVugg/colibri`](https://github.com/JustVugg/colibri) (`v2.1.0`, Apache-2.0, Copyright 2026 Vincenzo Fornaro).

Models gronds da Mixture-of-Experts (`GLM-5.3 744B`, `DeepSeek-V4.1-Flash 552B`, `Kimi K3 2.8T`) vegnan repartids sin trais stgalims da memoria:
- **Stgalim 0 (`VRAM Hot`)**: Experts chauds e cache KV MLA cumprimida `57x` sin la GPU.
- **Stgalim 1 (`Host RAM Warm`)**: Experts tievis en la memoria da lavur cun dumbraders FNV-1a `.coli_usage`.
- **Stgalim 2 (`Dual-NVMe Striped Cold`)**: Experts fraids chargiads directamain da SSD NVMe cun pre-lectura da Markov `.coli_pairs`.

## Cumonds Principauls

```bash
nala-colibri-rs doctor --lang rm
nala-colibri-rs plan glm-5.3
nala-colibri-rs decide --prompt "Verifitgà [Dok-COLIBRI-01, S. 4]" --mode noul --evidence-lock
nala-colibri-rs serve --bind 127.0.0.1:8000
```

## Segirezza Defensiva (`SEC-COLIBRI-01..04`)
- Limita da `10 MiB` (`HTTP 413`) e protecziun CSRF `Origin` (`SEC-COLIBRI-01`).
- Protecziun SSRF cun fixaziun directa da l'adressa IP publica encunter DNS-rebinding (`SEC-COLIBRI-02`).
- Autentificaziun dal cluster `HMAC-SHA256` en temp constant (`SEC-COLIBRI-03`).
- Limita da header Safetensors a `64 MiB` cun controlla d'overflow `checked_mul` (`SEC-COLIBRI-04`).

## Licenza
Publitgà sut la **Licenza Apache 2.0** ([LICENSE](../LICENSE), [NOTICE](../NOTICE)).
