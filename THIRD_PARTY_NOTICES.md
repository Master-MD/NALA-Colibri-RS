# Third-Party Notices

`NALA-Colibri-RS` (`https://github.com/Master-MD/NALA-Colibri-RS`) is a 100% Pure Rust (`Edition 2024`, zero external crates) derivative fork of **[`JustVugg/colibri`](https://github.com/JustVugg/colibri)** (`v2.1.0`, Apache License 2.0, Copyright 2026 Vincenzo Fornaro).

Below are the preserved upstream third-party attributions from `JustVugg/colibri` alongside clean-room schema compatibility notices:

---

## 1. Upstream `JustVugg/colibri` (`Apache-2.0`)

- **Project**: `JustVugg/colibri` (`https://github.com/JustVugg/colibri`)
- **Copyright**: `Copyright 2026 Vincenzo Fornaro`
- **License**: `Apache-2.0` (see [LICENSE](./LICENSE) and [NOTICE](./NOTICE))

---

## 2. `antirez/ds4` and `ggml` (Upstream Reference Notice — `MIT`)

In upstream `JustVugg/colibri`, portions of `backend_cuda_dsv4.cu` (DeepSeek-V4 GPU router selection algorithm) were adapted from `ds4_cuda.cu` in `https://github.com/antirez/ds4`:

```text
MIT License

Copyright (c) 2026 The ds4.c authors
Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 3. `leonickson1/Swiftlet` `qpack v1` Container Schema (`Apache-2.0`)

`src/containers.rs` in `NALA-Colibri-RS` implements clean-room header inspection for the `Swiftlet qpack v1` container format (`https://github.com/leonickson1/Swiftlet`, Apache License 2.0) to maintain 1:1 interoperability with Colibri `qpack` model installations. No Swift or upstream C source code is copied.

---

## 4. Model Weight Licenses (Important Notice)

`NALA-Colibri-RS` distributes **code only** (`Apache-2.0`) and **zero model weights**. When downloading weights via `coli-rs setup` or Hugging Face, each model's own license governs weight usage (for example, `Qwen-Image-2.1` carries a non-commercial upstream weight license, whereas `GLM-5.3`, `DeepSeek-V4.1-Flash`, `Qwen3.6-35B-A3B`, and `OLMoE-1B-7B` use permissive open-weight licenses). Run `nala-colibri-rs info <model>` to inspect the `license_spdx` field before commercial deployment.
