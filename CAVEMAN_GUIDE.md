# CAVEMAN GUIDE — `NALA-Colibri-RS` in 60 Seconds

Big AI brain (`GLM-5.3 744B`, `DeepSeek-V4.1 552B`, `Kimi K3 2.8T`) normally needs giant data-center GPUs (`400 GB+` VRAM).  
Why? Because whole model is huge.

**Secret of MoE (Mixture-of-Experts):**  
Model has `256` tiny specialist brains ("experts") per layer, **but each word only wakes up `8` of them!**

`NALA-Colibri-RS` (`coli-rs`) puts:
1. **Hot experts** -> **GPU VRAM** (super fast)
2. **Warm experts** -> **System RAM** (very fast)
3. **Cold experts** -> **NVMe SSD** (streamed on the fly before next layer even starts!)

And unlike old `colibri` (`32,000+` lines of Python + Node.js + C pointers):
- **`NALA-Colibri-RS` is 1 single ~580 KB Pure Safe Rust binary (`#![forbid(unsafe_code)]`, `0` external crates, `0` Python, `0` Node.js).**
- Starts in **`< 1 ms`** instead of `~60 ms` Python startup.
- Never crashes from C manual `release()` memory bugs (Rust RAII `ExpertLease` drops automatically).

---

## 3 Commands You Need

```bash
# 1. Check your GPU + RAM + NVMe hardware automatically:
nala-colibri-rs doctor

# 2. Plan how a 744B or 552B MoE model fits on your machine:
nala-colibri-rs plan glm-5.3
nala-colibri-rs plan deepseek-v4.1-flash

# 3. Start the hardened OpenAI + Anthropic + SystemOne + 5-Language Web Cockpit:
nala-colibri-rs serve --bind 127.0.0.1:8000
```

Then open **`http://127.0.0.1:8000`** in any browser!
