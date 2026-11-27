//! 14-Subsystem Deterministic 10,000-Iteration Self-Benchmark Suite (`--bench` / `--bench --json`).
//! Uses `std::hint::black_box` and `std::time::Instant` to measure real `ns/op` and `us/op` on host silicon.

use crate::catalog::{CATALOG_MODELS, find_model};
use crate::containers::{
    decompress_wnr4_mx_block, parse_gguf_header, parse_qpack_header, parse_safetensors_header,
    rans_compress_bf16_exponents, rans_decompress_bf16_exponents, validate_lora_hotswap,
};
use crate::expert_store::{
    BatchRequest, ChunkedPrefillArbiter, CoActivationMatrix64, ColiParkVault, DynamicBatchPacker,
    ExpertStore, GenerationScheduler, KvPageCompactor, KvPrefixCache, MultiplexedChain64,
    RadixKvPrefixTree, RegisteredNvmePageRing, SlidingWindowKvRing,
};
use crate::kernels::{
    bf16_to_f32_bulk, cc_norm_k_simd8, deltanet_step_simd8, dequantize_kv_q8_0_block32,
    f16_to_f32_bulk, flash_attn_online_softmax_simd8, gqa_head_group_scores_simd8,
    matmul_mla_latent_57x, matmul_q_fp8_e4m3fn, matmul_q_gs64_int4, matmul_q_gs64_int8,
    matmul_q_mxfp4_fmt7, matmul_q_ternary_158, matmul_small_k_f32_simd4, nsa_topk_blocks,
    quantize_kv_q8_0_block32, rmsnorm_residual_simd8, rope_yarn_rotate_simd8, swiglu_fused_simd8,
};
use crate::mcp::handle_mcp_jsonrpc;
use crate::planner::{
    HalProfile, HardwareEnvelope, compute_three_tier_plan, plan_vision_patch_tiles,
    recommend_best_model_for_machine,
};
use crate::security::{
    ClusterPeerPinTable, UpstreamCircuitBreaker, hex_encode, hmac_sha256, is_strictly_public_ip,
    validate_json_payload_bounds, verify_hmac_sha256_hex,
};
use crate::server::ReasoningStreamSplitter;
use crate::system_one::{
    BpePrefixCache, PromptNgramLookupMatcher, SystemOneMode, evaluate_system_one,
    trim_tool_output_utf8_safe, verify_dflash2_block_draft,
};
use std::hint::black_box;
use std::net::{IpAddr, Ipv4Addr};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct BenchRow {
    pub id: &'static str,
    pub subsystem: &'static str,
    pub iterations: u64,
    pub ns_per_op: f64,
    pub us_per_op: f64,
}

pub fn run_all_benchmarks(iterations: u64) -> Vec<BenchRow> {
    let iters = iterations.max(100);
    let mut results = Vec::with_capacity(14);

    // 1. 4-row interleaved IEEE-754 order-preserving matmul_q_gs64_int8 (16x256)
    {
        let rows = 16usize;
        let cols = 256usize;
        let x = vec![0.125f32; cols];
        let wq: Vec<i8> = (0..rows * cols).map(|i| ((i % 15) as i8) - 7).collect();
        let scales = vec![0.03125f32; rows * (cols / 64)];
        let mut out = vec![0.0f32; rows];
        let t0 = Instant::now();
        for _ in 0..iters {
            matmul_q_gs64_int8(
                black_box(&mut out),
                black_box(&x),
                black_box(&wq),
                black_box(&scales),
                rows,
                cols,
            );
            black_box(out[0]);
        }
        results.push(record("B01", "matmul_q_gs64_int8 (16x256 4-row IEEE-754)", iters, t0));
    }

    // 2. 4-row interleaved matmul_q_gs64_int4 (16x256 packed nibbles)
    {
        let rows = 16usize;
        let cols = 256usize;
        let x = vec![0.25f32; cols];
        let wp: Vec<u8> = (0..rows * (cols / 2)).map(|i| ((i * 13) & 0xff) as u8).collect();
        let scales = vec![0.0625f32; rows * (cols / 64)];
        let mut out = vec![0.0f32; rows];
        let t0 = Instant::now();
        for _ in 0..iters {
            matmul_q_gs64_int4(
                black_box(&mut out),
                black_box(&x),
                black_box(&wp),
                black_box(&scales),
                rows,
                cols,
            );
            black_box(out[0]);
        }
        results.push(record("B02", "matmul_q_gs64_int4 (16x256 packed int4_gs64)", iters, t0));
    }

    // 3. Bulk BF16 + F16 -> F32 conversion (512 elements)
    {
        let src: Vec<u16> = (0..512u16).map(|v| 0x3c00u16.wrapping_add(v)).collect();
        let mut dst = vec![0.0f32; 512];
        let t0 = Instant::now();
        for _ in 0..iters {
            bf16_to_f32_bulk(black_box(&src), black_box(&mut dst));
            f16_to_f32_bulk(black_box(&src[..128]), black_box(&mut dst[..128]));
            black_box(dst[0]);
        }
        results.push(record("B03", "bf16_to_f32(512) + f16_to_f32(128) bulk", iters, t0));
    }

    // 4. Safetensors Header Parser + SEC-COLIBRI-04 overflow guard
    {
        let hdr = br#"{"__metadata__":{"colibri.fmt":"int4_gs64"},"blk.0.ffn_gate.weight":{"dtype":"U8","shape":[128,64],"data_offsets":[0,8192]}}"#;
        let mut buf = Vec::with_capacity(8 + hdr.len());
        buf.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
        buf.extend_from_slice(hdr);
        let total_len = (8 + hdr.len() + 8192) as u64;
        let t0 = Instant::now();
        for _ in 0..iters {
            let parsed = parse_safetensors_header(black_box(&buf), Some(total_len)).unwrap();
            black_box(parsed.tensors.len());
        }
        results.push(record("B04", "Safetensors header parse + SEC-04 bounds", iters, t0));
    }

    // 5. GGUF v3 split-shard + Swiftlet qpack v1 header inspector
    {
        let mut gguf = [0u8; 24];
        gguf[0..4].copy_from_slice(b"GGUF");
        gguf[4..8].copy_from_slice(&3u32.to_le_bytes());
        gguf[8..16].copy_from_slice(&412u64.to_le_bytes());
        gguf[16..24].copy_from_slice(&36u64.to_le_bytes());
        let mut qp = [0u8; 20];
        qp[0..4].copy_from_slice(b"QPK1");
        qp[4..8].copy_from_slice(&1u32.to_le_bytes());
        qp[8..12].copy_from_slice(&64u32.to_le_bytes());
        qp[12..20].copy_from_slice(&10737418240u64.to_le_bytes());
        let t0 = Instant::now();
        for _ in 0..iters {
            let g = parse_gguf_header(black_box(&gguf), Some("model-00001-of-00005.gguf")).unwrap();
            let q = parse_qpack_header(black_box(&qp)).unwrap();
            black_box((g.tensor_count, q.chunk_count));
        }
        results.push(record("B05", "GGUF v3 split-part + Swiftlet qpack v1 parse", iters, t0));
    }

    // 6. RAII ExpertStore acquire + drop lease + Markov .coli_pairs lookahead
    {
        let mut store = ExpertStore::new("glm-5.3", 16, 64, 256, 256, 7.2, 6.4);
        store.record_markov_pair(3, 12, 41);
        let t0 = Instant::now();
        for _ in 0..iters {
            let lease = store.acquire(black_box(3), black_box(12)).unwrap();
            let pred = store.predict_next_layer_expert(black_box(3), black_box(12));
            black_box((lease.tier(), pred));
        }
        results.push(record("B06", "RAII ExpertStore lease + Markov lookahead", iters, t0));
    }

    // 7. Exact KvPrefixCache matching (512-token prefix)
    {
        let toks: Vec<u32> = (0..512u32).collect();
        let mut cache = KvPrefixCache::new(1024);
        cache.commit_tokens(0, &toks);
        let query: Vec<u32> = (0..520u32).collect();
        let t0 = Instant::now();
        for _ in 0..iters {
            let n = cache.reusable_prefix_len(black_box(&query));
            black_box(n);
        }
        results.push(record("B07", "KvPrefixCache 512-token exact reuse check", iters, t0));
    }

    // 8. COLI_PARK short-request conversation save/restore (32 rows x 64 floats)
    {
        let mut vault = ColiParkVault::new(64);
        let mut kv = vec![1.25f32; 128 * 64];
        let mut toks: Vec<u32> = (0..128u32).collect();
        let t0 = Instant::now();
        for _ in 0..iters {
            vault.park_save(black_box(&kv), black_box(&toks), 32);
            let r = vault.park_restore(black_box(&mut kv), black_box(&mut toks));
            black_box(r);
        }
        results.push(record("B08", "COLI_PARK 32-row conversation save+restore", iters, t0));
    }

    // 9. 16-Slot GenerationScheduler acquire + release cycle
    {
        let mut sched = GenerationScheduler::new(16, 64);
        let t0 = Instant::now();
        for _ in 0..iters {
            let slot = sched.try_acquire_slot().unwrap();
            sched.release_slot(black_box(slot));
        }
        results.push(record("B09", "16-Slot GenerationScheduler acquire+release", iters, t0));
    }

    // 10. 7-Profile NALA-HAL 3-Tier Resource Planner (`compute_three_tier_plan`)
    {
        let m = find_model("glm-5.3").unwrap();
        let hw = HardwareEnvelope {
            profile: HalProfile::NvidiaCudaDiscrete,
            os_arch: "linux-x86_64".to_string(),
            cpu_isa: "x86_64-sse4.2-avx2-fma".to_string(),
            cpu_threads: 32,
            vram_gb: 48.0,
            ram_gb: 128.0,
            nvme_stripes: 2,
            aggregate_nvme_gbps: 13.6,
            gpu_mem_bw_gbps: 896.0,
            resizable_bar: true,
            eco_thermal_quiet: false,
            standalone_mode: true,
        };
        let t0 = Instant::now();
        for _ in 0..iters {
            let plan = compute_three_tier_plan(black_box(m), black_box(&hw), 4, 8192);
            black_box(plan.estimated_decode_tok_s);
        }
        results.push(record("B10", "NALA-HAL 3-Tier MoE Planner (GLM-5.3 744B)", iters, t0));
    }

    // 11. Catalog lookup across 13 families & 21 presets
    {
        let t0 = Instant::now();
        for _ in 0..iters {
            let a = find_model(black_box("deepseek-v4.1-flash")).unwrap();
            let b = find_model(black_box("qwen3.8-flash-next")).unwrap();
            black_box((a.total_params_b, b.routed_experts, CATALOG_MODELS.len()));
        }
        results.push(record("B11", "13-Family Catalog lookup & metadata resolve", iters, t0));
    }

    // 12. System One Calibrated Choice + Shannon Entropy + NALA-RAG Evidence-Lock
    {
        let opts = vec![
            "tier0_vram".to_string(),
            "tier1_ram".to_string(),
            "tier2_nvme".to_string(),
        ];
        let prompt = "Select tier0_vram hot placement verified [Dok-COLIBRI-02, S. 9]";
        let t0 = Instant::now();
        for _ in 0..iters {
            let r = evaluate_system_one(
                black_box(prompt),
                SystemOneMode::Choice,
                black_box(&opts),
                1.0,
                true,
            );
            black_box((r.confidence, r.evidence_lock_ok));
        }
        results.push(record("B12", "System One calibrated choice + Evidence-Lock", iters, t0));
    }

    // 13. sHeriFF SEC-COLIBRI-02/03 Constant-Time HMAC-SHA256 + SSRF IP Classifier
    {
        let key = b"nala-cluster-zero-trust-key";
        let msg = b"worker-node-register:8000:glm-5.3";
        let sig = hex_encode(&hmac_sha256(key, msg));
        let t0 = Instant::now();
        for _ in 0..iters {
            let ok_mac = verify_hmac_sha256_hex(black_box(key), black_box(msg), black_box(&sig));
            let block_meta = is_strictly_public_ip(IpAddr::V4(Ipv4Addr::new(169, 254, 169, 254)));
            black_box((ok_mac, block_meta));
        }
        results.push(record("B13", "SEC-02/03 HMAC-SHA256 + SSRF metadata guard", iters, t0));
    }

    // 14. JSON-RPC 2.0 MCP Server request dispatch (`tools/call status`)
    {
        let req = r#"{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"rsi_proposals"}}"#;
        let t0 = Instant::now();
        for _ in 0..iters {
            let resp = handle_mcp_jsonrpc(black_box(req)).unwrap();
            black_box(resp.len());
        }
        results.push(record("B14", "MCP JSON-RPC 2.0 dispatch (rsi_proposals)", iters, t0));
    }

    // 15. RSI-COLIBRI-013/014 Queue-Depth 8 RegisteredNvmePageRing batch + DietGPU rANS BF16 decode
    {
        let ring = RegisteredNvmePageRing::new(true);
        let batch = [4096usize; 8];
        let bf16_src: Vec<u16> = (0..128u16).map(|i| 0x3f80u16 | (i & 0x001f)).collect();
        let compressed = rans_compress_bf16_exponents(&bf16_src);
        let mut decoded = [0u16; 128];
        let t0 = Instant::now();
        for _ in 0..iters {
            let bytes = ring.submit_qd8_batch(black_box(&batch)).unwrap();
            let csum = rans_decompress_bf16_exponents(black_box(&compressed), black_box(&mut decoded)).unwrap();
            black_box((bytes, csum));
        }
        results.push(record("B15", "QD=8 RegisteredNvmePageRing + DietGPU rANS BF16(128)", iters, t0));
    }

    // 16. RSI-COLIBRI-017 4-row interleaved AMD Quark MXFP4 (`E2M1` + `E8M0` shared exponent, fmt 7) GEMV
    {
        let rows = 16usize;
        let cols = 256usize;
        let x = vec![0.25f32; cols];
        let wp: Vec<u8> = (0..rows * (cols / 2)).map(|i| ((i * 29) & 0xff) as u8).collect();
        let e8m0 = vec![127u8; rows * (cols / 64)];
        let mut out = vec![0.0f32; rows];
        let t0 = Instant::now();
        for _ in 0..iters {
            matmul_q_mxfp4_fmt7(
                black_box(&mut out),
                black_box(&x),
                black_box(&wp),
                black_box(&e8m0),
                rows,
                cols,
            );
            black_box(out[0]);
        }
        results.push(record("B16", "matmul_q_mxfp4_fmt7 (16x256 Quark E2M1+E8M0)", iters, t0));
    }

    // 17. RSI-COLIBRI-021 Fused Low-Rank MLA c_KV Up-Projection + Register-Resident RoPE Rotation (16x128)
    {
        let head_dim = 16usize;
        let latent_rank = 128usize;
        let c_kv = vec![0.125f32; latent_rank];
        let w_uk: Vec<i8> = (0..head_dim * latent_rank)
            .map(|i| ((i % 13) as i8) - 6)
            .collect();
        let scales = vec![0.03125f32; head_dim * (latent_rank / 64)];
        let cos = vec![0.8660254f32; head_dim / 2];
        let sin = vec![0.5f32; head_dim / 2];
        let mut out_k = vec![0.0f32; head_dim];
        let t0 = Instant::now();
        for _ in 0..iters {
            matmul_mla_latent_57x(
                black_box(&mut out_k),
                black_box(&c_kv),
                black_box(&w_uk),
                black_box(&scales),
                black_box(&cos),
                black_box(&sin),
                head_dim,
                latent_rank,
            );
            black_box(out_k[0]);
        }
        results.push(record("B17", "matmul_mla_latent_57x (16x128 fused c_KV + RoPE)", iters, t0));
    }

    // 18. RSI-COLIBRI-025 Block-Sparse NSA / DSA Top-K Lightning Indexer SIMD Kernel (64 blocks x 64-dim)
    {
        let q = [0.25f32; 64];
        let num_blocks = 64usize;
        let block_keys: Vec<i8> = (0..num_blocks * 64)
            .map(|i| ((i % 15) as i8) - 7)
            .collect();
        let scales = vec![0.0625f32; num_blocks];
        let t0 = Instant::now();
        for _ in 0..iters {
            let top = nsa_topk_blocks(
                black_box(&q),
                black_box(&block_keys),
                black_box(&scales),
                num_blocks,
                4096,
                8,
            );
            black_box(top[0].block_idx);
        }
        results.push(record("B18", "nsa_topk_blocks (64 blocks x 64-dim NSA Indexer)", iters, t0));
    }

    // 19. RSI-COLIBRI-029 Fused Gated-DeltaNet / Mamba-2 Recurrent State Update SIMD Kernel (16x16 O(1))
    {
        let d = 16usize;
        let mut state = vec![0.01f32; d * d];
        let mut out_y = vec![0.0f32; d];
        let q = [0.25f32; 16];
        let k = [0.20f32; 16];
        let v = [0.50f32; 16];
        let t0 = Instant::now();
        for _ in 0..iters {
            deltanet_step_simd8(
                black_box(&mut state),
                black_box(&mut out_y),
                black_box(&q),
                black_box(&k),
                black_box(&v),
                0.98,
                0.5,
                d,
            )
            .unwrap();
            black_box(out_y[0]);
        }
        results.push(record("B19", "deltanet_step_simd8 (16x16 fused O(1) Householder)", iters, t0));
    }

    // 20. RSI-COLIBRI-033 Continuous Prefill-Decode Chunked Token Scheduler (512-tok chunk + decode interleave)
    {
        let mut arb = ChunkedPrefillArbiter::new(512, 516);
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = arb.submit_prompt(0, 32_768);
            let _ = arb.enqueue_decode(1);
            let _ = arb.enqueue_decode(2);
            let ops = arb.next_batch();
            black_box(ops.len());
        }
        results.push(record("B20", "ChunkedPrefillArbiter (32k prefill + 2 decodes)", iters, t0));
    }

    // 21. RSI-COLIBRI-034 Native Pairwise-SIMD FP8 E4M3FN Software Emulation GEMV (16x256 LUT)
    {
        let rows = 16usize;
        let cols = 256usize;
        let x = vec![0.25f32; cols];
        let w_fp8: Vec<u8> = (0..rows * cols).map(|i| ((i * 19) & 0x7e) as u8).collect();
        let scales = vec![0.0625f32; rows * (cols / 64)];
        let mut out = vec![0.0f32; rows];
        let t0 = Instant::now();
        for _ in 0..iters {
            matmul_q_fp8_e4m3fn(
                black_box(&mut out),
                black_box(&x),
                black_box(&w_fp8),
                black_box(&scales),
                rows,
                cols,
            );
            black_box(out[0]);
        }
        results.push(record("B21", "matmul_q_fp8_e4m3fn (16x256 LUT + SIMD8)", iters, t0));
    }

    // 22. RSI-COLIBRI-037 Zero-Copy Multi-Slot Radix KV-Prefix Tree (256-token 16-block tenant match)
    {
        let mut tree = RadixKvPrefixTree::new(256);
        let prompt_toks: Vec<u32> = (0..256u32).map(|i| 1000 + i).collect();
        tree.insert_prefix("tenant-swarm-alpha", &prompt_toks);
        let t0 = Instant::now();
        for _ in 0..iters {
            let m = tree.match_prefix(black_box("tenant-swarm-alpha"), black_box(&prompt_toks));
            black_box(m.matched_tokens);
        }
        results.push(record("B22", "RadixKvPrefixTree (256-tok 16-block tenant match)", iters, t0));
    }

    // 23. RSI-COLIBRI-038 Native Multiplier-Free 1.58-Bit Ternary SIMD Kernel (16x256 add/sub GEMV)
    {
        let rows = 16usize;
        let cols = 256usize;
        let x = vec![0.25f32; cols];
        let w_2bit: Vec<u8> = (0..rows * (cols / 4)).map(|i| ((i * 37) & 0xff) as u8).collect();
        let scales = vec![0.5f32; rows * (cols / 64)];
        let mut out = vec![0.0f32; rows];
        let t0 = Instant::now();
        for _ in 0..iters {
            matmul_q_ternary_158(
                black_box(&mut out),
                black_box(&x),
                black_box(&w_2bit),
                black_box(&scales),
                rows,
                cols,
            );
            black_box(out[0]);
        }
        results.push(record("B23", "matmul_q_ternary_158 (16x256 1.58-bit ternary GEMV)", iters, t0));
    }

    // 24. RSI-COLIBRI-041 Zero-Copy Symmetric Q8_0 KV-Cache Block Quantizer + Dequantize (32 elems)
    {
        let mut src = [0.0f32; 32];
        for i in 0..32 {
            src[i] = ((i as f32) - 15.5) * 0.125;
        }
        let mut dst = [0.0f32; 32];
        let t0 = Instant::now();
        for _ in 0..iters {
            let blk = quantize_kv_q8_0_block32(black_box(&src));
            dequantize_kv_q8_0_block32(black_box(&blk), black_box(&mut dst));
            black_box(dst[0]);
        }
        results.push(record("B24", "quantize_kv_q8_0_block32 (32-elem KV Q8_0 codec)", iters, t0));
    }

    // 25. RSI-COLIBRI-042 Adaptive Cross-Layer MoE Expert Co-Activation Prefetcher (64x64 matrix)
    {
        let mut mat = CoActivationMatrix64::new();
        let active = [3usize, 11, 27, 42];
        mat.record_co_activation(&active);
        let t0 = Instant::now();
        for _ in 0..iters {
            mat.record_co_activation(black_box(&active));
            let top = mat.predict_partners(black_box(3), 4);
            black_box(top.len());
        }
        results.push(record("B25", "CoActivationMatrix64 (record + top-4 prefetch)", iters, t0));
    }

    // 26. RSI-COLIBRI-045 Fused RMSNorm + Residual Add SIMD Kernel (256-dim single L1 pass)
    {
        let n = 256usize;
        let x = vec![0.25f32; n];
        let mut res = vec![0.125f32; n];
        let w = vec![1.0f32; n];
        let mut out = vec![0.0f32; n];
        let t0 = Instant::now();
        for _ in 0..iters {
            res[0] = 0.125;
            let rms = rmsnorm_residual_simd8(
                black_box(&mut out),
                black_box(&mut res),
                black_box(&x),
                black_box(&w),
                1e-6,
            )
            .unwrap();
            black_box((rms, out[0]));
        }
        results.push(record("B26", "rmsnorm_residual_simd8 (256-dim fused L1 pass)", iters, t0));
    }

    // 27. RSI-COLIBRI-046/047 KvPageCompactor (64-page defrag) + BpePrefixCache lookup
    {
        let mut cache = BpePrefixCache::new();
        cache.insert("{\"role\":\"system\",\"content\":\"", vec![10, 20, 30, 40, 50]);
        let t0 = Instant::now();
        for _ in 0..iters {
            let mut comp = KvPageCompactor::new(64);
            let _ = comp.allocate_page(60, 1);
            let _ = comp.allocate_page(62, 2);
            let moves = comp.compact_step(4);
            let hit = cache.lookup(black_box("{\"role\":\"system\",\"content\":\""));
            black_box((moves.len(), hit.map(|s| s.len()).unwrap_or(0)));
        }
        results.push(record("B27", "KvPageCompactor (64p) + BpePrefixCache lookup", iters, t0));
    }

    // 28. RSI-COLIBRI-049 Fused SwiGLU Gate-Up Activation SIMD Kernel (256-dim L1 pass)
    {
        let n = 256usize;
        let gate = vec![0.75f32; n];
        let up = vec![1.25f32; n];
        let mut out = vec![0.0f32; n];
        let t0 = Instant::now();
        for _ in 0..iters {
            swiglu_fused_simd8(black_box(&mut out), black_box(&gate), black_box(&up)).unwrap();
            black_box(out[0]);
        }
        results.push(record("B28", "swiglu_fused_simd8 (256-dim fused silu*up)", iters, t0));
    }

    // 29. RSI-COLIBRI-054 Colibri v2.2 WEX4 WNR4-MX (codec 8, #3144) 64-weight MXFP4 Decompressor
    {
        let mut blk = [0x35u8; 34];
        blk[0] = 127; // scale = 1.0
        blk[1] = 0;
        let mut out = [0.0f32; 64];
        let t0 = Instant::now();
        for _ in 0..iters {
            let sc = decompress_wnr4_mx_block(black_box(&blk), black_box(&mut out)).unwrap();
            black_box((sc, out[0]));
        }
        results.push(record("B29", "WNR4-MX codec-8 (#3144) 64-weight MXFP4 decode", iters, t0));
    }

    // 30. RSI-COLIBRI-051/055 VisionPatchTilePlanner + Colibri v2.2 Auto-Setup Recommender (28 models)
    {
        let hw = HardwareEnvelope {
            profile: HalProfile::NvidiaCudaDiscrete,
            os_arch: "linux-x86_64".to_string(),
            cpu_isa: "x86_64-avx2-fma".to_string(),
            cpu_threads: 16,
            vram_gb: 24.0,
            ram_gb: 64.0,
            nvme_stripes: 2,
            aggregate_nvme_gbps: 12.0,
            gpu_mem_bw_gbps: 600.0,
            resizable_bar: true,
            eco_thermal_quiet: true,
            standalone_mode: true,
        };
        let t0 = Instant::now();
        for _ in 0..iters {
            let vis = plan_vision_patch_tiles(black_box(1920), black_box(1080), 12).unwrap();
            let rec = recommend_best_model_for_machine(black_box(&hw));
            black_box((vis.total_tiles_with_thumb, rec.candidates.len()));
        }
        results.push(record("B30", "Colibri v2.2 Auto-Recommender (28m) + VisionTile", iters, t0));
    }

    // 31. RSI-COLIBRI-057 Upstream #3142 Single-Pass cc_norm_k_simd8 (256d) + Small-K F32 GEMV (16x64)
    {
        let x256 = vec![0.35f32; 256];
        let w256 = vec![1.0f32; 256];
        let mut out256 = vec![0.0f32; 256];
        let x64 = vec![0.25f32; 64];
        let w16x64 = vec![0.125f32; 16 * 64];
        let mut out16 = vec![0.0f32; 16];
        let t0 = Instant::now();
        for _ in 0..iters {
            let inv = cc_norm_k_simd8(black_box(&mut out256), black_box(&x256), black_box(&w256), 1e-6).unwrap();
            matmul_small_k_f32_simd4(black_box(&mut out16), black_box(&x64), black_box(&w16x64), 16, 64).unwrap();
            black_box((inv, out16[0]));
        }
        results.push(record("B31", "cc_norm_k_simd8 (256d) + small_k_f32 (16x64)", iters, t0));
    }

    // 32. RSI-COLIBRI-058 Upstream #3133 64-Slot MultiplexedChain64 + Packed Prefill Batch
    {
        let t0 = Instant::now();
        for _ in 0..iters {
            let mut ch = MultiplexedChain64::new(8);
            let s0 = ch.acquire_slot("tenant-a", 512).unwrap();
            let s1 = ch.acquire_slot("tenant-b", 512).unwrap();
            let packed = ch.build_packed_prefill_batch(768);
            ch.release_slot(s0);
            ch.release_slot(s1);
            black_box(packed.len());
        }
        results.push(record("B32", "MultiplexedChain64 (64-slot packed prefill)", iters, t0));
    }

    // 33. RSI-COLIBRI-059/060 DFlash2 6-Token Block Verifier + Native ANSI TUI Expert Map Render
    {
        let toks = [10u32, 20, 30, 40, 50, 60];
        let dp = [0.6f64; 6];
        let tp = [0.7f64; 6];
        let u = [0.2f64; 6];
        let rt = [0.2, 0.8];
        let rd = [0.5, 0.5];
        let hw = HardwareEnvelope {
            profile: HalProfile::AppleSiliconUma,
            os_arch: "macos-aarch64".to_string(),
            cpu_isa: "arm64-neon-fp16-dotprod".to_string(),
            cpu_threads: 16,
            vram_gb: 48.0,
            ram_gb: 16.0,
            nvme_stripes: 1,
            aggregate_nvme_gbps: 7.0,
            gpu_mem_bw_gbps: 546.0,
            resizable_bar: true,
            eco_thermal_quiet: true,
            standalone_mode: true,
        };
        let plan = compute_three_tier_plan(find_model("inkling-2-mini").unwrap(), &hw, 4, 8192);
        let t0 = Instant::now();
        for _ in 0..iters {
            let v = verify_dflash2_block_draft(
                black_box(&toks),
                black_box(&dp),
                black_box(&tp),
                black_box(&u),
                black_box(&rt),
                black_box(&rd),
                0.8,
            )
            .unwrap();
            let frame = crate::ui_html::render_tui_expert_map(black_box(&plan), 4, 0, "Reasoning step verified");
            black_box((v.accepted_count, frame.len()));
        }
        results.push(record("B33", "DFlash2 6-tok Verifier + Native ANSI TUI Map", iters, t0));
    }

    // 34. RSI-COLIBRI-061 Zero-Copy SlidingWindowKvRing (4 sinks + 256 sliding window push)
    {
        let mut ring = SlidingWindowKvRing::new(4, 256);
        let t0 = Instant::now();
        for i in 0..iters {
            ring.push_token(black_box(i as u32));
        }
        black_box(ring.active_kv_len());
        results.push(record("B34", "SlidingWindowKvRing (4 sinks + 256w O(1) push)", iters, t0));
    }

    // 35. RSI-COLIBRI-062 Single-Pass Online FlashAttention Softmax Kernel (32 seq x 64 head_dim)
    {
        let seq_len = 32usize;
        let head_dim = 64usize;
        let q = vec![0.25f32; head_dim];
        let k_seq = vec![0.125f32; seq_len * head_dim];
        let v_seq = vec![0.5f32; seq_len * head_dim];
        let mut out = vec![0.0f32; head_dim];
        let t0 = Instant::now();
        for _ in 0..iters {
            let m = flash_attn_online_softmax_simd8(
                black_box(&mut out),
                black_box(&q),
                black_box(&k_seq),
                black_box(&v_seq),
                seq_len,
                head_dim,
            )
            .unwrap();
            black_box((m, out[0]));
        }
        results.push(record("B35", "flash_attn_online_softmax_simd8 (32x64 L1)", iters, t0));
    }

    // 36. RSI-COLIBRI-063/064 Multi-LoRA Hot-Swap Validator + SEC-13 Cluster Peer Pin Handshake
    {
        let sha = "a".repeat(64);
        let mut pin_tbl = ClusterPeerPinTable::new();
        let fp = pin_tbl.pin_peer("worker-alpha", b"spki-der-cert-bytes", 2_000_000_000).unwrap();
        let t0 = Instant::now();
        for _ in 0..iters {
            let lp = validate_lora_hotswap("coder-r32", 32, 64.0, 4096, 4096, black_box(&sha), black_box(&sha)).unwrap();
            pin_tbl.verify_peer_handshake("worker-alpha", black_box(&fp), 1_760_000_000).unwrap();
            black_box(lp.rank);
        }
        results.push(record("B36", "LoraHotSwapPlan + SEC-13 Peer SPKI Pin Check", iters, t0));
    }

    // 37. RSI-COLIBRI-065 Zero-Compute N-Gram Prompt Lookup Draft Matcher (512-tok history)
    {
        let mut hist: Vec<u32> = (1u32..=512).collect();
        hist.extend_from_slice(&[120, 121, 122, 123]);
        let matcher = PromptNgramLookupMatcher::default();
        let t0 = Instant::now();
        for _ in 0..iters {
            let m = matcher.propose_draft_tokens(black_box(&hist), 8);
            black_box(m.draft_tokens.len());
        }
        results.push(record("B37", "PromptNgramLookupMatcher (512t -> 8 draft)", iters, t0));
    }

    // 38. RSI-COLIBRI-066 Native Pairwise-SIMD GQA 8-Head Broadcast Dot (8 Q-heads x 128d)
    {
        let q_group = vec![0.25f32; 8 * 128];
        let k_head = vec![0.5f32; 128];
        let mut scores = [0.0f32; 8];
        let t0 = Instant::now();
        for _ in 0..iters {
            let n = gqa_head_group_scores_simd8(
                black_box(&mut scores),
                black_box(&q_group),
                black_box(&k_head),
                8,
                128,
            )
            .unwrap();
            black_box((n, scores[0]));
        }
        results.push(record("B38", "gqa_head_group_scores_simd8 (8x128 GQA dot)", iters, t0));
    }

    // 39. RSI-COLIBRI-067/068 ReasoningStreamSplitter + SEC-14 Atomic UpstreamCircuitBreaker
    {
        let cb = UpstreamCircuitBreaker::new(3, 500, 8000);
        let t0 = Instant::now();
        for _ in 0..iters {
            let dec = cb.allow_request(black_box(1_000));
            let mut sp = ReasoningStreamSplitter::new(4096);
            let c1 = sp.feed_chunk(black_box("<think>plan</think>Final answer"));
            black_box((dec, c1.answer.len()));
        }
        results.push(record("B39", "ReasoningStreamSplitter + SEC-14 CircuitCheck", iters, t0));
    }

    // 40. RSI-COLIBRI-069 Zero-Copy Continuous Dynamic Batch Pack Planner (16 mixed requests)
    {
        let packer = DynamicBatchPacker::new(1024, 32, 512);
        let reqs: Vec<BatchRequest> = (0u16..16)
            .map(|i| BatchRequest {
                slot_id: i,
                remaining_tokens: if i % 2 == 0 { 1 } else { 320 },
                is_decode: i % 2 == 0,
                wait_ticks: i * 2,
            })
            .collect();
        let t0 = Instant::now();
        for _ in 0..iters {
            let wave = packer.pack_wave(black_box(&reqs));
            black_box(wave.total_tokens);
        }
        results.push(record("B40", "DynamicBatchPacker (16 mixed reqs -> 1024t)", iters, t0));
    }

    // 41. RSI-COLIBRI-070 Bit-Exact YaRN / NTK-Aware Frequency-Scaled RoPE (128d @ pos 262144)
    {
        let mut head = vec![0.5f32; 128];
        let t0 = Instant::now();
        for _ in 0..iters {
            head[0] = 0.5;
            let ms = rope_yarn_rotate_simd8(black_box(&mut head), 262_144, 128, 500_000.0, 8.0, 32_768)
                .unwrap();
            black_box((ms, head[0]));
        }
        results.push(record("B41", "rope_yarn_rotate_simd8 (128d YaRN @ 262k)", iters, t0));
    }

    // 42. RSI-COLIBRI-071/072 UTF-8 Tool-Output Trimmer + SEC-15 Streaming JSON Bounds Shield
    {
        let long_out = "HEADER_LOG_LINE_01\n".repeat(120) + "ERROR: root cause trace line 42";
        let sample_json = br#"{"model":"inkling-2-mini","messages":[{"role":"user","content":"status check"}]}"#;
        let t0 = Instant::now();
        for _ in 0..iters {
            let tr = trim_tool_output_utf8_safe(black_box(&long_out), 256);
            let st = validate_json_payload_bounds(black_box(sample_json), 32, 4096, 65_536).unwrap();
            black_box((tr.text.len(), st.max_depth_seen));
        }
        results.push(record("B42", "ToolOutputTrimmer + SEC-15 JSON Bounds Guard", iters, t0));
    }

    results
}

fn record(id: &'static str, subsystem: &'static str, iterations: u64, t0: Instant) -> BenchRow {
    let elapsed_ns = t0.elapsed().as_nanos() as f64;
    let ns_per_op = elapsed_ns / (iterations as f64);
    BenchRow {
        id,
        subsystem,
        iterations,
        ns_per_op: (ns_per_op * 100.0).round() / 100.0,
        us_per_op: ((ns_per_op / 1000.0) * 1000.0).round() / 1000.0,
    }
}
