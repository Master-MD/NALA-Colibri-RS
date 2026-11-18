//! 14-Subsystem Deterministic 10,000-Iteration Self-Benchmark Suite (`--bench` / `--bench --json`).
//! Uses `std::hint::black_box` and `std::time::Instant` to measure real `ns/op` and `us/op` on host silicon.

use crate::catalog::{CATALOG_MODELS, find_model};
use crate::containers::{parse_gguf_header, parse_qpack_header, parse_safetensors_header};
use crate::expert_store::{ColiParkVault, ExpertStore, GenerationScheduler, KvPrefixCache};
use crate::kernels::{bf16_to_f32_bulk, f16_to_f32_bulk, matmul_q_gs64_int4, matmul_q_gs64_int8};
use crate::mcp::handle_mcp_jsonrpc;
use crate::planner::{HalProfile, HardwareEnvelope, compute_three_tier_plan};
use crate::security::{hmac_sha256, is_strictly_public_ip, verify_hmac_sha256_hex, hex_encode};
use crate::system_one::{SystemOneMode, evaluate_system_one};
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
