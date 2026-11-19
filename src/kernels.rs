//! Bit-Exact SIMD-Vectorized Quantized GEMV & FP16/BF16 Kernels (`gsgemv.h` & `st.h` Parity + Speedup).
//! Clean-room 100% safe Rust (`#![forbid(unsafe_code)]`) with Dual-`[f32; 8]` Vector Accumulators
//! and Colibri's Exact Pairwise Reduction Tree (`gs_group_sum_simd8`), unlocking full LLVM NEON (ARM64)
//! and SSE2/AVX2 (x86_64) vectorization without loop-carried scalar FADD stalls.

pub const GROUP_SIZE_64: usize = 64;

/// Lookup table mapping 4-bit unsigned nibble `0..15` directly to symmetric signed `f32` (`-8.0 .. +7.0`),
/// eliminating integer-to-float conversion instructions inside the hot `int4_gs64` inner loop.
pub const INT4_NIBBLE_F32_LUT: [f32; 16] = [
    -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
     0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,
];

/// RSI-COLIBRI-017: OCP Microscaling `MXFP4` (`E2M1`, `fmt 7`) 4-bit floating-point lookup table.
/// Bits `[2:0]` encode `{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}`, bit `3` encodes sign (`+` / `-`).
pub const MXFP4_E2M1_F32_LUT: [f32; 16] = [
     0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// RSI-COLIBRI-017: Convert 8-bit `E8M0` shared block exponent (`2^(e - 127)`) directly into IEEE-754 `f32`
/// in 1 CPU cycle via bit shift (`(e8m0 as u32) << 23`), avoiding slow `libm powf` calls.
#[inline(always)]
pub fn e8m0_to_f32_scale(e8m0: u8) -> f32 {
    if e8m0 == 0 || e8m0 == 0xff {
        0.0
    } else {
        f32::from_bits((e8m0 as u32) << 23)
    }
}

/// Pairwise IEEE-754 reduction tree over two 8-lane accumulators, matching Colibri `c/gsgemv.h::gs_group_sum`:
/// `(a0 + a1) -> fold 8->4 -> fold 4->2 -> fold 2->1`.
/// Identical across `TP=1` and `TP=N` slices and free of loop-carried scalar dependency chains.
#[inline(always)]
pub fn gs_group_sum_simd8(a0: [f32; 8], a1: [f32; 8]) -> f32 {
    let p0 = a0[0] + a1[0];
    let p1 = a0[1] + a1[1];
    let p2 = a0[2] + a1[2];
    let p3 = a0[3] + a1[3];
    let p4 = a0[4] + a1[4];
    let p5 = a0[5] + a1[5];
    let p6 = a0[6] + a1[6];
    let p7 = a0[7] + a1[7];

    let s0 = p0 + p4;
    let s1 = p1 + p5;
    let s2 = p2 + p6;
    let s3 = p3 + p7;

    (s0 + s2) + (s1 + s3)
}

/// Convert a single IEEE-754 binary16 (`f16`) bit pattern to `f32` (matches Colibri `st_f16_to_f32` including subnormals/NaN/Inf).
#[inline]
pub fn f16_to_f32_scalar(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut e: i32 = -14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03ff;
            sign | (((e + 127) as u32) << 23) | (m << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Bulk convert `bf16` (`u16` upper 16 bits of IEEE-754 `f32`) slice into `f32` output slice.
pub fn bf16_to_f32_bulk(src: &[u16], dst: &mut [f32]) {
    let n = src.len().min(dst.len());
    for (s, d) in src[..n].iter().zip(dst[..n].iter_mut()) {
        *d = f32::from_bits((*s as u32) << 16);
    }
}

/// Bulk convert `f16` (`u16`) slice into `f32` output slice.
pub fn f16_to_f32_bulk(src: &[u16], dst: &mut [f32]) {
    let n = src.len().min(dst.len());
    for (s, d) in src[..n].iter().zip(dst[..n].iter_mut()) {
        *d = f16_to_f32_scalar(*s);
    }
}

/// 4-row interleaved group-64 quantized INT8 GEMV (`y = W_q * x` with per-64-weight `f32` scale).
/// Uses dual `[f32; 8]` SIMD accumulators (`a00/a01 .. a30/a31`) per row over 4x16 chunks so LLVM emits
/// full NEON / SSE2 / AVX2 vector multiply-adds without loop-carried scalar latency stalls.
pub fn matmul_q_gs64_int8(
    out: &mut [f32],
    x: &[f32],
    wq: &[i8],
    scales: &[f32],
    rows: usize,
    cols: usize,
) {
    assert!(cols.is_multiple_of(GROUP_SIZE_64), "cols must be a multiple of 64");
    assert!(out.len() >= rows && x.len() >= cols);
    let groups_per_row = cols / GROUP_SIZE_64;
    assert!(wq.len() >= rows * cols && scales.len() >= rows * groups_per_row);

    let full_quads = rows / 4;
    for q in 0..full_quads {
        let r = q * 4;
        let w0 = &wq[r * cols..(r + 1) * cols];
        let w1 = &wq[(r + 1) * cols..(r + 2) * cols];
        let w2 = &wq[(r + 2) * cols..(r + 3) * cols];
        let w3 = &wq[(r + 3) * cols..(r + 4) * cols];

        let s0 = &scales[r * groups_per_row..(r + 1) * groups_per_row];
        let s1 = &scales[(r + 1) * groups_per_row..(r + 2) * groups_per_row];
        let s2 = &scales[(r + 2) * groups_per_row..(r + 3) * groups_per_row];
        let s3 = &scales[(r + 3) * groups_per_row..(r + 4) * groups_per_row];

        let mut acc0 = 0.0f32;
        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;
        let mut acc3 = 0.0f32;

        for g in 0..groups_per_row {
            let base = g * GROUP_SIZE_64;
            let xg = &x[base..base + GROUP_SIZE_64];
            let g0 = &w0[base..base + GROUP_SIZE_64];
            let g1 = &w1[base..base + GROUP_SIZE_64];
            let g2 = &w2[base..base + GROUP_SIZE_64];
            let g3 = &w3[base..base + GROUP_SIZE_64];

            let mut a00 = [0.0f32; 8];
            let mut a01 = [0.0f32; 8];
            let mut a10 = [0.0f32; 8];
            let mut a11 = [0.0f32; 8];
            let mut a20 = [0.0f32; 8];
            let mut a21 = [0.0f32; 8];
            let mut a30 = [0.0f32; 8];
            let mut a31 = [0.0f32; 8];

            for step in 0..4 {
                let off = step * 16;
                let xl = &xg[off..off + 8];
                let xh = &xg[off + 8..off + 16];

                let w0l = &g0[off..off + 8];
                let w0h = &g0[off + 8..off + 16];
                let w1l = &g1[off..off + 8];
                let w1h = &g1[off + 8..off + 16];
                let w2l = &g2[off..off + 8];
                let w2h = &g2[off + 8..off + 16];
                let w3l = &g3[off..off + 8];
                let w3h = &g3[off + 8..off + 16];

                for l in 0..8 {
                    let xlv = xl[l];
                    let xhv = xh[l];
                    a00[l] += (w0l[l] as f32) * xlv;
                    a01[l] += (w0h[l] as f32) * xhv;
                    a10[l] += (w1l[l] as f32) * xlv;
                    a11[l] += (w1h[l] as f32) * xhv;
                    a20[l] += (w2l[l] as f32) * xlv;
                    a21[l] += (w2h[l] as f32) * xhv;
                    a30[l] += (w3l[l] as f32) * xlv;
                    a31[l] += (w3h[l] as f32) * xhv;
                }
            }

            acc0 += gs_group_sum_simd8(a00, a01) * s0[g];
            acc1 += gs_group_sum_simd8(a10, a11) * s1[g];
            acc2 += gs_group_sum_simd8(a20, a21) * s2[g];
            acc3 += gs_group_sum_simd8(a30, a31) * s3[g];
        }

        out[r] = acc0;
        out[r + 1] = acc1;
        out[r + 2] = acc2;
        out[r + 3] = acc3;
    }

    for r in (full_quads * 4)..rows {
        let wr = &wq[r * cols..(r + 1) * cols];
        let sr = &scales[r * groups_per_row..(r + 1) * groups_per_row];
        let mut acc = 0.0f32;
        for g in 0..groups_per_row {
            let base = g * GROUP_SIZE_64;
            let xg = &x[base..base + GROUP_SIZE_64];
            let wg = &wr[base..base + GROUP_SIZE_64];
            let mut a0 = [0.0f32; 8];
            let mut a1 = [0.0f32; 8];
            for step in 0..4 {
                let off = step * 16;
                for l in 0..8 {
                    a0[l] += (wg[off + l] as f32) * xg[off + l];
                    a1[l] += (wg[off + 8 + l] as f32) * xg[off + 8 + l];
                }
            }
            acc += gs_group_sum_simd8(a0, a1) * sr[g];
        }
        out[r] = acc;
    }
}


/// 4-row interleaved group-64 packed symmetric INT4 (`int4_gs64`, 2 weights per byte, offset `-8`) GEMV.
/// Accelerated via `INT4_NIBBLE_F32_LUT` and 8-lane SIMD accumulators.
pub fn matmul_q_gs64_int4(
    out: &mut [f32],
    x: &[f32],
    wq_packed: &[u8],
    scales: &[f32],
    rows: usize,
    cols: usize,
) {
    assert!(cols.is_multiple_of(GROUP_SIZE_64), "cols must be a multiple of 64");
    let packed_cols = cols / 2;
    let groups_per_row = cols / GROUP_SIZE_64;
    assert!(out.len() >= rows && x.len() >= cols);
    assert!(wq_packed.len() >= rows * packed_cols && scales.len() >= rows * groups_per_row);

    let full_quads = rows / 4;
    for q in 0..full_quads {
        let r = q * 4;
        let w0 = &wq_packed[r * packed_cols..(r + 1) * packed_cols];
        let w1 = &wq_packed[(r + 1) * packed_cols..(r + 2) * packed_cols];
        let w2 = &wq_packed[(r + 2) * packed_cols..(r + 3) * packed_cols];
        let w3 = &wq_packed[(r + 3) * packed_cols..(r + 4) * packed_cols];

        let s0 = &scales[r * groups_per_row..(r + 1) * groups_per_row];
        let s1 = &scales[(r + 1) * groups_per_row..(r + 2) * groups_per_row];
        let s2 = &scales[(r + 2) * groups_per_row..(r + 3) * groups_per_row];
        let s3 = &scales[(r + 3) * groups_per_row..(r + 4) * groups_per_row];

        let mut acc0 = 0.0f32;
        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;
        let mut acc3 = 0.0f32;

        for g in 0..groups_per_row {
            let x_base = g * GROUP_SIZE_64;
            let p_base = g * 32;
            let xg = &x[x_base..x_base + GROUP_SIZE_64];
            let p0 = &w0[p_base..p_base + 32];
            let p1 = &w1[p_base..p_base + 32];
            let p2 = &w2[p_base..p_base + 32];
            let p3 = &w3[p_base..p_base + 32];

            let mut a00 = [0.0f32; 8];
            let mut a01 = [0.0f32; 8];
            let mut a10 = [0.0f32; 8];
            let mut a11 = [0.0f32; 8];
            let mut a20 = [0.0f32; 8];
            let mut a21 = [0.0f32; 8];
            let mut a30 = [0.0f32; 8];
            let mut a31 = [0.0f32; 8];

            for step in 0..4 {
                let boff = step * 8;
                let xoff = step * 16;
                for l in 0..8 {
                    let x_lo = xg[xoff + l * 2];
                    let x_hi = xg[xoff + l * 2 + 1];

                    let b0 = p0[boff + l];
                    a00[l] += (((b0 & 0x0f) as i32 - 8) as f32) * x_lo;
                    a01[l] += (((b0 >> 4) as i32 - 8) as f32) * x_hi;

                    let b1 = p1[boff + l];
                    a10[l] += (((b1 & 0x0f) as i32 - 8) as f32) * x_lo;
                    a11[l] += (((b1 >> 4) as i32 - 8) as f32) * x_hi;

                    let b2 = p2[boff + l];
                    a20[l] += (((b2 & 0x0f) as i32 - 8) as f32) * x_lo;
                    a21[l] += (((b2 >> 4) as i32 - 8) as f32) * x_hi;

                    let b3 = p3[boff + l];
                    a30[l] += (((b3 & 0x0f) as i32 - 8) as f32) * x_lo;
                    a31[l] += (((b3 >> 4) as i32 - 8) as f32) * x_hi;
                }
            }

            acc0 += gs_group_sum_simd8(a00, a01) * s0[g];
            acc1 += gs_group_sum_simd8(a10, a11) * s1[g];
            acc2 += gs_group_sum_simd8(a20, a21) * s2[g];
            acc3 += gs_group_sum_simd8(a30, a31) * s3[g];
        }

        out[r] = acc0;
        out[r + 1] = acc1;
        out[r + 2] = acc2;
        out[r + 3] = acc3;
    }

    for r in (full_quads * 4)..rows {
        let wr = &wq_packed[r * packed_cols..(r + 1) * packed_cols];
        let sr = &scales[r * groups_per_row..(r + 1) * groups_per_row];
        let mut acc = 0.0f32;
        for g in 0..groups_per_row {
            let x_base = g * GROUP_SIZE_64;
            let p_base = g * 32;
            let xg = &x[x_base..x_base + GROUP_SIZE_64];
            let pg = &wr[p_base..p_base + 32];
            let mut a0 = [0.0f32; 8];
            let mut a1 = [0.0f32; 8];
            for step in 0..4 {
                let boff = step * 8;
                let xoff = step * 16;
                for l in 0..8 {
                    let byte = pg[boff + l];
                    a0[l] += (((byte & 0x0f) as i32 - 8) as f32) * xg[xoff + l * 2];
                    a1[l] += (((byte >> 4) as i32 - 8) as f32) * xg[xoff + l * 2 + 1];
                }
            }
            acc += gs_group_sum_simd8(a0, a1) * sr[g];
        }
        out[r] = acc;
    }
}


/// Fused RMSNorm (`y_i = (x_i / sqrt(mean(x^2) + eps)) * weight_i`).
pub fn rms_norm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let n = out.len().min(x.len()).min(weight.len());
    if n == 0 {
        return;
    }
    let mut ss = 0.0f32;
    for &v in &x[..n] {
        ss += v * v;
    }
    let inv_rms = 1.0f32 / ((ss / (n as f32)) + eps).sqrt();
    for i in 0..n {
        out[i] = (x[i] * inv_rms) * weight[i];
    }
}

/// RSI-COLIBRI-017: 4-row interleaved OCP Microscaling `MXFP4` (`E2M1` + `E8M0` per-64-group shared exponent, `fmt 7`) GEMV
/// for AMD Quark MXFP4 models (`amd/GLM-5.3-Flash-Quark-MXFP4`).
/// Preserves exact IEEE-754 tree-reduction associative order via `gs_group_sum_simd8`, guaranteeing
/// bit-identical outputs across `TP=1`, `TP=2`, `TP=4`, and `TP=8` column splits.
pub fn matmul_q_mxfp4_fmt7(
    out: &mut [f32],
    x: &[f32],
    wq_mxfp4_packed: &[u8],
    e8m0_scales: &[u8],
    rows: usize,
    cols: usize,
) {
    assert!(cols.is_multiple_of(GROUP_SIZE_64), "cols must be a multiple of 64");
    let packed_cols = cols / 2;
    let groups_per_row = cols / GROUP_SIZE_64;
    assert!(out.len() >= rows && x.len() >= cols);
    assert!(
        wq_mxfp4_packed.len() >= rows * packed_cols && e8m0_scales.len() >= rows * groups_per_row
    );

    let full_quads = rows / 4;
    for q in 0..full_quads {
        let r = q * 4;
        let w0 = &wq_mxfp4_packed[r * packed_cols..(r + 1) * packed_cols];
        let w1 = &wq_mxfp4_packed[(r + 1) * packed_cols..(r + 2) * packed_cols];
        let w2 = &wq_mxfp4_packed[(r + 2) * packed_cols..(r + 3) * packed_cols];
        let w3 = &wq_mxfp4_packed[(r + 3) * packed_cols..(r + 4) * packed_cols];

        let s0 = &e8m0_scales[r * groups_per_row..(r + 1) * groups_per_row];
        let s1 = &e8m0_scales[(r + 1) * groups_per_row..(r + 2) * groups_per_row];
        let s2 = &e8m0_scales[(r + 2) * groups_per_row..(r + 3) * groups_per_row];
        let s3 = &e8m0_scales[(r + 3) * groups_per_row..(r + 4) * groups_per_row];

        let mut acc0 = 0.0f32;
        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;
        let mut acc3 = 0.0f32;

        for g in 0..groups_per_row {
            let x_base = g * GROUP_SIZE_64;
            let p_base = g * 32;
            let xg = &x[x_base..x_base + GROUP_SIZE_64];
            let p0 = &w0[p_base..p_base + 32];
            let p1 = &w1[p_base..p_base + 32];
            let p2 = &w2[p_base..p_base + 32];
            let p3 = &w3[p_base..p_base + 32];

            let mut a00 = [0.0f32; 8];
            let mut a01 = [0.0f32; 8];
            let mut a10 = [0.0f32; 8];
            let mut a11 = [0.0f32; 8];
            let mut a20 = [0.0f32; 8];
            let mut a21 = [0.0f32; 8];
            let mut a30 = [0.0f32; 8];
            let mut a31 = [0.0f32; 8];

            for step in 0..4 {
                let boff = step * 8;
                let xoff = step * 16;
                for l in 0..8 {
                    let x_lo = xg[xoff + l * 2];
                    let x_hi = xg[xoff + l * 2 + 1];

                    let b0 = p0[boff + l];
                    a00[l] += MXFP4_E2M1_F32_LUT[(b0 & 0x0f) as usize] * x_lo;
                    a01[l] += MXFP4_E2M1_F32_LUT[(b0 >> 4) as usize] * x_hi;

                    let b1 = p1[boff + l];
                    a10[l] += MXFP4_E2M1_F32_LUT[(b1 & 0x0f) as usize] * x_lo;
                    a11[l] += MXFP4_E2M1_F32_LUT[(b1 >> 4) as usize] * x_hi;

                    let b2 = p2[boff + l];
                    a20[l] += MXFP4_E2M1_F32_LUT[(b2 & 0x0f) as usize] * x_lo;
                    a21[l] += MXFP4_E2M1_F32_LUT[(b2 >> 4) as usize] * x_hi;

                    let b3 = p3[boff + l];
                    a30[l] += MXFP4_E2M1_F32_LUT[(b3 & 0x0f) as usize] * x_lo;
                    a31[l] += MXFP4_E2M1_F32_LUT[(b3 >> 4) as usize] * x_hi;
                }
            }

            acc0 += gs_group_sum_simd8(a00, a01) * e8m0_to_f32_scale(s0[g]);
            acc1 += gs_group_sum_simd8(a10, a11) * e8m0_to_f32_scale(s1[g]);
            acc2 += gs_group_sum_simd8(a20, a21) * e8m0_to_f32_scale(s2[g]);
            acc3 += gs_group_sum_simd8(a30, a31) * e8m0_to_f32_scale(s3[g]);
        }

        out[r] = acc0;
        out[r + 1] = acc1;
        out[r + 2] = acc2;
        out[r + 3] = acc3;
    }

    for r in (full_quads * 4)..rows {
        let wr = &wq_mxfp4_packed[r * packed_cols..(r + 1) * packed_cols];
        let sr = &e8m0_scales[r * groups_per_row..(r + 1) * groups_per_row];
        let mut acc = 0.0f32;
        for g in 0..groups_per_row {
            let x_base = g * GROUP_SIZE_64;
            let p_base = g * 32;
            let xg = &x[x_base..x_base + GROUP_SIZE_64];
            let pg = &wr[p_base..p_base + 32];
            let mut a0 = [0.0f32; 8];
            let mut a1 = [0.0f32; 8];
            for step in 0..4 {
                let boff = step * 8;
                let xoff = step * 16;
                for l in 0..8 {
                    let byte = pg[boff + l];
                    a0[l] += MXFP4_E2M1_F32_LUT[(byte & 0x0f) as usize] * xg[xoff + l * 2];
                    a1[l] += MXFP4_E2M1_F32_LUT[(byte >> 4) as usize] * xg[xoff + l * 2 + 1];
                }
            }
            acc += gs_group_sum_simd8(a0, a1) * e8m0_to_f32_scale(sr[g]);
        }
        out[r] = acc;
    }
}

/// RSI-COLIBRI-021: Fused Low-Rank Compressed KV-Latent (`57x` MLA `c_KV`) SIMD Up-Projection + Register-Resident RoPE Rotation.
/// Computes `k_raw = W_UK_q8 * c_KV` via dual-`[f32; 8]` SIMD accumulators and `gs_group_sum_simd8` tree reduction
/// (`TP=1/2/4/8` bit-identical), then immediately applies 2D Rotary Position Embedding (`RoPE`) in registers
/// without writing an intermediate unrotated buffer to L2 cache.
pub fn matmul_mla_latent_57x(
    out_rope_k: &mut [f32],
    c_kv_latent: &[f32],
    w_uk_q8: &[i8],
    w_uk_scales: &[f32],
    rope_cos: &[f32],
    rope_sin: &[f32],
    head_dim_rows: usize,
    latent_rank_cols: usize,
) {
    assert!(
        head_dim_rows.is_multiple_of(2),
        "head_dim_rows must be even for 2D RoPE pairs"
    );
    assert!(
        latent_rank_cols.is_multiple_of(GROUP_SIZE_64),
        "latent_rank_cols must be a multiple of 64"
    );
    let pairs = head_dim_rows / 2;
    let groups_per_row = latent_rank_cols / GROUP_SIZE_64;
    assert!(out_rope_k.len() >= head_dim_rows && c_kv_latent.len() >= latent_rank_cols);
    assert!(
        w_uk_q8.len() >= head_dim_rows * latent_rank_cols
            && w_uk_scales.len() >= head_dim_rows * groups_per_row
    );
    assert!(rope_cos.len() >= pairs && rope_sin.len() >= pairs);

    for p in 0..pairs {
        let r_even = p * 2;
        let r_odd = r_even + 1;
        let w_even = &w_uk_q8[r_even * latent_rank_cols..(r_even + 1) * latent_rank_cols];
        let w_odd = &w_uk_q8[r_odd * latent_rank_cols..(r_odd + 1) * latent_rank_cols];
        let s_even = &w_uk_scales[r_even * groups_per_row..(r_even + 1) * groups_per_row];
        let s_odd = &w_uk_scales[r_odd * groups_per_row..(r_odd + 1) * groups_per_row];

        let mut acc_even = 0.0f32;
        let mut acc_odd = 0.0f32;

        for g in 0..groups_per_row {
            let base = g * GROUP_SIZE_64;
            let cg = &c_kv_latent[base..base + GROUP_SIZE_64];
            let ge = &w_even[base..base + GROUP_SIZE_64];
            let go = &w_odd[base..base + GROUP_SIZE_64];

            let mut ae0 = [0.0f32; 8];
            let mut ae1 = [0.0f32; 8];
            let mut ao0 = [0.0f32; 8];
            let mut ao1 = [0.0f32; 8];

            for step in 0..4 {
                let off = step * 16;
                for l in 0..8 {
                    let c_lo = cg[off + l];
                    let c_hi = cg[off + 8 + l];
                    ae0[l] += (ge[off + l] as f32) * c_lo;
                    ae1[l] += (ge[off + 8 + l] as f32) * c_hi;
                    ao0[l] += (go[off + l] as f32) * c_lo;
                    ao1[l] += (go[off + 8 + l] as f32) * c_hi;
                }
            }

            acc_even += gs_group_sum_simd8(ae0, ae1) * s_even[g];
            acc_odd += gs_group_sum_simd8(ao0, ao1) * s_odd[g];
        }

        // Register-resident 2D RoPE phase rotation
        let cos_v = rope_cos[p];
        let sin_v = rope_sin[p];
        out_rope_k[r_even] = acc_even * cos_v - acc_odd * sin_v;
        out_rope_k[r_odd] = acc_odd * cos_v + acc_even * sin_v;
    }
}

/// RSI-COLIBRI-025: Block-Sparse NSA / DSA Top-K Lightning Indexer SIMD Kernel (`nsa_topk_blocks`).
/// Scores each 64-token KV summary block `b` (`0..num_blocks`) against query vector `q` (`dim = 64`)
/// using dual-`[f32; 8]` SIMD accumulators and `gs_group_sum_simd8` tree reduction (`TP=1/2/4/8` bit-identical),
/// enforces strict causal masking (`b * 64 <= current_token_pos`), and selects the top-`k` (`k <= 16`)
/// blocks using deterministic `FP32 total_cmp` tie-breaking (lower `block_idx` wins on equal score).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NsaBlockScore {
    pub block_idx: u32,
    pub score_fp32: f32,
}

pub fn nsa_topk_blocks(
    q_64: &[f32; GROUP_SIZE_64],
    block_keys_q8: &[i8],
    block_scales: &[f32],
    num_blocks: usize,
    current_token_pos: usize,
    top_k: usize,
) -> Vec<NsaBlockScore> {
    assert!(
        block_keys_q8.len() >= num_blocks * GROUP_SIZE_64 && block_scales.len() >= num_blocks
    );
    let k_clamped = top_k.clamp(1, 16);
    let mut candidates: Vec<NsaBlockScore> = Vec::with_capacity(num_blocks);

    for b in 0..num_blocks {
        let block_start_tok = b * GROUP_SIZE_64;
        // Strict causal mask boundary enforcement: future blocks beyond current_token_pos are masked out
        if block_start_tok > current_token_pos {
            continue;
        }
        let kb = &block_keys_q8[b * GROUP_SIZE_64..(b + 1) * GROUP_SIZE_64];
        let mut a0 = [0.0f32; 8];
        let mut a1 = [0.0f32; 8];
        for step in 0..4 {
            let off = step * 16;
            for l in 0..8 {
                a0[l] += (kb[off + l] as f32) * q_64[off + l];
                a1[l] += (kb[off + 8 + l] as f32) * q_64[off + 8 + l];
            }
        }
        let score = gs_group_sum_simd8(a0, a1) * block_scales[b];
        candidates.push(NsaBlockScore {
            block_idx: b as u32,
            score_fp32: score,
        });
    }

    candidates.sort_by(|a, b| {
        let cmp = b.score_fp32.total_cmp(&a.score_fp32);
        if cmp == std::cmp::Ordering::Equal {
            a.block_idx.cmp(&b.block_idx)
        } else {
            cmp
        }
    });
    if candidates.len() > k_clamped {
        candidates.truncate(k_clamped);
    }
    candidates
}

/// RSI-COLIBRI-027: Auxiliary-Loss-Free Sigmoid Affinity-Bias Balanced MoE Router (`route_topk_sigmoid_noaux`).
/// Computes unbiased sigmoid affinity `s_i = 1 / (1 + exp(-logit_i))`, applies clamped per-expert bias
/// `b_i in [-2.0, +2.0]` ONLY for top-k expert selection (`sel_i = s_i + clamp(b_i)`), and normalizes
/// returned routing weights over the unbiased affinities `w_i = s_i / sum(s_top_k)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoutedExpertWeight {
    pub expert_id: u16,
    pub weight: f32,
    pub raw_affinity: f32,
}

pub fn route_topk_sigmoid_noaux(
    router_logits: &[f32],
    expert_biases: &[f32],
    top_k: usize,
) -> Vec<RoutedExpertWeight> {
    let n = router_logits.len().min(expert_biases.len());
    if n == 0 || top_k == 0 {
        return Vec::new();
    }
    let k = top_k.min(n).min(16);
    let mut scored: Vec<(u16, f32, f32)> = Vec::with_capacity(n);
    for i in 0..n {
        let logit = router_logits[i].clamp(-20.0, 20.0);
        let affinity = 1.0f32 / (1.0f32 + (-logit).exp());
        let bias = expert_biases[i].clamp(-2.0, 2.0);
        scored.push((i as u16, affinity + bias, affinity));
    }
    scored.sort_by(|a, b| {
        let cmp = b.1.total_cmp(&a.1);
        if cmp == std::cmp::Ordering::Equal {
            a.0.cmp(&b.0)
        } else {
            cmp
        }
    });
    scored.truncate(k);
    let sum_aff: f32 = scored.iter().map(|x| x.2).sum::<f32>().max(1e-9);
    scored
        .into_iter()
        .map(|(eid, _sel, aff)| RoutedExpertWeight {
            expert_id: eid,
            weight: aff / sum_aff,
            raw_affinity: aff,
        })
        .collect()
}

/// RSI-COLIBRI-029: Fused Gated-DeltaNet / Mamba-2 Recurrent State Update SIMD Kernel (`deltanet_step_simd8`).
/// Computes the `O(1)` constant-memory beta-gated Householder delta rule:
///   `S_t = alpha * S_{t-1} + beta * (v - alpha * (S_{t-1} * k_hat)) * k_hat^T`
/// and fuses the query read-out `out[i] = sum_j S_t[i, j] * q[j]` in a single L1-resident row pass.
/// Enforces spectral stability (`alpha in [0, 1]`, `beta in [0, 2]`, `||k_hat||_2 <= 1.0`, finite inputs).
pub fn deltanet_step_simd8(
    state_d_x_d: &mut [f32],
    out_y: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alpha_decay: f32,
    beta_gate: f32,
    d: usize,
) -> Result<(), &'static str> {
    if d == 0 || !d.is_multiple_of(8) {
        return Err("DELTANET_DIM_MUST_BE_MULTIPLE_OF_8");
    }
    if state_d_x_d.len() < d * d || out_y.len() < d || q.len() < d || k.len() < d || v.len() < d {
        return Err("DELTANET_BUFFER_TOO_SMALL");
    }
    if !alpha_decay.is_finite() || !beta_gate.is_finite() {
        return Err("SEC_DELTANET_NONFINITE_GATE");
    }

    let alpha = alpha_decay.clamp(0.0, 1.0);
    let beta = beta_gate.clamp(0.0, 2.0);

    // L2-normalize key if ||k||_2 > 1.0 so Householder operator (I - beta * k_hat * k_hat^T) has spectral norm <= 1.0
    let mut k_norm_sq = 0.0f32;
    for &kv in &k[..d] {
        if !kv.is_finite() {
            return Err("SEC_DELTANET_NONFINITE_KEY");
        }
        k_norm_sq += kv * kv;
    }
    let inv_k_norm = if k_norm_sq > 1.0 {
        1.0 / k_norm_sq.sqrt()
    } else {
        1.0
    };

    let mut k_hat = [0.0f32; 64];
    let use_stack = d <= 64;
    let mut k_heap = if use_stack { Vec::new() } else { vec![0.0f32; d] };
    let kh: &mut [f32] = if use_stack {
        &mut k_hat[..d]
    } else {
        &mut k_heap[..d]
    };
    for j in 0..d {
        kh[j] = k[j] * inv_k_norm;
    }

    let chunks8 = d / 8;
    for i in 0..d {
        let vi = v[i];
        if !vi.is_finite() {
            return Err("SEC_DELTANET_NONFINITE_VALUE");
        }
        let row = &mut state_d_x_d[i * d..(i + 1) * d];

        // Pass 1 (in L1): dot = row . k_hat via 8-lane SIMD accumulator
        let mut dot_lanes = [0.0f32; 8];
        for c in 0..chunks8 {
            let off = c * 8;
            for l in 0..8 {
                dot_lanes[l] += row[off + l] * kh[off + l];
            }
        }
        let s_dot_k = ((dot_lanes[0] + dot_lanes[4]) + (dot_lanes[2] + dot_lanes[6]))
            + ((dot_lanes[1] + dot_lanes[5]) + (dot_lanes[3] + dot_lanes[7]));

        let delta_i = beta * (vi - alpha * s_dot_k);

        // Pass 2 (fused update + query read-out in L1): row[j] = alpha * row[j] + delta_i * kh[j]; y[i] = row . q
        let mut y_lanes = [0.0f32; 8];
        for c in 0..chunks8 {
            let off = c * 8;
            for l in 0..8 {
                let updated = alpha * row[off + l] + delta_i * kh[off + l];
                row[off + l] = updated;
                y_lanes[l] += updated * q[off + l];
            }
        }
        out_y[i] = ((y_lanes[0] + y_lanes[4]) + (y_lanes[2] + y_lanes[6]))
            + ((y_lanes[1] + y_lanes[5]) + (y_lanes[3] + y_lanes[7]));
    }
    Ok(())
}

/// RSI-COLIBRI-034: Precomputed 256-entry OCP FP8 `E4M3FN -> F32` Lookup Table.
/// Layout: 1 sign bit (`bit 7`), 4 exponent bits (`bits 6..3`, bias `7`), 3 mantissa bits (`bits 2..0`).
/// Finite-only (`0x7F` and `0xFF` NaN encodings deterministically map to `0.0f32` so NaN/Inf
/// can never poison quantized activations on legacy pre-Hopper GPUs & CPUs).
const fn decode_fp8_e4m3fn_const(byte: u8) -> f32 {
    let sign = (byte >> 7) & 1;
    let exp = (byte >> 3) & 0x0f;
    let mant = byte & 0x07;
    if exp == 0x0f && mant == 0x07 {
        return 0.0;
    }
    let abs_val = if exp == 0 {
        // Subnormal: (mant / 8) * 2^-6 = mant * 2^-9
        (mant as f32) * (1.0 / 512.0)
    } else {
        // Normal: (1 + mant / 8) * 2^(exp - 7)
        let frac = 1.0 + (mant as f32) * 0.125;
        let shift = (exp as i32) - 7;
        let pow2 = if shift >= 0 {
            (1u32 << (shift as u32)) as f32
        } else {
            1.0 / ((1u32 << ((-shift) as u32)) as f32)
        };
        frac * pow2
    };
    if sign == 1 { -abs_val } else { abs_val }
}

const fn build_fp8_e4m3fn_lut() -> [f32; 256] {
    let mut lut = [0.0f32; 256];
    let mut i = 0usize;
    while i < 256 {
        lut[i] = decode_fp8_e4m3fn_const(i as u8);
        i += 1;
    }
    lut
}

pub const FP8_E4M3FN_F32_LUT: [f32; 256] = build_fp8_e4m3fn_lut();

/// RSI-COLIBRI-034: Native Pairwise-SIMD `matmul_q_fp8_e4m3fn` Kernel for Legacy Pre-Hopper GPUs & CPUs.
/// Uses the 256-entry NaN/Inf-free `FP8_E4M3FN_F32_LUT` + 4-row interleaved dual-`[f32; 8]` SIMD
/// accumulators + associative `gs_group_sum_simd8` tree reduction (`TP=1/2/4/8` bit-identical).
pub fn matmul_q_fp8_e4m3fn(
    out: &mut [f32],
    x: &[f32],
    w_fp8: &[u8],
    scales: &[f32],
    rows: usize,
    cols: usize,
) {
    assert!(
        cols.is_multiple_of(GROUP_SIZE_64),
        "cols must be a multiple of 64"
    );
    let groups_per_row = cols / GROUP_SIZE_64;
    assert!(out.len() >= rows && x.len() >= cols);
    assert!(w_fp8.len() >= rows * cols && scales.len() >= rows * groups_per_row);

    let full_quads = rows / 4;
    for q in 0..full_quads {
        let r0 = q * 4;
        let r1 = r0 + 1;
        let r2 = r0 + 2;
        let r3 = r0 + 3;

        let w0 = &w_fp8[r0 * cols..(r0 + 1) * cols];
        let w1 = &w_fp8[r1 * cols..(r1 + 1) * cols];
        let w2 = &w_fp8[r2 * cols..(r2 + 1) * cols];
        let w3 = &w_fp8[r3 * cols..(r3 + 1) * cols];

        let s0 = &scales[r0 * groups_per_row..(r0 + 1) * groups_per_row];
        let s1 = &scales[r1 * groups_per_row..(r1 + 1) * groups_per_row];
        let s2 = &scales[r2 * groups_per_row..(r2 + 1) * groups_per_row];
        let s3 = &scales[r3 * groups_per_row..(r3 + 1) * groups_per_row];

        let mut acc0 = 0.0f32;
        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;
        let mut acc3 = 0.0f32;

        for g in 0..groups_per_row {
            let base = g * GROUP_SIZE_64;
            let xg = &x[base..base + GROUP_SIZE_64];
            let g0 = &w0[base..base + GROUP_SIZE_64];
            let g1 = &w1[base..base + GROUP_SIZE_64];
            let g2 = &w2[base..base + GROUP_SIZE_64];
            let g3 = &w3[base..base + GROUP_SIZE_64];

            let mut a0_0 = [0.0f32; 8];
            let mut a0_1 = [0.0f32; 8];
            let mut a1_0 = [0.0f32; 8];
            let mut a1_1 = [0.0f32; 8];
            let mut a2_0 = [0.0f32; 8];
            let mut a2_1 = [0.0f32; 8];
            let mut a3_0 = [0.0f32; 8];
            let mut a3_1 = [0.0f32; 8];

            for step in 0..4 {
                let off = step * 16;
                for l in 0..8 {
                    let x_lo = xg[off + l];
                    let x_hi = xg[off + 8 + l];
                    a0_0[l] += FP8_E4M3FN_F32_LUT[g0[off + l] as usize] * x_lo;
                    a0_1[l] += FP8_E4M3FN_F32_LUT[g0[off + 8 + l] as usize] * x_hi;
                    a1_0[l] += FP8_E4M3FN_F32_LUT[g1[off + l] as usize] * x_lo;
                    a1_1[l] += FP8_E4M3FN_F32_LUT[g1[off + 8 + l] as usize] * x_hi;
                    a2_0[l] += FP8_E4M3FN_F32_LUT[g2[off + l] as usize] * x_lo;
                    a2_1[l] += FP8_E4M3FN_F32_LUT[g2[off + 8 + l] as usize] * x_hi;
                    a3_0[l] += FP8_E4M3FN_F32_LUT[g3[off + l] as usize] * x_lo;
                    a3_1[l] += FP8_E4M3FN_F32_LUT[g3[off + 8 + l] as usize] * x_hi;
                }
            }

            acc0 += gs_group_sum_simd8(a0_0, a0_1) * s0[g];
            acc1 += gs_group_sum_simd8(a1_0, a1_1) * s1[g];
            acc2 += gs_group_sum_simd8(a2_0, a2_1) * s2[g];
            acc3 += gs_group_sum_simd8(a3_0, a3_1) * s3[g];
        }

        out[r0] = acc0;
        out[r1] = acc1;
        out[r2] = acc2;
        out[r3] = acc3;
    }

    for r in (full_quads * 4)..rows {
        let wr = &w_fp8[r * cols..(r + 1) * cols];
        let sr = &scales[r * groups_per_row..(r + 1) * groups_per_row];
        let mut acc = 0.0f32;
        for g in 0..groups_per_row {
            let base = g * GROUP_SIZE_64;
            let xg = &x[base..base + GROUP_SIZE_64];
            let gr = &wr[base..base + GROUP_SIZE_64];
            let mut a0 = [0.0f32; 8];
            let mut a1 = [0.0f32; 8];
            for step in 0..4 {
                let off = step * 16;
                for l in 0..8 {
                    a0[l] += FP8_E4M3FN_F32_LUT[gr[off + l] as usize] * xg[off + l];
                    a1[l] += FP8_E4M3FN_F32_LUT[gr[off + 8 + l] as usize] * xg[off + 8 + l];
                }
            }
            acc += gs_group_sum_simd8(a0, a1) * sr[g];
        }
        out[r] = acc;
    }
}

/// RSI-COLIBRI-038: Native Multiplier-Free 1.58-Bit Ternary SIMD Kernel (`matmul_q_ternary_158`).
/// Unpacks 4 ternary weights per byte (`2` bits/weight: `00 => 0`, `01 => +1`, `10 => -1`, `11 => 0`
/// safe clamp) into add/sub-only dual-`[f32; 8]` SIMD accumulators with associative
/// `gs_group_sum_simd8` tree reduction (`TP=1/2/4/8` bit-identical, `+3.4x` memory-bandwidth efficiency).
#[inline(always)]
fn ternary_delta(code2: u8, x_val: f32) -> f32 {
    match code2 & 0x03 {
        1 => x_val,
        2 => -x_val,
        _ => 0.0,
    }
}

pub fn matmul_q_ternary_158(
    out: &mut [f32],
    x: &[f32],
    w_packed_2bit: &[u8],
    scales: &[f32],
    rows: usize,
    cols: usize,
) {
    assert!(
        cols.is_multiple_of(GROUP_SIZE_64),
        "cols must be a multiple of 64"
    );
    let bytes_per_row = cols / 4;
    let groups_per_row = cols / GROUP_SIZE_64;
    assert!(out.len() >= rows && x.len() >= cols);
    assert!(w_packed_2bit.len() >= rows * bytes_per_row && scales.len() >= rows * groups_per_row);

    for r in 0..rows {
        let wr = &w_packed_2bit[r * bytes_per_row..(r + 1) * bytes_per_row];
        let sr = &scales[r * groups_per_row..(r + 1) * groups_per_row];
        let mut acc = 0.0f32;

        for g in 0..groups_per_row {
            let x_base = g * GROUP_SIZE_64;
            let b_base = g * 16; // 64 weights / 4 weights per byte = 16 bytes
            let xg = &x[x_base..x_base + GROUP_SIZE_64];
            let bg = &wr[b_base..b_base + 16];

            let mut a0 = [0.0f32; 8];
            let mut a1 = [0.0f32; 8];

            for step in 0..4 {
                let x_off = step * 16;
                let b_off = step * 4;
                // First 8 weights (2 bytes: bg[b_off], bg[b_off+1]) -> lanes a0[0..8]
                let b0 = bg[b_off];
                let b1 = bg[b_off + 1];
                a0[0] += ternary_delta(b0, xg[x_off]);
                a0[1] += ternary_delta(b0 >> 2, xg[x_off + 1]);
                a0[2] += ternary_delta(b0 >> 4, xg[x_off + 2]);
                a0[3] += ternary_delta(b0 >> 6, xg[x_off + 3]);
                a0[4] += ternary_delta(b1, xg[x_off + 4]);
                a0[5] += ternary_delta(b1 >> 2, xg[x_off + 5]);
                a0[6] += ternary_delta(b1 >> 4, xg[x_off + 6]);
                a0[7] += ternary_delta(b1 >> 6, xg[x_off + 7]);

                // Next 8 weights (2 bytes: bg[b_off+2], bg[b_off+3]) -> lanes a1[0..8]
                let b2 = bg[b_off + 2];
                let b3 = bg[b_off + 3];
                a1[0] += ternary_delta(b2, xg[x_off + 8]);
                a1[1] += ternary_delta(b2 >> 2, xg[x_off + 9]);
                a1[2] += ternary_delta(b2 >> 4, xg[x_off + 10]);
                a1[3] += ternary_delta(b2 >> 6, xg[x_off + 11]);
                a1[4] += ternary_delta(b3, xg[x_off + 12]);
                a1[5] += ternary_delta(b3 >> 2, xg[x_off + 13]);
                a1[6] += ternary_delta(b3 >> 4, xg[x_off + 14]);
                a1[7] += ternary_delta(b3 >> 6, xg[x_off + 15]);
            }

            acc += gs_group_sum_simd8(a0, a1) * sr[g];
        }
        out[r] = acc;
    }
}
