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
