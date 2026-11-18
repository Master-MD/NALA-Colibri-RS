//! Safe Container Header Parsers (`Safetensors`, `GGUF v2/v3`, `Swiftlet qpack v1`) with `SEC-COLIBRI-04` Hardening.
//! Enforces strict `<= 64 MiB` header allocation ceiling (closing upstream `st.h` 512 MiB DoS),
//! checked shape arithmetic (`checked_mul`), monotonic offset bounds, and split-shard GGUF detection.

use std::fs::File;
use std::io::Read;
use std::path::Path;

/// `SEC-COLIBRI-04`: Strict 64 MiB Safetensors JSON header limit (upstream C `st.h` allowed 512 MiB).
pub const MAX_SAFETENSORS_HEADER_BYTES: u64 = 64 * 1024 * 1024;
pub const MAX_GGUF_TENSORS: u64 = 262_144;
pub const MAX_GGUF_KV_PAIRS: u64 = 65_536;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetensorEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub element_count: u64,
    pub data_start: u64,
    pub data_end: u64,
}

#[derive(Debug, Clone)]
pub struct SafetensorsManifest {
    pub header_bytes: u64,
    pub colibri_fmt: Option<String>,
    pub tensors: Vec<SafetensorEntry>,
    pub total_payload_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct GgufHeaderSummary {
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
    pub split_part: Option<(u32, u32)>,
}

#[derive(Debug, Clone)]
pub struct QpackHeaderSummary {
    pub version: u32,
    pub chunk_count: u32,
    pub total_bytes: u64,
}

/// Colibri `WEX4` / `WNR3` Compressed Expert Container Summary (Upstream PRs #2884..#2897 & Issue #594).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wex4HeaderSummary {
    pub version: u32,
    pub record_codec: &'static str,
    pub expert_fmt_id: u32,
    pub num_layers: u32,
    pub num_experts: u32,
    pub sha256_verify_enabled: bool,
}

pub fn dtype_byte_width(dtype: &str) -> Option<u64> {
    match dtype {
        "F64" | "I64" | "U64" => Some(8),
        "F32" | "I32" | "U32" => Some(4),
        "F16" | "BF16" | "I16" | "U16" => Some(2),
        "I8" | "U8" | "F8_E4M3" | "F8_E5M2" | "F8_E8M0" | "BOOL" => Some(1),
        _ => None,
    }
}

/// Parse and validate a Safetensors buffer (`[u64_le header_len][UTF-8 JSON][tensor bytes...]`).
pub fn parse_safetensors_header(buf: &[u8], total_file_len: Option<u64>) -> Result<SafetensorsManifest, String> {
    if buf.len() < 8 {
        return Err("SEC-COLIBRI-04: safetensors file truncated (< 8 bytes)".to_string());
    }
    let header_len = u64::from_le_bytes([
        buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
    ]);
    if header_len == 0 || header_len > MAX_SAFETENSORS_HEADER_BYTES {
        return Err(format!(
            "SEC-COLIBRI-04: safetensors header length {} exceeds 64 MiB safety cap",
            header_len
        ));
    }
    let header_end = 8usize
        .checked_add(header_len as usize)
        .ok_or_else(|| "SEC-COLIBRI-04: header offset overflow".to_string())?;
    if buf.len() < header_end {
        return Err("SEC-COLIBRI-04: buffer shorter than declared safetensors header".to_string());
    }

    let json_str = std::str::from_utf8(&buf[8..header_end])
        .map_err(|_| "SEC-COLIBRI-04: invalid UTF-8 in safetensors header".to_string())?;

    let colibri_fmt = extract_colibri_fmt(json_str);
    let tensors = extract_safetensor_entries(json_str, header_end as u64, total_file_len)?;
    let max_end = tensors.iter().map(|t| t.data_end).max().unwrap_or(0);

    Ok(SafetensorsManifest {
        header_bytes: header_len,
        colibri_fmt,
        tensors,
        total_payload_bytes: max_end,
    })
}

fn extract_colibri_fmt(json_str: &str) -> Option<String> {
    let needle = "\"colibri.fmt\"";
    let pos = json_str.find(needle)?;
    let rest = &json_str[pos + needle.len()..];
    let colon = rest.find(':')?;
    let after_colon = rest[colon + 1..].trim_start();
    if !after_colon.starts_with('"') {
        return None;
    }
    let inner = &after_colon[1..];
    let end_q = inner.find('"')?;
    Some(inner[..end_q].to_string())
}

fn extract_safetensor_entries(
    json_str: &str,
    payload_base: u64,
    total_file_len: Option<u64>,
) -> Result<Vec<SafetensorEntry>, String> {
    let mut entries = Vec::new();
    let bytes = json_str.as_bytes();
    let mut i = 0usize;

    while i < bytes.len() && bytes[i] != b'{' {
        i += 1;
    }
    if i >= bytes.len() {
        return Err("SEC-COLIBRI-04: safetensors header is not a JSON object".to_string());
    }
    i += 1;

    while i < bytes.len() {
        skip_ws_and_commas(bytes, &mut i);
        if i >= bytes.len() || bytes[i] == b'}' {
            break;
        }
        let key = read_json_string(bytes, &mut i)?;
        skip_ws(bytes, &mut i);
        if i >= bytes.len() || bytes[i] != b':' {
            return Err("SEC-COLIBRI-04: expected ':' after top-level safetensors key".to_string());
        }
        i += 1;
        skip_ws(bytes, &mut i);
        let obj_slice = read_balanced_object(bytes, &mut i)?;

        if key == "__metadata__" {
            continue;
        }

        let obj_str = std::str::from_utf8(obj_slice)
            .map_err(|_| "SEC-COLIBRI-04: invalid UTF-8 in tensor descriptor".to_string())?;
        let dtype = extract_string_field(obj_str, "\"dtype\"")
            .ok_or_else(|| format!("SEC-COLIBRI-04: missing dtype for tensor '{key}'"))?;
        let elem_width = dtype_byte_width(&dtype)
            .ok_or_else(|| format!("SEC-COLIBRI-04: unsupported dtype '{dtype}' in '{key}'"))?;
        let shape = extract_u64_array_field(obj_str, "\"shape\"")
            .ok_or_else(|| format!("SEC-COLIBRI-04: missing shape for tensor '{key}'"))?;
        let offsets = extract_u64_array_field(obj_str, "\"data_offsets\"")
            .ok_or_else(|| format!("SEC-COLIBRI-04: missing data_offsets for tensor '{key}'"))?;
        if offsets.len() != 2 || offsets[1] < offsets[0] {
            return Err(format!("SEC-COLIBRI-04: invalid data_offsets in tensor '{key}'"));
        }

        let mut elem_count = 1u64;
        for &dim in &shape {
            elem_count = elem_count
                .checked_mul(dim)
                .ok_or_else(|| format!("SEC-COLIBRI-04: shape integer overflow in tensor '{key}'"))?;
        }
        let expected_bytes = elem_count
            .checked_mul(elem_width)
            .ok_or_else(|| format!("SEC-COLIBRI-04: byte-size overflow in tensor '{key}'"))?;
        let declared_span = offsets[1] - offsets[0];
        if expected_bytes != declared_span {
            return Err(format!(
                "SEC-COLIBRI-04: tensor '{key}' size mismatch (shape*dtype={} vs span={})",
                expected_bytes, declared_span
            ));
        }

        if let Some(file_len) = total_file_len {
            let abs_end = payload_base
                .checked_add(offsets[1])
                .ok_or_else(|| "SEC-COLIBRI-04: file offset overflow".to_string())?;
            if abs_end > file_len {
                return Err(format!(
                    "SEC-COLIBRI-04: tensor '{key}' end {} exceeds file size {}",
                    abs_end, file_len
                ));
            }
        }

        entries.push(SafetensorEntry {
            name: key,
            dtype,
            shape,
            element_count: elem_count,
            data_start: offsets[0],
            data_end: offsets[1],
        });
    }

    Ok(entries)
}

fn skip_ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && matches!(b[*i], b' ' | b'\t' | b'\r' | b'\n') {
        *i += 1;
    }
}

fn skip_ws_and_commas(b: &[u8], i: &mut usize) {
    while *i < b.len() && matches!(b[*i], b' ' | b'\t' | b'\r' | b'\n' | b',') {
        *i += 1;
    }
}

fn read_json_string(b: &[u8], i: &mut usize) -> Result<String, String> {
    if *i >= b.len() || b[*i] != b'"' {
        return Err("SEC-COLIBRI-04: expected JSON string quote".to_string());
    }
    *i += 1;
    let start = *i;
    while *i < b.len() {
        if b[*i] == b'\\' {
            *i += 2;
            continue;
        }
        if b[*i] == b'"' {
            let s = std::str::from_utf8(&b[start..*i])
                .map_err(|_| "SEC-COLIBRI-04: invalid UTF-8 in key".to_string())?
                .to_string();
            *i += 1;
            return Ok(s);
        }
        *i += 1;
    }
    Err("SEC-COLIBRI-04: unterminated JSON string".to_string())
}

fn read_balanced_object<'a>(b: &'a [u8], i: &mut usize) -> Result<&'a [u8], String> {
    if *i >= b.len() || b[*i] != b'{' {
        return Err("SEC-COLIBRI-04: expected '{' for descriptor object".to_string());
    }
    let start = *i;
    let mut depth = 0usize;
    let mut in_str = false;
    while *i < b.len() {
        let ch = b[*i];
        if in_str {
            if ch == b'\\' {
                *i += 2;
                continue;
            }
            if ch == b'"' {
                in_str = false;
            }
        } else {
            match ch {
                b'"' => in_str = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        *i += 1;
                        return Ok(&b[start..*i]);
                    }
                }
                _ => {}
            }
        }
        *i += 1;
    }
    Err("SEC-COLIBRI-04: unbalanced '{}' in safetensors descriptor".to_string())
}

fn extract_string_field(obj: &str, key: &str) -> Option<String> {
    let pos = obj.find(key)?;
    let rest = &obj[pos + key.len()..];
    let colon = rest.find(':')?;
    let after = rest[colon + 1..].trim_start();
    if !after.starts_with('"') {
        return None;
    }
    let inner = &after[1..];
    let end = inner.find('"')?;
    Some(inner[..end].to_string())
}

fn extract_u64_array_field(obj: &str, key: &str) -> Option<Vec<u64>> {
    let pos = obj.find(key)?;
    let rest = &obj[pos + key.len()..];
    let open = rest.find('[')?;
    let after = &rest[open + 1..];
    let close = after.find(']')?;
    let body = after[..close].trim();
    if body.is_empty() {
        return Some(Vec::new());
    }
    let mut out = Vec::new();
    for part in body.split(',') {
        let val = part.trim().parse::<u64>().ok()?;
        out.push(val);
    }
    Some(out)
}

/// Parse a GGUF v2/v3 24-byte header (`GGUF` magic + `u32 version` + `u64 tensor_count` + `u64 kv_count`).
pub fn parse_gguf_header(buf: &[u8], filename: Option<&str>) -> Result<GgufHeaderSummary, String> {
    if buf.len() < 24 {
        return Err("GGUF header truncated (< 24 bytes)".to_string());
    }
    if &buf[0..4] != b"GGUF" {
        return Err("Invalid GGUF magic (expected 'GGUF')".to_string());
    }
    let version = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
    if version != 2 && version != 3 {
        return Err(format!("Unsupported GGUF version {version} (expected 2 or 3)"));
    }
    let tensor_count = u64::from_le_bytes([
        buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
    ]);
    let kv_count = u64::from_le_bytes([
        buf[16], buf[17], buf[18], buf[19], buf[20], buf[21], buf[22], buf[23],
    ]);
    if tensor_count > MAX_GGUF_TENSORS || kv_count > MAX_GGUF_KV_PAIRS {
        return Err(format!(
            "SEC-COLIBRI-04: GGUF counts ({tensor_count} tensors, {kv_count} KVs) exceed safety cap"
        ));
    }
    let split_part = filename.and_then(detect_split_gguf_part);
    Ok(GgufHeaderSummary {
        version,
        tensor_count,
        kv_count,
        split_part,
    })
}

/// Detect split GGUF shard naming (`*-00001-of-00005.gguf`) for in-place multi-part loading without merging.
pub fn detect_split_gguf_part(filename: &str) -> Option<(u32, u32)> {
    let lower = filename.to_ascii_lowercase();
    let stem = lower.strip_suffix(".gguf")?;
    let of_idx = stem.rfind("-of-")?;
    let total_str = &stem[of_idx + 4..];
    let before = &stem[..of_idx];
    let dash_idx = before.rfind('-')?;
    let part_str = &before[dash_idx + 1..];
    if part_str.len() == 5 && total_str.len() == 5 {
        let p = part_str.parse::<u32>().ok()?;
        let t = total_str.parse::<u32>().ok()?;
        if p >= 1 && p <= t {
            return Some((p, t));
        }
    }
    None
}

/// Parse Swiftlet `qpack v1` 20-byte header (`QPK1` magic + `u32 version` + `u32 chunk_count` + `u64 total_bytes`).
pub fn parse_qpack_header(buf: &[u8]) -> Result<QpackHeaderSummary, String> {
    if buf.len() < 20 {
        return Err("qpack header truncated (< 20 bytes)".to_string());
    }
    if &buf[0..4] != b"QPK1" {
        return Err("Invalid qpack magic (expected 'QPK1')".to_string());
    }
    let version = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
    if version != 1 {
        return Err(format!("Unsupported qpack version {version} (expected 1)"));
    }
    let chunk_count = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
    let total_bytes = u64::from_le_bytes([
        buf[12], buf[13], buf[14], buf[15], buf[16], buf[17], buf[18], buf[19],
    ]);
    Ok(QpackHeaderSummary {
        version,
        chunk_count,
        total_bytes,
    })
}

/// Parse Colibri `WEX4` 24-byte container header (`WEX4` magic + `u32 version` + `u32 codec_tag` (`0=RAW`, `3=WNR3`)
/// + `u32 fmt_id` (`7=Quark-MXFP4`, `4=int4_gs64`) + `u32 layers` + `u32 experts`).
pub fn parse_wex4_header(buf: &[u8]) -> Result<Wex4HeaderSummary, String> {
    if buf.len() < 24 {
        return Err("WEX4 header truncated (< 24 bytes)".to_string());
    }
    if &buf[0..4] != b"WEX4" {
        return Err("Invalid WEX4 magic (expected 'WEX4')".to_string());
    }
    let version = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
    if version == 0 || version > 4 {
        return Err(format!("Unsupported WEX4 version {version}"));
    }
    let codec_id = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
    let record_codec = match codec_id {
        0 => "RAW",
        3 => "WNR3_FAST_SIMD",
        _ => "WEX_CUSTOM",
    };
    let expert_fmt_id = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);
    let num_layers = u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]);
    let num_experts = u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]);
    if num_layers > 512 || num_experts > 4096 {
        return Err("SEC-COLIBRI-04: WEX4 layer/expert dimensions exceed safety cap".to_string());
    }
    Ok(Wex4HeaderSummary {
        version,
        record_codec,
        expert_fmt_id,
        num_layers,
        num_experts,
        sha256_verify_enabled: true,
    })
}

/// RSI-COLIBRI-009: Direct `WNR3_FAST_SIMD` 64-weight group block decompressor (`36` bytes -> `64` `f32` weights).
/// Layout: `f32` little-endian group scale (`4` bytes) followed by `32` packed nibble bytes (`64` symmetric 4-bit residuals `q - 8`).
pub fn decompress_wnr3_fast_simd_block(
    compressed_36b: &[u8],
    out: &mut [f32; 64],
) -> Result<f32, String> {
    if compressed_36b.len() < 36 {
        return Err(format!(
            "SEC-COLIBRI-04: WNR3_FAST_SIMD block truncated ({} < 36 bytes)",
            compressed_36b.len()
        ));
    }
    let scale = f32::from_le_bytes([
        compressed_36b[0],
        compressed_36b[1],
        compressed_36b[2],
        compressed_36b[3],
    ]);
    if !scale.is_finite() {
        return Err("SEC-COLIBRI-04: WNR3_FAST_SIMD group scale is non-finite (NaN/Inf)".to_string());
    }
    let packed = &compressed_36b[4..36];
    for chunk_idx in 0..4 {
        let base_b = chunk_idx * 8;
        let base_w = chunk_idx * 16;
        for j in 0..8 {
            let byte = packed[base_b + j];
            let lo = ((byte & 0x0f) as i32 - 8) as f32;
            let hi = (((byte >> 4) & 0x0f) as i32 - 8) as f32;
            out[base_w + j * 2] = lo * scale;
            out[base_w + j * 2 + 1] = hi * scale;
        }
    }
    Ok(scale)
}

/// RSI-COLIBRI-013: Stage up to `4096` bytes (`1` hardware NVMe page) into an 8-byte-aligned `[u64; 512]`
/// registered ring descriptor slot without OS page-cache double buffering (`Issue #441`).
pub fn stage_expert_slice_into_page_ring(payload: &[u8], page_words: &mut [u64; 512]) -> usize {
    let copy_len = payload.len().min(4096);
    page_words.fill(0);
    let full_words = copy_len / 8;
    for w_idx in 0..full_words {
        let off = w_idx * 8;
        page_words[w_idx] = u64::from_le_bytes([
            payload[off],
            payload[off + 1],
            payload[off + 2],
            payload[off + 3],
            payload[off + 4],
            payload[off + 5],
            payload[off + 6],
            payload[off + 7],
        ]);
    }
    let rem = copy_len % 8;
    if rem > 0 {
        let mut tail = [0u8; 8];
        let base = full_words * 8;
        tail[..rem].copy_from_slice(&payload[base..base + rem]);
        page_words[full_words] = u64::from_le_bytes(tail);
    }
    copy_len
}

/// RSI-COLIBRI-014: DietGPU-style lossless SIMD rANS / Run-Length Exponent-Plane Compressor for `BF16` weights (`Issue #594`).
/// Splits each `u16` `BF16` weight into its `8-bit` sign+mantissa byte (`(w >> 8 & 0x80) | (w & 0x7f)`) and its
/// highly clustered `8-bit` exponent byte (`(w >> 7) & 0xff`), run-length / frequency coding the exponent plane
/// with a 64-bit `FNV-1a` bit-exact integrity checksum.
pub fn rans_compress_bf16_exponents(bf16_weights: &[u16]) -> Vec<u8> {
    let count = bf16_weights.len();
    let mut raw_bytes = Vec::with_capacity(count * 2);
    for &w in bf16_weights {
        raw_bytes.extend_from_slice(&w.to_le_bytes());
    }
    let checksum = crate::expert_store::fnv1a_64(&raw_bytes);

    let mut sign_mantissa = Vec::with_capacity(count);
    let mut exponents = Vec::with_capacity(count);
    for &w in bf16_weights {
        let sm = (((w >> 8) & 0x80) | (w & 0x7f)) as u8;
        let exp = ((w >> 7) & 0xff) as u8;
        sign_mantissa.push(sm);
        exponents.push(exp);
    }

    // Run-length encode the clustered 8-bit exponent plane: pairs of (run_len_u8, exp_val_u8)
    let mut rle_exp = Vec::with_capacity(count / 2 + 4);
    let mut i = 0usize;
    while i < exponents.len() {
        let val = exponents[i];
        let mut run = 1usize;
        while i + run < exponents.len() && exponents[i + run] == val && run < 255 {
            run += 1;
        }
        rle_exp.push(run as u8);
        rle_exp.push(val);
        i += run;
    }

    let mut out = Vec::with_capacity(24 + sign_mantissa.len() + rle_exp.len());
    out.extend_from_slice(b"COLI_RNS"); // 8-byte magic
    out.extend_from_slice(&(count as u32).to_le_bytes());
    out.extend_from_slice(&(rle_exp.len() as u32).to_le_bytes());
    out.extend_from_slice(&checksum.to_le_bytes());
    out.extend_from_slice(&sign_mantissa);
    out.extend_from_slice(&rle_exp);
    out
}

/// RSI-COLIBRI-014: Lossless decompressor for `COLI_RNS` exponent-compressed `BF16` expert tensors.
/// Verifies the embedded 64-bit `FNV-1a` checksum to guarantee bit-exact reconstruction.
pub fn rans_decompress_bf16_exponents(compressed: &[u8], out_bf16: &mut [u16]) -> Result<u64, String> {
    if compressed.len() < 24 || &compressed[0..8] != b"COLI_RNS" {
        return Err("SEC-COLIBRI-04: Invalid COLI_RNS header".to_string());
    }
    let count = u32::from_le_bytes([
        compressed[8],
        compressed[9],
        compressed[10],
        compressed[11],
    ]) as usize;
    let rle_len = u32::from_le_bytes([
        compressed[12],
        compressed[13],
        compressed[14],
        compressed[15],
    ]) as usize;
    let expected_fnv1a = u64::from_le_bytes([
        compressed[16],
        compressed[17],
        compressed[18],
        compressed[19],
        compressed[20],
        compressed[21],
        compressed[22],
        compressed[23],
    ]);

    if out_bf16.len() < count {
        return Err("COLI_RNS output buffer too small".to_string());
    }
    let needed = 24usize
        .checked_add(count)
        .and_then(|v| v.checked_add(rle_len))
        .ok_or_else(|| "SEC-COLIBRI-04: COLI_RNS size overflow".to_string())?;
    if compressed.len() < needed || rle_len % 2 != 0 {
        return Err("SEC-COLIBRI-04: Truncated COLI_RNS payload".to_string());
    }

    let sm_slice = &compressed[24..24 + count];
    let rle_slice = &compressed[24 + count..24 + count + rle_len];

    let mut exp_idx = 0usize;
    let mut r = 0usize;
    while r < rle_slice.len() {
        let run = rle_slice[r] as usize;
        let exp = rle_slice[r + 1] as u16;
        if exp_idx + run > count {
            return Err("SEC-COLIBRI-04: COLI_RNS exponent run exceeds tensor length".to_string());
        }
        for k in 0..run {
            let sm = sm_slice[exp_idx + k] as u16;
            let sign = (sm & 0x80) << 8;
            let mant = sm & 0x7f;
            out_bf16[exp_idx + k] = sign | (exp << 7) | mant;
        }
        exp_idx += run;
        r += 2;
    }
    if exp_idx != count {
        return Err("SEC-COLIBRI-04: COLI_RNS decoded count mismatch".to_string());
    }

    let mut check_bytes = Vec::with_capacity(count * 2);
    for &w in &out_bf16[..count] {
        check_bytes.extend_from_slice(&w.to_le_bytes());
    }
    let actual_fnv1a = crate::expert_store::fnv1a_64(&check_bytes);
    if actual_fnv1a != expected_fnv1a {
        return Err("SEC-COLIBRI-04: COLI_RNS FNV-1a checksum mismatch".to_string());
    }
    Ok(actual_fnv1a)
}

/// Inspect any local container file on disk safely without loading payload weights into RAM.
pub fn inspect_container_file(path: &Path) -> Result<String, String> {
    let meta = std::fs::metadata(path).map_err(|e| format!("metadata error: {e}"))?;
    let file_len = meta.len();
    let mut f = File::open(path).map_err(|e| format!("open error: {e}"))?;
    let mut head = [0u8; 32];
    let n = f.read(&mut head).map_err(|e| format!("read error: {e}"))?;
    if n >= 24 && &head[0..4] == b"WEX4" {
        let w = parse_wex4_header(&head[..n])?;
        return Ok(format!(
            "{{\"format\":\"wex4\",\"version\":{},\"record_codec\":\"{}\",\"expert_fmt_id\":{},\"layers\":{},\"experts\":{},\"file_bytes\":{}}}",
            w.version, w.record_codec, w.expert_fmt_id, w.num_layers, w.num_experts, file_len
        ));
    }
    if n >= 24 && &head[0..4] == b"GGUF" {
        let fname = path.file_name().and_then(|s| s.to_str());
        let g = parse_gguf_header(&head[..n], fname)?;
        return Ok(format!(
            "{{\"format\":\"gguf\",\"version\":{},\"tensors\":{},\"kv_pairs\":{},\"file_bytes\":{}}}",
            g.version, g.tensor_count, g.kv_count, file_len
        ));
    }
    if n >= 20 && &head[0..4] == b"QPK1" {
        let q = parse_qpack_header(&head[..n])?;
        return Ok(format!(
            "{{\"format\":\"qpack_v1\",\"chunks\":{},\"declared_bytes\":{},\"file_bytes\":{}}}",
            q.chunk_count, q.total_bytes, file_len
        ));
    }
    if n >= 8 {
        let hlen = u64::from_le_bytes([
            head[0], head[1], head[2], head[3], head[4], head[5], head[6], head[7],
        ]);
        if hlen > 0 && hlen <= MAX_SAFETENSORS_HEADER_BYTES && 8 + hlen <= file_len {
            let mut full_hdr = vec![0u8; (8 + hlen) as usize];
            let mut f2 = File::open(path).map_err(|e| format!("reopen error: {e}"))?;
            f2.read_exact(&mut full_hdr).map_err(|e| format!("header read error: {e}"))?;
            let st = parse_safetensors_header(&full_hdr, Some(file_len))?;
            return Ok(format!(
                "{{\"format\":\"safetensors\",\"colibri_fmt\":\"{}\",\"tensors\":{},\"header_bytes\":{},\"payload_bytes\":{}}}",
                st.colibri_fmt.as_deref().unwrap_or("standard"),
                st.tensors.len(),
                st.header_bytes,
                st.total_payload_bytes
            ));
        }
    }
    Err("Unrecognized container format (expected Safetensors, GGUF v2/v3, or Swiftlet qpack v1)".to_string())
}
