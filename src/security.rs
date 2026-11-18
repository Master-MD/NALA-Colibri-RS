//! `sHeriFF + NVIDIA SkillSpector` Defensive Security Hardening (`SEC-COLIBRI-01..04` + `SEC-05`).
//! 100% Pure Safe Rust (`#![forbid(unsafe_code)]`) — includes FIPS-180-4 SHA-256, RFC 2104 HMAC-SHA256
//! with constant-time MAC verification, IP-pinned SSRF resolution (zero DNS-rebinding TOCTOU),
//! 10 MiB HTTP 413 body cap, loopback Origin CSRF guard, and canonical symlink-escape protection.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, ToSocketAddrs};
use std::path::{Component, Path, PathBuf};

/// `SEC-COLIBRI-01` / `TheStandart SEC-04`: Strict 10 MiB request body ceiling (`HTTP 413 Payload Too Large`).
pub const MAX_HTTP_BODY_BYTES: usize = 10 * 1024 * 1024;

pub const HTTP_SECURITY_HEADERS: &str = concat!(
    "X-Content-Type-Options: nosniff\r\n",
    "X-Frame-Options: DENY\r\n",
    "Referrer-Policy: no-referrer\r\n",
    "Cross-Origin-Resource-Policy: same-origin\r\n",
    "Content-Security-Policy: default-src 'self' 'unsafe-inline' data:; frame-ancestors 'none'; base-uri 'self'\r\n"
);

/// Pure-Rust streaming FIPS 180-4 SHA-256 over any `Read` stream (`64 KiB` constant memory, handles 100GB+ shards).
pub fn sha256_reader<R: std::io::Read>(mut reader: R) -> Result<[u8; 32], String> {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    let compress = |h: &mut [u32; 8], chunk: &[u8]| {
        let mut w = [0u32; 64];
        for i in 0..16 {
            let j = i * 4;
            w[i] = u32::from_be_bytes([chunk[j], chunk[j + 1], chunk[j + 2], chunk[j + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            hh = g; g = f; f = e; e = d.wrapping_add(temp1); d = c; c = b; b = a; a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a); h[1] = h[1].wrapping_add(b); h[2] = h[2].wrapping_add(c); h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e); h[5] = h[5].wrapping_add(f); h[6] = h[6].wrapping_add(g); h[7] = h[7].wrapping_add(hh);
    };

    let mut total_bytes: u64 = 0;
    let mut tail = Vec::with_capacity(128);
    let mut buf = vec![0u8; 65_536];
    loop {
        let n = reader.read(&mut buf).map_err(|e| format!("read error: {e}"))?;
        if n == 0 {
            break;
        }
        total_bytes = total_bytes.wrapping_add(n as u64);
        tail.extend_from_slice(&buf[..n]);
        let full_len = (tail.len() / 64) * 64;
        for chunk in tail[..full_len].chunks_exact(64) {
            compress(&mut h, chunk);
        }
        tail.drain(..full_len);
    }

    tail.push(0x80);
    while (tail.len() % 64) != 56 {
        tail.push(0x00);
    }
    tail.extend_from_slice(&(total_bytes.wrapping_mul(8)).to_be_bytes());
    for chunk in tail.chunks_exact(64) {
        compress(&mut h, chunk);
    }

    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    Ok(out)
}

pub fn sha256_file(path: &Path) -> Result<String, String> {
    let f = std::fs::File::open(path).map_err(|e| format!("cannot open '{}': {e}", path.display()))?;
    let digest = sha256_reader(f)?;
    Ok(hex_encode(&digest))
}

/// Pure-Rust FIPS 180-4 SHA-256 digest (`[u8; 32]`).
pub fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    let bit_len = (data.len() as u64) * 8;
    let mut padded = Vec::with_capacity(data.len() + 72);
    padded.extend_from_slice(data);
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0x00);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            let j = i * 4;
            w[i] = u32::from_be_bytes([chunk[j], chunk[j + 1], chunk[j + 2], chunk[j + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        let bytes = word.to_be_bytes();
        out[i * 4..i * 4 + 4].copy_from_slice(&bytes);
    }
    out
}

/// RFC 2104 HMAC-SHA256 (`[u8; 32]`).
pub fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    let mut k_block = [0u8; 64];
    if key.len() > 64 {
        let kh = sha256(key);
        k_block[..32].copy_from_slice(&kh);
    } else {
        k_block[..key.len()].copy_from_slice(key);
    }

    let mut ipad = [0x36u8; 64];
    let mut opad = [0x5cu8; 64];
    for i in 0..64 {
        ipad[i] ^= k_block[i];
        opad[i] ^= k_block[i];
    }

    let mut inner = Vec::with_capacity(64 + message.len());
    inner.extend_from_slice(&ipad);
    inner.extend_from_slice(message);
    let inner_hash = sha256(&inner);

    let mut outer = [0u8; 96];
    outer[..64].copy_from_slice(&opad);
    outer[64..].copy_from_slice(&inner_hash);
    sha256(&outer)
}

pub fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

/// `SEC-COLIBRI-03`: Constant-time hex HMAC-SHA256 verification (`0` timing side-channel).
pub fn verify_hmac_sha256_hex(secret: &[u8], payload: &[u8], provided_hex: &str) -> bool {
    let expected_hex = hex_encode(&hmac_sha256(secret, payload));
    let a = expected_hex.as_bytes();
    let b = provided_hex.trim().as_bytes();
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (&x, &y) in a.iter().zip(b.iter()) {
        diff |= x.to_ascii_lowercase() ^ y.to_ascii_lowercase();
    }
    diff == 0
}

/// `SEC-COLIBRI-02` / `TheStandart SEC-01`: Strictly classify whether an IP is public global unicast.
/// Blocks loopback, RFC1918 private LAN, link-local / AWS-GCP metadata (`169.254.169.254`),
/// CGNAT/Tailscale (`100.64.0.0/10`), multicast, broadcast, unspecified, IPv6 ULA (`fc00::/7`), and IPv4-mapped IPv6.
pub fn is_strictly_public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_strictly_public_ipv4(v4),
        IpAddr::V6(v6) => {
            if let Some(mapped_v4) = v6.to_ipv4_mapped() {
                return is_strictly_public_ipv4(mapped_v4);
            }
            if v6.is_loopback() || v6.is_unspecified() || v6.is_multicast() {
                return false;
            }
            let seg0 = v6.segments()[0];
            // Block ULA fc00::/7, link-local fe80::/10, documentation 2001:db8::/32
            if (seg0 & 0xfe00) == 0xfc00 || (seg0 & 0xffc0) == 0xfe80 {
                return false;
            }
            if seg0 == 0x2001 && v6.segments()[1] == 0x0db8 {
                return false;
            }
            v6 != Ipv6Addr::LOCALHOST
        }
    }
}

fn is_strictly_public_ipv4(v4: Ipv4Addr) -> bool {
    let o = v4.octets();
    if v4.is_loopback()
        || v4.is_private()
        || v4.is_link_local()
        || v4.is_broadcast()
        || v4.is_multicast()
        || v4.is_unspecified()
        || v4.is_documentation()
    {
        return false;
    }
    // 0.0.0.0/8, 100.64.0.0/10 (CGNAT / WireGuard / Tailscale), 198.18.0.0/15, 240.0.0.0/4
    if o[0] == 0
        || (o[0] == 100 && (64..=127).contains(&o[1]))
        || (o[0] == 198 && (o[1] == 18 || o[1] == 19))
        || o[0] >= 240
    {
        return false;
    }
    true
}

/// `SEC-COLIBRI-02`: Parse outbound `http://` or `https://` URL, resolve DNS **once**, validate **all**
/// returned IPs are strictly global public unicast, and return a pinned `(SocketAddr, host, path)`
/// so callers connect directly to the verified `SocketAddr` without a second DNS lookup (zero TOCTOU).
pub fn verify_and_pin_public_url(url: &str) -> Result<(SocketAddr, String, String, bool), String> {
    let (is_tls, rest) = if let Some(r) = url.strip_prefix("https://") {
        (true, r)
    } else if let Some(r) = url.strip_prefix("http://") {
        (false, r)
    } else {
        return Err("SEC-COLIBRI-02: only http:// and https:// schemes are permitted".to_string());
    };

    let (authority, path) = match rest.find('/') {
        Some(idx) => (&rest[..idx], &rest[idx..]),
        None => (rest, "/"),
    };
    if authority.is_empty() || authority.contains('@') {
        return Err("SEC-COLIBRI-02: invalid or credential-bearing URL authority".to_string());
    }

    let default_port = if is_tls { 443u16 } else { 80u16 };
    let (host, port) = split_host_port(authority, default_port)?;
    let lower_host = host.to_ascii_lowercase();
    if lower_host == "localhost"
        || lower_host.ends_with(".local")
        || lower_host.ends_with(".internal")
        || lower_host.ends_with(".lan")
    {
        return Err(format!("SEC-COLIBRI-02: internal hostname '{host}' blocked"));
    }

    if let Ok(ip) = host.parse::<IpAddr>() {
        if !is_strictly_public_ip(ip) {
            return Err(format!("SEC-COLIBRI-02: non-public IP literal '{ip}' blocked"));
        }
        return Ok((SocketAddr::new(ip, port), host, path.to_string(), is_tls));
    }

    let resolved: Vec<SocketAddr> = (host.as_str(), port)
        .to_socket_addrs()
        .map_err(|e| format!("SEC-COLIBRI-02: DNS resolution failed for '{host}': {e}"))?
        .collect();
    if resolved.is_empty() {
        return Err(format!("SEC-COLIBRI-02: DNS returned 0 records for '{host}'"));
    }
    for sa in &resolved {
        if !is_strictly_public_ip(sa.ip()) {
            return Err(format!(
                "SEC-COLIBRI-02: DNS rebinding / private IP '{}' blocked for host '{}'",
                sa.ip(),
                host
            ));
        }
    }

    Ok((resolved[0], host, path.to_string(), is_tls))
}

fn split_host_port(authority: &str, default_port: u16) -> Result<(String, u16), String> {
    if let Some(rest) = authority.strip_prefix('[') {
        let end = rest
            .find(']')
            .ok_or_else(|| "SEC-COLIBRI-02: malformed IPv6 literal".to_string())?;
        let host = rest[..end].to_string();
        let after = &rest[end + 1..];
        if after.is_empty() {
            return Ok((host, default_port));
        }
        let port_str = after
            .strip_prefix(':')
            .ok_or_else(|| "SEC-COLIBRI-02: invalid IPv6 port separator".to_string())?;
        let port = port_str
            .parse::<u16>()
            .map_err(|_| "SEC-COLIBRI-02: invalid port".to_string())?;
        Ok((host, port))
    } else if let Some(idx) = authority.rfind(':') {
        let host = authority[..idx].to_string();
        let port = authority[idx + 1..]
            .parse::<u16>()
            .map_err(|_| "SEC-COLIBRI-02: invalid port".to_string())?;
        Ok((host, port))
    } else {
        Ok((authority.to_string(), default_port))
    }
}

/// `SEC-COLIBRI-01`: Validate `Origin` header on state-changing loopback HTTP requests to defeat browser CSRF.
pub fn verify_loopback_origin(origin: Option<&str>) -> bool {
    let Some(orig) = origin else {
        // Non-browser CLI / curl / SDK callers do not send an Origin header.
        return true;
    };
    let o = orig.trim().to_ascii_lowercase();
    o == "null"
        || o.starts_with("http://127.0.0.1:")
        || o == "http://127.0.0.1"
        || o.starts_with("http://localhost:")
        || o == "http://localhost"
        || o.starts_with("http://[::1]:")
}

/// `SEC-COLIBRI-04` / `TheStandart SEC-03`: Canonicalize base directory and verify relative path cannot escape via `..` or symlinks.
pub fn safe_subpath(base_dir: &Path, user_rel: &str) -> Result<PathBuf, String> {
    if user_rel.is_empty() || user_rel.contains('\0') {
        return Err("SEC-COLIBRI-04: empty or NUL-containing path rejected".to_string());
    }
    let rel_path = Path::new(user_rel);
    if rel_path.is_absolute() {
        return Err("SEC-COLIBRI-04: absolute path rejected".to_string());
    }
    for comp in rel_path.components() {
        match comp {
            Component::Normal(_) | Component::CurDir => {}
            _ => return Err("SEC-COLIBRI-04: parent traversal ('..') rejected".to_string()),
        }
    }
    let canon_base = std::fs::canonicalize(base_dir)
        .map_err(|e| format!("SEC-COLIBRI-04: cannot canonicalize base dir: {e}"))?;
    let candidate = canon_base.join(rel_path);
    if candidate.exists() {
        let canon_cand = std::fs::canonicalize(&candidate)
            .map_err(|e| format!("SEC-COLIBRI-04: cannot canonicalize target: {e}"))?;
        if !canon_cand.starts_with(&canon_base) {
            return Err("SEC-COLIBRI-04: symlink escape outside base directory blocked".to_string());
        }
        Ok(canon_cand)
    } else {
        if let Some(parent) = candidate.parent() {
            if parent.exists() {
                let canon_parent = std::fs::canonicalize(parent)
                    .map_err(|e| format!("SEC-COLIBRI-04: parent canonicalize failed: {e}"))?;
                if !canon_parent.starts_with(&canon_base) {
                    return Err("SEC-COLIBRI-04: parent symlink escape blocked".to_string());
                }
            }
        }
        Ok(candidate)
    }
}
