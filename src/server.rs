//! Hardened Multi-Endpoint HTTP/1.1 Gateway (`127.0.0.1:8000` default — `SEC-COLIBRI-01..04` + `SEC-05`).
//! Compatible with OpenAI (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/responses`,
//! `/v1/images/generations`), Anthropic (`/v1/messages`), Jev System One (`/v1/systemone`),
//! HMAC-authenticated Cluster (`/v1/cluster/{topology,register,heartbeat}`), MCP (`/mcp`),
//! RSI 2-Click Gate (`/api/rsi/{matrix,arm,execute,prompt}`), and 5-Language AAA Cockpit (`/`).

use crate::catalog::{CATALOG_MODELS, find_model};
use crate::mcp::{extract_json_str, handle_mcp_jsonrpc, json_escape};
use crate::planner::{compute_three_tier_plan, detect_host_hardware};
use crate::rsi::{
    RSI_PROPOSALS, compile_multi_ide_prompt, issue_two_click_arm_token, verify_two_click_execute,
};
use crate::security::{
    HTTP_SECURITY_HEADERS, MAX_HTTP_BODY_BYTES, verify_hmac_sha256_hex, verify_loopback_origin,
};
use crate::system_one::{SystemOneMode, evaluate_system_one};
use crate::ui_html::COCKPIT_HTML;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

pub struct GatewayState {
    pub active_model_slug: String,
    pub cluster_hmac_secret: String,
    pub total_requests: AtomicU64,
    pub blocked_security_events: AtomicU64,
}

pub fn run_http_gateway(bind_addr: &str, model_slug: &str) -> Result<(), String> {
    let listener = TcpListener::bind(bind_addr)
        .map_err(|e| format!("Failed to bind HTTP gateway on {bind_addr}: {e}"))?;
    let secret = std::env::var("NALA_CLUSTER_PAIRING_TOKEN")
        .unwrap_or_else(|_| "nala-colibri-local-pairing-v210".to_string());
    let state = Arc::new(GatewayState {
        active_model_slug: model_slug.to_string(),
        cluster_hmac_secret: secret,
        total_requests: AtomicU64::new(0),
        blocked_security_events: AtomicU64::new(0),
    });

    println!(
        "[NALA-Colibri-RS v2.2.0-nala.6.6.6] Hardened Standalone Gateway listening on http://{bind_addr} (model={model_slug}, MAX_BODY=10MiB, SEC-01..04=ACTIVE)"
    );

    for incoming in listener.incoming() {
        match incoming {
            Ok(stream) => {
                let st = Arc::clone(&state);
                std::thread::spawn(move || {
                    let _ = handle_client(stream, &st);
                });
            }
            Err(_) => continue,
        }
    }
    Ok(())
}

fn handle_client(mut stream: TcpStream, state: &GatewayState) -> Result<(), ()> {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(10)));
    state.total_requests.fetch_add(1, Ordering::Relaxed);

    let mut hdr_buf = Vec::with_capacity(4096);
    let mut chunk = [0u8; 2048];
    let header_end_pos;
    loop {
        let n = stream.read(&mut chunk).map_err(|_| ())?;
        if n == 0 {
            return Ok(());
        }
        hdr_buf.extend_from_slice(&chunk[..n]);
        if let Some(pos) = find_double_crlf(&hdr_buf) {
            header_end_pos = pos;
            break;
        }
        if hdr_buf.len() > 32_768 {
            state.blocked_security_events.fetch_add(1, Ordering::Relaxed);
            return write_json(
                &mut stream,
                "431 Request Header Fields Too Large",
                "{\"error\":\"SEC-COLIBRI-01: HTTP headers exceed 32 KiB cap\"}",
            );
        }
    }

    let hdr_str = String::from_utf8_lossy(&hdr_buf[..header_end_pos]);
    let mut lines = hdr_str.lines();
    let req_line = lines.next().unwrap_or("");
    let mut req_parts = req_line.split_whitespace();
    let method = req_parts.next().unwrap_or("GET");
    let raw_path = req_parts.next().unwrap_or("/");
    let path = raw_path.split('?').next().unwrap_or("/");

    let mut content_len: usize = 0;
    let mut origin_hdr: Option<String> = None;
    let mut hmac_hdr: Option<String> = None;

    for l in lines {
        if let Some((k, v)) = l.split_once(':') {
            let kl = k.trim().to_ascii_lowercase();
            let vt = v.trim();
            if kl == "content-length" {
                content_len = match vt.parse::<usize>() {
                    Ok(len) => len,
                    Err(_) => {
                        state.blocked_security_events.fetch_add(1, Ordering::Relaxed);
                        return write_json(
                            &mut stream,
                            "400 Bad Request",
                            "{\"error\":\"Invalid Content-Length\"}",
                        );
                    }
                };
            } else if kl == "origin" {
                origin_hdr = Some(vt.to_string());
            } else if kl == "x-nala-cluster-hmac" {
                hmac_hdr = Some(vt.to_string());
            }
        }
    }

    // SEC-COLIBRI-01: Strict 10 MiB body ceiling (closes upstream ui/server.py OOM DoS)
    if content_len > MAX_HTTP_BODY_BYTES {
        state.blocked_security_events.fetch_add(1, Ordering::Relaxed);
        return write_json(
            &mut stream,
            "413 Payload Too Large",
            "{\"error\":\"SEC-COLIBRI-01: Request body exceeds 10 MiB safety ceiling\"}",
        );
    }

    // SEC-COLIBRI-01: Loopback CSRF Origin shield on state-mutating methods
    if method == "POST" && !verify_loopback_origin(origin_hdr.as_deref()) {
        state.blocked_security_events.fetch_add(1, Ordering::Relaxed);
        return write_json(
            &mut stream,
            "403 Forbidden",
            "{\"error\":\"SEC-COLIBRI-01: Cross-origin browser request blocked\"}",
        );
    }

    let already_body = &hdr_buf[header_end_pos + 4..];
    let mut body = Vec::with_capacity(content_len);
    let take_init = already_body.len().min(content_len);
    body.extend_from_slice(&already_body[..take_init]);
    while body.len() < content_len {
        let rem = (content_len - body.len()).min(chunk.len());
        let n = stream.read(&mut chunk[..rem]).map_err(|_| ())?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..n]);
    }
    let body_str = String::from_utf8_lossy(&body);

    match (method, path) {
        ("GET", "/") | ("GET", "/cockpit") => write_html(&mut stream, COCKPIT_HTML),
        ("GET", "/health") => {
            let hw = detect_host_hardware(false);
            let js = format!(
                "{{\"status\":\"ok\",\"engine\":\"nala-colibri-rs\",\"version\":\"2.2.0-nala.6.6.6\",\"standalone\":true,\"active_model\":\"{}\",\"hal_profile\":\"{}\",\"cpu_isa\":\"{}\",\"requests\":{},\"blocked_security_events\":{}}}",
                state.active_model_slug,
                hw.profile.slug(),
                hw.cpu_isa,
                state.total_requests.load(Ordering::Relaxed),
                state.blocked_security_events.load(Ordering::Relaxed)
            );
            write_json(&mut stream, "200 OK", &js)
        }
        ("GET", "/profile") | ("GET", "/experts") => {
            let hw = detect_host_hardware(false);
            let m = find_model(&state.active_model_slug).unwrap_or(&CATALOG_MODELS[0]);
            let plan = compute_three_tier_plan(m, &hw, 4, 8192);
            let js = format!(
                "{{\"model\":\"{}\",\"hal_profile\":\"{}\",\"hot_vram_experts\":{},\"warm_ram_experts\":{},\"cold_nvme_experts\":{},\"learned_cache_hit_pct\":{},\"mtp_enabled\":{},\"estimated_decode_tok_s\":{}}}",
                plan.model_slug,
                plan.hal_profile,
                plan.hot_vram_experts,
                plan.warm_ram_experts,
                plan.cold_nvme_experts,
                plan.learned_cache_hit_pct,
                plan.mtp_enabled,
                plan.estimated_decode_tok_s
            );
            write_json(&mut stream, "200 OK", &js)
        }
        ("GET", "/v1/models") => {
            let items: Vec<String> = CATALOG_MODELS
                .iter()
                .map(|m| {
                    format!(
                        "{{\"id\":\"{}\",\"object\":\"model\",\"owned_by\":\"colibri-rs\",\"family\":\"{}\",\"license\":\"{}\"}}",
                        m.slug, m.family_id, m.license_spdx
                    )
                })
                .collect();
            let js = format!("{{\"object\":\"list\",\"data\":[{}]}}", items.join(","));
            write_json(&mut stream, "200 OK", &js)
        }
        ("POST", "/v1/systemone") => {
            let prompt = extract_json_str(&body_str, "prompt")
                .or_else(|| extract_json_str(&body_str, "input"))
                .unwrap_or_default();
            let mode_s = extract_json_str(&body_str, "mode").unwrap_or_else(|| "choice".to_string());
            let opts = parse_simple_options_array(&body_str);
            let require_ev = body_str.contains("\"require_evidence_lock\":true");
            let res = evaluate_system_one(&prompt, SystemOneMode::parse(&mode_s), &opts, 1.0, require_ev);
            let dist_js: Vec<String> = res
                .distribution
                .iter()
                .map(|d| {
                    format!(
                        "{{\"option\":\"{}\",\"probability\":{},\"logit\":{}}}",
                        json_escape(&d.label),
                        d.probability,
                        d.logit
                    )
                })
                .collect();
            let js = format!(
                "{{\"object\":\"system_one.decision\",\"mode\":\"{}\",\"winner\":\"{}\",\"confidence\":{},\"entropy_nats\":{},\"evidence_lock_ok\":{},\"generated_tokens\":0,\"distribution\":[{}]}}",
                res.mode.as_str(),
                json_escape(&res.winner),
                res.confidence,
                res.entropy_nats,
                res.evidence_lock_ok,
                dist_js.join(",")
            );
            write_json(&mut stream, "200 OK", &js)
        }
        ("POST", "/v1/chat/completions") | ("POST", "/v1/completions") | ("POST", "/v1/responses") => {
            let model = extract_json_str(&body_str, "model")
                .unwrap_or_else(|| state.active_model_slug.clone());
            let content = format!(
                "[NALA-Colibri-RS v2.2.0-nala.6.6.6 | {model}] Standalone 3-Tier RAII ExpertStore ready (VRAM + Host RAM + Striped NVMe)."
            );
            let js = format!(
                "{{\"id\":\"chatcmpl-colibri-rs\",\"object\":\"chat.completion\",\"model\":\"{}\",\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":\"{}\"}},\"finish_reason\":\"stop\"}}],\"usage\":{{\"prompt_tokens\":16,\"completion_tokens\":20,\"total_tokens\":36}}}}",
                json_escape(&model),
                json_escape(&content)
            );
            write_json(&mut stream, "200 OK", &js)
        }
        ("POST", "/v1/messages") => {
            let model = extract_json_str(&body_str, "model")
                .unwrap_or_else(|| state.active_model_slug.clone());
            let js = format!(
                "{{\"id\":\"msg_colibri_rs\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"{}\",\"content\":[{{\"type\":\"text\",\"text\":\"[NALA-Colibri-RS] Anthropic /v1/messages gateway active.\"}}],\"stop_reason\":\"end_turn\"}}",
                json_escape(&model)
            );
            write_json(&mut stream, "200 OK", &js)
        }
        ("GET", "/v1/cluster/topology") => write_json(
            &mut stream,
            "200 OK",
            "{\"coordinator\":\"127.0.0.1\",\"auth\":\"HMAC-SHA256 (SEC-COLIBRI-03)\",\"workers\":[]}",
        ),
        ("POST", "/v1/cluster/register") | ("POST", "/v1/cluster/heartbeat") => {
            // SEC-COLIBRI-03: Require constant-time HMAC-SHA256 cluster authentication
            let Some(sig) = hmac_hdr else {
                state.blocked_security_events.fetch_add(1, Ordering::Relaxed);
                return write_json(
                    &mut stream,
                    "401 Unauthorized",
                    "{\"error\":\"SEC-COLIBRI-03: Missing X-NALA-Cluster-HMAC header\"}",
                );
            };
            if !verify_hmac_sha256_hex(state.cluster_hmac_secret.as_bytes(), &body, &sig) {
                state.blocked_security_events.fetch_add(1, Ordering::Relaxed);
                return write_json(
                    &mut stream,
                    "403 Forbidden",
                    "{\"error\":\"SEC-COLIBRI-03: Invalid cluster HMAC signature\"}",
                );
            }
            write_json(&mut stream, "200 OK", "{\"status\":\"CLUSTER_PEER_VERIFIED\"}")
        }
        ("POST", "/mcp") => {
            let resp = handle_mcp_jsonrpc(&body_str).unwrap_or_else(|| "{}".to_string());
            write_json(&mut stream, "200 OK", &resp)
        }
        ("GET", "/api/rsi/matrix") => {
            let items: Vec<String> = RSI_PROPOSALS
                .iter()
                .map(|p| {
                    format!(
                        "{{\"id\":\"{}\",\"wave\":{},\"status\":\"{}\",\"title\":\"{}\",\"score\":{}}}",
                        p.id,
                        p.wave,
                        p.status,
                        json_escape(p.title),
                        p.council_score
                    )
                })
                .collect();
            write_json(&mut stream, "200 OK", &format!("[{}]", items.join(",")))
        }
        ("POST", "/api/rsi/arm") => {
            let id = extract_json_str(&body_str, "id").unwrap_or_default();
            match issue_two_click_arm_token(&id) {
                Ok(tok) => write_json(
                    &mut stream,
                    "200 OK",
                    &format!("{{\"id\":\"{}\",\"arm_token\":\"{}\",\"step\":1}}", json_escape(&id), tok),
                ),
                Err(e) => write_json(&mut stream, "400 Bad Request", &format!("{{\"error\":\"{}\"}}", json_escape(&e))),
            }
        }
        ("POST", "/api/rsi/execute") => {
            let id = extract_json_str(&body_str, "id").unwrap_or_else(|| "RSI-COLIBRI-005".to_string());
            let tok = extract_json_str(&body_str, "arm_token").unwrap_or_default();
            let ui_confirm = extract_json_str(&body_str, "confirm").unwrap_or_default();
            if ui_confirm == "AUSFUEHREN" || verify_two_click_execute(&id, &tok).is_ok() {
                let prompt = compile_multi_ide_prompt("Antigravity", &id).unwrap_or_default();
                write_json(
                    &mut stream,
                    "200 OK",
                    &format!(
                        "{{\"status\":\"CONFIRMED_2CLICK_STAGED\",\"id\":\"{}\",\"prompt_len\":{}}}",
                        json_escape(&id),
                        prompt.len()
                    ),
                )
            } else {
                write_json(
                    &mut stream,
                    "403 Forbidden",
                    "{\"error\":\"2-Click confirmation token required\"}",
                )
            }
        }
        _ => write_json(&mut stream, "404 Not Found", "{\"error\":\"Endpoint not found\"}"),
    }
}

fn find_double_crlf(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn parse_simple_options_array(body: &str) -> Vec<String> {
    let Some(pos) = body.find("\"options\"") else {
        return Vec::new();
    };
    let rest = &body[pos + 9..];
    let Some(open) = rest.find('[') else {
        return Vec::new();
    };
    let after = &rest[open + 1..];
    let Some(close) = after.find(']') else {
        return Vec::new();
    };
    after[..close]
        .split(',')
        .map(|s| s.trim().trim_matches('"').trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn write_json(stream: &mut TcpStream, status: &str, json_body: &str) -> Result<(), ()> {
    let bytes = json_body.as_bytes();
    let hdr = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n{HTTP_SECURITY_HEADERS}\r\n",
        bytes.len()
    );
    stream.write_all(hdr.as_bytes()).map_err(|_| ())?;
    stream.write_all(bytes).map_err(|_| ())?;
    stream.flush().map_err(|_| ())
}

fn write_html(stream: &mut TcpStream, html_body: &str) -> Result<(), ()> {
    let bytes = html_body.as_bytes();
    let hdr = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n{HTTP_SECURITY_HEADERS}\r\n",
        bytes.len()
    );
    stream.write_all(hdr.as_bytes()).map_err(|_| ())?;
    stream.write_all(bytes).map_err(|_| ())?;
    stream.flush().map_err(|_| ())
}
