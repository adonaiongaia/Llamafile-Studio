//! Hugging Face model browser.
//!
//! Everything here talks to the public Hugging Face Hub API:
//!
//! * `GET /api/models?search=…&filter=gguf` - repository search with GGUF
//!   metadata (`architecture`, `context_length`, total size) in one round trip.
//! * `GET /api/models/{repo}` - single repository, including its `chat_template`
//!   which we inspect for tool-calling and reasoning support and then discard.
//! * `GET /api/models/{repo}/tree/main?recursive=true` - file listing with real
//!   sizes (LFS sizes are the authoritative ones for model weights).
//! * `GET /{repo}/resolve/main/{file}` - the actual weight download.
//!
//! Capability badges (tool calling, thinking, vision, MoE, code, multilingual)
//! are *inferred*: tags, pipeline tag, architecture, repository name and - the
//! most reliable signal - the repository's Jinja chat template. The UI labels
//! them as detected hints rather than hard facts, because the Hub has no
//! normalised capability field.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter};

const HF_HOST: &str = "https://huggingface.co";
const USER_AGENT: &str = "llamafile-studio/1.0 (+https://github.com/llamafile-studio)";

/// Set by `cancel_model_download`, observed by the streaming loop.
static CANCEL_DOWNLOAD: AtomicBool = AtomicBool::new(false);

/* ------------------------------------------------------------------ types */

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelBadges {
    pub tool_use: bool,
    pub reasoning: bool,
    pub vision: bool,
    pub moe: bool,
    pub code: bool,
    pub multilingual: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HfSummary {
    pub repo: String,
    pub author: String,
    pub name: String,
    pub downloads: u64,
    pub likes: u64,
    pub updated: String,
    pub pipeline: String,
    pub architecture: String,
    pub context_length: u64,
    /// Total size of every file in the repository (bytes).
    pub repo_size: u64,
    /// Parameter count parsed from the repository name, e.g. `8B`.
    pub params: String,
    pub license: String,
    pub gated: bool,
    pub badges: ModelBadges,
    pub signals: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct QuantFile {
    /// Human label such as `Q4_K_M`, `IQ2_XXS`, `BF16`.
    pub label: String,
    pub file: String,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HfDetail {
    pub repo: String,
    pub author: String,
    pub name: String,
    pub architecture: String,
    pub context_length: u64,
    pub repo_size: u64,
    pub downloads: u64,
    pub likes: u64,
    pub updated: String,
    pub pipeline: String,
    pub license: String,
    pub base_model: String,
    pub languages: Vec<String>,
    pub gated: bool,
    pub badges: ModelBadges,
    pub signals: Vec<String>,
    pub quants: Vec<QuantFile>,
    /// Multimodal projector files (`mmproj`); not embedded by the compiler yet.
    pub mmproj: Vec<QuantFile>,
    /// Files skipped because they are sharded across several parts.
    pub split_count: usize,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelProgress {
    repo: String,
    file: String,
    downloaded: u64,
    total: u64,
    percent: f64,
}

/* --------------------------------------------------------- API responses */

#[derive(Debug, Deserialize)]
struct ApiModel {
    #[serde(default)]
    id: String,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    likes: u64,
    #[serde(default, rename = "lastModified")]
    last_modified: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default, rename = "pipeline_tag")]
    pipeline_tag: Option<String>,
    #[serde(default, rename = "cardData")]
    card_data: Option<serde_json::Value>,
    #[serde(default)]
    gguf: Option<GgufMeta>,
    #[serde(default)]
    gated: Option<serde_json::Value>,
}

#[derive(Debug, Default, Clone, Deserialize)]
struct GgufMeta {
    #[serde(default)]
    architecture: String,
    #[serde(default)]
    context_length: u64,
    #[serde(default, rename = "totalFileSize")]
    total_file_size: u64,
    /// Only requested for a single repository; deliberately never serialised.
    #[serde(default)]
    chat_template: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TreeEntry {
    #[serde(default, rename = "type")]
    kind: String,
    #[serde(default)]
    path: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    lfs: Option<LfsInfo>,
}

#[derive(Debug, Deserialize)]
struct LfsInfo {
    #[serde(default)]
    size: u64,
}

/* ------------------------------------------------------------- utilities */

fn client(timeout: Option<Duration>) -> Result<reqwest::Client, String> {
    let mut builder = reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .connect_timeout(Duration::from_secs(20))
        .redirect(reqwest::redirect::Policy::limited(10));
    // Downloads can legitimately run for many minutes, so only the metadata
    // requests get a hard timeout.
    if let Some(timeout) = timeout {
        builder = builder.timeout(timeout);
    }
    builder
        .build()
        .map_err(|error| format!("cannot create the HTTP client: {error}"))
}

/// Maps an HTTP failure to a message a human can act on.
fn http_error(context: &str, status: reqwest::StatusCode) -> String {
    match status.as_u16() {
        401 | 403 => format!(
            "{context}: access denied (HTTP {}). Gated repositories need you to accept their licence on huggingface.co first.",
            status.as_u16()
        ),
        404 => format!("{context}: not found (HTTP 404)"),
        429 => format!("{context}: rate limited by huggingface.co, try again in a minute"),
        code => format!("{context}: HTTP {code}"),
    }
}

/// Total file size from a `Content-Range: bytes 0-499/500` header.
fn content_range_total(response: &reqwest::Response) -> Option<u64> {
    let value = response
        .headers()
        .get(reqwest::header::CONTENT_RANGE)?
        .to_str()
        .ok()?;
    parse_content_range_total(value)
}

/// Parses the `/total` part of a `Content-Range` header. `*` (unknown) and any
/// malformed value yield `None`, so the caller falls back to the length header.
fn parse_content_range_total(value: &str) -> Option<u64> {
    let (_, total) = value.rsplit_once('/')?;
    total.trim().parse().ok()
}

/// Percent-encodes one path segment for use in a URL.
fn encode_segment(input: &str) -> String {
    let mut encoded = String::with_capacity(input.len());
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char)
            }
            other => encoded.push_str(&format!("%{other:02X}")),
        }
    }
    encoded
}

/// Splits and validates `owner/name`, rejecting anything that could escape the
/// expected URL shape.
fn validate_repo(repo: &str) -> Result<(String, String), String> {
    let trimmed = repo.trim();
    let mut parts = trimmed.split('/');
    let (Some(owner), Some(name), None) = (parts.next(), parts.next(), parts.next()) else {
        return Err(format!(
            "\"{trimmed}\" is not a repository id (expected \"owner/name\")"
        ));
    };
    let valid = |segment: &str| {
        !segment.is_empty()
            && segment
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_'))
    };
    if !valid(owner) || !valid(name) {
        return Err(format!("\"{trimmed}\" contains unsupported characters"));
    }
    Ok((owner.to_string(), name.to_string()))
}

/// Builds the download URL. Each path segment is encoded separately because
/// repository files may live in sub-folders (for example
/// `IQ2_XS/model-00001-of-00002.gguf`), and encoding the slash would break the
/// request.
fn resolve_url(owner: &str, name: &str, file: &str) -> String {
    let path: Vec<String> = file.split('/').map(encode_segment).collect();
    format!(
        "{HF_HOST}/{}/{}/resolve/main/{}?download=true",
        encode_segment(owner),
        encode_segment(name),
        path.join("/")
    )
}

/// True when the Hub's `gated` field marks the repository as restricted.
fn is_gated(gated: &Option<serde_json::Value>) -> bool {
    match gated {
        Some(serde_json::Value::Bool(flag)) => *flag,
        Some(serde_json::Value::String(text)) => text != "false",
        _ => false,
    }
}

/// Parameter count from the repository name (`Qwen3-8B-GGUF` -> `8B`).
fn parse_params(name: &str) -> String {
    let bytes = name.as_bytes();
    let mut found: Option<String> = None;
    let mut index = 0;
    while index < bytes.len() {
        if !bytes[index].is_ascii_digit() {
            index += 1;
            continue;
        }
        let start = index;
        while index < bytes.len() && (bytes[index].is_ascii_digit() || bytes[index] == b'.') {
            index += 1;
        }
        // Require a B/M suffix that is not followed by another word character.
        // Both `8B` and `1b` spellings occur in repository names.
        if index < bytes.len() && matches!(bytes[index], b'B' | b'M' | b'b' | b'm') {
            let next_is_word = bytes
                .get(index + 1)
                .map(|next| next.is_ascii_alphanumeric())
                .unwrap_or(false);
            if !next_is_word {
                let number = &name[start..index];
                // Ignore context sizes and versions like "3.1" without a suffix.
                // Keep the first match: in `35B-A3B` the total is 35B and the
                // second number is only the active parameter count.
                let suffix = (bytes[index] as char).to_ascii_uppercase();
                if found.is_none() {
                    found = Some(format!("{number}{suffix}"));
                }
            }
        }
    }
    found.unwrap_or_default()
}

/// Extracts the quantisation label from a GGUF filename.
///
/// Quant labels are the trailing run of `[A-Z0-9_]` characters in the stem, so
/// `Qwen3-8B-Q4_K_M.gguf` yields `Q4_K_M`. Runs that do not start with a known
/// quant prefix (for example the `8B` in `Qwen3-8B.gguf`) are rejected.
fn quant_label(file: &str) -> String {
    let stem = file.trim_end_matches(".gguf").trim_end_matches(".GGUF");
    let mut tail = String::new();
    for character in stem.chars().rev() {
        if character.is_ascii_alphanumeric() || character == '_' {
            tail.push(character);
        } else {
            break;
        }
    }
    let candidate: String = tail.chars().rev().collect();
    let upper = candidate.to_uppercase();
    let known_prefix = ["IQ", "Q", "F", "BF", "MX", "TQ"]
        .iter()
        .any(|prefix| upper.starts_with(prefix));
    let has_digit = upper.chars().any(|c| c.is_ascii_digit());
    if known_prefix && has_digit {
        upper
    } else {
        // Fall back to the stem so the row is still identifiable.
        stem.to_string()
    }
}

/// Heuristic capability detection.
///
/// Returns the badges plus the human-readable reasons behind them so the UI can
/// explain itself instead of presenting guesses as facts.
fn detect(
    repo: &str,
    tags: &[String],
    pipeline: &str,
    architecture: &str,
    template: Option<&str>,
    language_count: usize,
    has_mmproj: bool,
) -> (ModelBadges, Vec<String>) {
    let mut haystack = format!("{repo} {} {pipeline} {architecture}", tags.join(" "));
    haystack = haystack.to_lowercase();
    let template = template.unwrap_or("").to_lowercase();
    let mut signals: Vec<String> = Vec::new();

    let tool_by_template = ["tool_call", "tool_response", "<tools>", "function_call"]
        .iter()
        .any(|needle| template.contains(needle));
    let tool_by_tags = ["tool-use", "tool_use", "function-calling", "tool calling"]
        .iter()
        .any(|needle| haystack.contains(needle));
    let tool_use = tool_by_template || tool_by_tags;
    if tool_by_template {
        signals.push("chat template implements tool calls".to_string());
    } else if tool_by_tags {
        signals.push("repository is tagged for tool use".to_string());
    }

    let reasoning_by_template = [
        "reasoning_content",
        "enable_thinking",
        "reasoning_effort",
        " thinking",
        "reasoning effort",
    ]
    .iter()
    .any(|needle| template.contains(needle));
    let reasoning_by_name = [
        "deepseek-r1",
        "qwq",
        "magistral",
        "reasoner",
        "-r1-",
        "reasoning",
        "gpt-oss",
        "glm-4.5",
        "thinking",
    ]
    .iter()
    .any(|needle| haystack.contains(needle));
    let reasoning = reasoning_by_template || reasoning_by_name;
    if reasoning_by_template {
        signals.push("chat template exposes a thinking/reasoning mode".to_string());
    } else if reasoning_by_name {
        signals.push("name or tags indicate a reasoning model".to_string());
    }

    let vision = has_mmproj
        || pipeline == "image-text-to-text"
        || ["vision", "multimodal", "llava", "-vl", "image-text"]
            .iter()
            .any(|needle| haystack.contains(needle));
    if vision {
        signals.push(if has_mmproj {
            "repository ships a multimodal projector".to_string()
        } else {
            "vision or multimodal tag".to_string()
        });
    }

    let moe = haystack.contains("moe")
        || architecture.to_lowercase().contains("moe")
        || ["-a3b", "-a22b", "a3b-", "a22b-"]
            .iter()
            .any(|needle| haystack.contains(needle));
    if moe {
        signals.push("mixture-of-experts architecture".to_string());
    }

    let code = ["coder", "codestral", "code-", "-code", "devstral"]
        .iter()
        .any(|needle| haystack.contains(needle));
    if code {
        signals.push("code-focused model family".to_string());
    }

    let multilingual = language_count > 1 || haystack.contains("multilingual");
    if multilingual {
        signals.push("card lists multiple languages".to_string());
    }

    (
        ModelBadges {
            tool_use,
            reasoning,
            vision,
            moe,
            code,
            multilingual,
        },
        signals,
    )
}

/* ---------------------------------------------------------------- search */

/// Searches GGUF repositories. An empty query lists the most popular GGUF
/// repositories overall, which doubles as a discovery view.
#[tauri::command]
pub async fn hf_search(query: String, sort: String, limit: u32) -> Result<Vec<HfSummary>, String> {
    let client = client(Some(Duration::from_secs(30)))?;
    let order = match sort.as_str() {
        "likes" => "likes",
        "lastModified" => "lastModified",
        _ => "downloads",
    };
    let limit = limit.clamp(1, 50).to_string();

    let mut params: Vec<(&str, String)> = vec![
        ("filter", "gguf".to_string()),
        ("sort", order.to_string()),
        ("direction", "-1".to_string()),
        ("limit", limit),
        ("expand[]", "gguf".to_string()),
        ("expand[]", "cardData".to_string()),
        ("expand[]", "tags".to_string()),
        ("expand[]", "downloads".to_string()),
        ("expand[]", "likes".to_string()),
        ("expand[]", "lastModified".to_string()),
        ("expand[]", "pipeline_tag".to_string()),
        ("expand[]", "gated".to_string()),
    ];
    let trimmed = query.trim();
    if !trimmed.is_empty() {
        params.push(("search", trimmed.to_string()));
    }

    let response = client
        .get(format!("{HF_HOST}/api/models"))
        .query(&params)
        .send()
        .await
        .map_err(|error| format!("cannot reach huggingface.co: {error}"))?;
    if !response.status().is_success() {
        return Err(http_error("searching huggingface.co", response.status()));
    }

    let models: Vec<ApiModel> = response
        .json()
        .await
        .map_err(|error| format!("the Hugging Face response could not be read: {error}"))?;

    Ok(models
        .into_iter()
        .filter(|model| !model.id.is_empty())
        .map(|model| {
            let (author, name) = model
                .id
                .split_once('/')
                .map(|(owner, rest)| (owner.to_string(), rest.to_string()))
                .unwrap_or_else(|| ("".to_string(), model.id.clone()));
            let pipeline = model.pipeline_tag.clone().unwrap_or_default();
            let tags = model.tags.clone();
            let gguf = model.gguf.unwrap_or_default();
            let card = model.card_data.clone().unwrap_or(serde_json::Value::Null);
            let languages = card
                .get("language")
                .map(|value| match value {
                    serde_json::Value::Array(items) => items.len(),
                    serde_json::Value::String(_) => 1,
                    _ => 0,
                })
                .unwrap_or(0);
            let (badges, signals) = detect(
                &model.id,
                &tags,
                &pipeline,
                &gguf.architecture,
                None,
                languages,
                false,
            );
            let license = card
                .get("license")
                .and_then(|value| value.as_str())
                .map(str::to_string)
                .or_else(|| {
                    tags.iter()
                        .find_map(|tag| tag.strip_prefix("license:").map(str::to_string))
                })
                .unwrap_or_default();

            HfSummary {
                repo: model.id.clone(),
                author,
                name,
                downloads: model.downloads,
                likes: model.likes,
                updated: model.last_modified,
                pipeline,
                architecture: gguf.architecture,
                context_length: gguf.context_length,
                repo_size: gguf.total_file_size,
                params: parse_params(&model.id),
                license,
                gated: is_gated(&model.gated),
                badges,
                signals,
            }
        })
        .collect())
}

/* ---------------------------------------------------------------- detail */

/// Full detail for one repository: metadata, inferred capabilities and the list
/// of downloadable quantisations.
#[tauri::command]
pub async fn hf_model(repo: String) -> Result<HfDetail, String> {
    let (owner, name) = validate_repo(&repo)?;
    let client = client(Some(Duration::from_secs(30)))?;

    let meta_response = client
        .get(format!("{HF_HOST}/api/models/{owner}/{name}"))
        .query(&[
            ("expand[]", "gguf"),
            ("expand[]", "cardData"),
            ("expand[]", "tags"),
            ("expand[]", "pipeline_tag"),
            ("expand[]", "downloads"),
            ("expand[]", "likes"),
            ("expand[]", "lastModified"),
            ("expand[]", "gated"),
        ])
        .send()
        .await
        .map_err(|error| format!("cannot reach huggingface.co: {error}"))?;
    if !meta_response.status().is_success() {
        return Err(http_error(&format!("loading {repo}"), meta_response.status()));
    }
    let model: ApiModel = meta_response
        .json()
        .await
        .map_err(|error| format!("the repository metadata could not be read: {error}"))?;

    let tree_response = client
        .get(format!("{HF_HOST}/api/models/{owner}/{name}/tree/main"))
        .query(&[("recursive", "true"), ("limit", "1000")])
        .send()
        .await
        .map_err(|error| format!("cannot list the files of {repo}: {error}"))?;
    if !tree_response.status().is_success() {
        return Err(http_error(
            &format!("listing the files of {repo}"),
            tree_response.status(),
        ));
    }
    let entries: Vec<TreeEntry> = tree_response
        .json()
        .await
        .map_err(|error| format!("the file listing could not be read: {error}"))?;

    let mut quants: Vec<QuantFile> = Vec::new();
    let mut mmproj: Vec<QuantFile> = Vec::new();
    let mut split_count = 0usize;

    for entry in &entries {
        if entry.kind != "file" {
            continue;
        }
        let lower = entry.path.to_lowercase();
        if !lower.ends_with(".gguf") {
            continue;
        }
        // LFS entries report the pointer size in `size`; the payload size lives
        // in the `lfs` sub-object.
        let size = entry
            .lfs
            .as_ref()
            .map(|lfs| lfs.size)
            .filter(|size| *size > 0)
            .unwrap_or(entry.size);
        let file_name = entry
            .path
            .rsplit('/')
            .next()
            .unwrap_or(&entry.path)
            .to_string();

        if lower.contains("mmproj") {
            mmproj.push(QuantFile {
                label: quant_label(&file_name),
                file: entry.path.clone(),
                size,
            });
            continue;
        }
        // Sharded GGUFs cannot be embedded as one model, so they are skipped.
        if lower.contains("-of-0") || lower.contains("_of_0") {
            split_count += 1;
            continue;
        }
        quants.push(QuantFile {
            label: quant_label(&file_name),
            file: entry.path.clone(),
            size,
        });
    }

    quants.sort_by(|a, b| a.size.cmp(&b.size));
    mmproj.sort_by(|a, b| a.size.cmp(&b.size));

    let gguf = model.gguf.clone().unwrap_or_default();
    let card = model.card_data.clone().unwrap_or(serde_json::Value::Null);
    let languages: Vec<String> = match card.get("language") {
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .filter_map(|item| item.as_str().map(str::to_string))
            .collect(),
        Some(serde_json::Value::String(text)) => vec![text.clone()],
        _ => Vec::new(),
    };
    let base_model = match card.get("base_model") {
        Some(serde_json::Value::Array(items)) => items
            .first()
            .and_then(|item| item.as_str())
            .unwrap_or("")
            .to_string(),
        Some(serde_json::Value::String(text)) => text.clone(),
        _ => String::new(),
    };
    let license = card
        .get("license")
        .and_then(|value| value.as_str())
        .map(str::to_string)
        .or_else(|| {
            model
                .tags
                .iter()
                .find_map(|tag| tag.strip_prefix("license:").map(str::to_string))
        })
        .unwrap_or_default();
    let pipeline = model.pipeline_tag.clone().unwrap_or_default();

    let (badges, signals) = detect(
        &model.id,
        &model.tags,
        &pipeline,
        &gguf.architecture,
        gguf.chat_template.as_deref(),
        languages.len(),
        !mmproj.is_empty(),
    );

    Ok(HfDetail {
        repo: model.id.clone(),
        author: owner,
        name,
        architecture: gguf.architecture,
        context_length: gguf.context_length,
        repo_size: gguf.total_file_size,
        downloads: model.downloads,
        likes: model.likes,
        updated: model.last_modified,
        pipeline,
        license,
        base_model,
        languages,
        gated: is_gated(&model.gated),
        badges,
        signals,
        quants,
        mmproj,
        split_count,
    })
}

/* -------------------------------------------------------------- download */

/// Default destination for downloaded weights: `$HOME/Models`, created on
/// demand. Falls back to the temp directory when there is no home directory.
#[tauri::command]
pub fn default_model_dir() -> String {
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        let models = home.join("Models");
        if fs_create(&models).is_ok() {
            return models.to_string_lossy().to_string();
        }
        return home.to_string_lossy().to_string();
    }
    std::env::temp_dir().to_string_lossy().to_string()
}

fn fs_create(path: &Path) -> std::io::Result<()> {
    if path.is_dir() {
        return Ok(());
    }
    std::fs::create_dir_all(path)
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CachedModel {
    pub file: String,
    pub path: String,
    pub size: u64,
}

/// Lists the GGUF files already present in a folder.
///
/// The download folder doubles as the cache: no database is needed, and the
/// browser can show a "Downloaded" badge for a quantisation whose file is
/// already on disk (matched by name and exact byte size, so a truncated or
/// different build is not mistaken for a cached one).
#[tauri::command]
pub fn cached_models(dir: String) -> Vec<CachedModel> {
    let directory = PathBuf::from(dir.trim());
    if directory.as_os_str().is_empty() {
        return Vec::new();
    }
    let Ok(entries) = std::fs::read_dir(&directory) else {
        return Vec::new();
    };

    let mut cached: Vec<CachedModel> = entries
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().map(|kind| kind.is_file()).unwrap_or(false))
        .filter_map(|entry| {
            let path = entry.path();
            let name = path.file_name()?.to_string_lossy().to_string();
            // In-flight `.part` files are never reported as available.
            if !name.to_lowercase().ends_with(".gguf") {
                return None;
            }
            let size = entry.metadata().ok()?.len();
            Some(CachedModel {
                file: name,
                path: path.to_string_lossy().to_string(),
                size,
            })
        })
        .collect();

    cached.sort_by(|a, b| a.file.cmp(&b.file));
    cached
}

/* --------------------------------------------------------- mmproj pairing */

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MmprojCandidate {
    pub file: String,
    pub path: String,
    pub size: u64,
}

/// Words that identify the model family, ignoring quantisation noise.
fn family_tokens(stem: &str) -> Vec<String> {
    stem.to_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|token| token.len() >= 3)
        .filter(|token| !matches!(*token, "gguf" | "model" | "mmproj" | "clip"))
        .filter(|token| !(token.starts_with('q') && token[1..].chars().all(|c| c.is_ascii_digit())))
        .map(str::to_string)
        .collect()
}

/// Ranks a projector for a model: family overlap first, then the most precise
/// flavour. Projectors are published at a fixed precision and `f16` is the usual
/// recommendation, so it wins the tie when two files match equally well.
fn mmproj_rank(model_stem: &str, file_name: &str) -> (usize, u8) {
    let candidate = file_name.to_lowercase();
    let overlap = family_tokens(model_stem)
        .iter()
        .filter(|token| candidate.contains(token.as_str()))
        .count();
    let precision = if candidate.contains("f16") || candidate.contains("fp16") {
        2
    } else if candidate.contains("bf16") {
        1
    } else {
        0
    };
    (overlap, precision)
}

/// Lists the projector files sitting next to `model`, best match first.
///
/// Projectors are published beside the weights (`mmproj-*.gguf`), so the model's
/// own folder is the only place worth searching. An empty result means the folder
/// has no projector rather than that pairing failed.
#[tauri::command]
pub fn mmproj_candidates(model: String) -> Vec<MmprojCandidate> {
    let model_path = PathBuf::from(model.trim());
    let Some(directory) = model_path.parent() else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(directory) else {
        return Vec::new();
    };
    let stem = model_path
        .file_stem()
        .map(|value| value.to_string_lossy().to_string())
        .unwrap_or_default();

    let mut ranked: Vec<(MmprojCandidate, (usize, u8))> = entries
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().map(|kind| kind.is_file()).unwrap_or(false))
        .filter_map(|entry| {
            let path = entry.path();
            if path == model_path {
                return None;
            }
            let name = path.file_name()?.to_string_lossy().to_string();
            let lower = name.to_lowercase();
            if !lower.ends_with(".gguf") || !lower.starts_with("mmproj") {
                return None;
            }
            let size = entry.metadata().ok()?.len();
            let rank = mmproj_rank(&stem, &lower);
            Some((
                MmprojCandidate {
                    file: name,
                    path: path.to_string_lossy().to_string(),
                    size,
                },
                rank,
            ))
        })
        .collect();

    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.file.cmp(&b.0.file)));
    ranked.into_iter().map(|(candidate, _)| candidate).collect()
}

/* ------------------------------------------------------- cache management */

/// Deletes one downloaded file, refusing anything outside `dir`.
///
/// The path arrives from the UI, so it is re-checked here rather than trusted:
/// the canonicalised parent must be the canonicalised download folder and the
/// file must be a `.gguf`. Returns the number of bytes freed.
#[tauri::command]
pub fn delete_cached_model(path: String, dir: String) -> Result<u64, String> {
    let directory = std::fs::canonicalize(dir.trim())
        .map_err(|error| format!("cannot open the download folder: {error}"))?;
    let file = std::fs::canonicalize(path.trim())
        .map_err(|error| format!("cannot open that file: {error}"))?;

    if file.parent() != Some(directory.as_path()) {
        return Err("refusing to delete a file outside the download folder".to_string());
    }
    let is_gguf = file
        .extension()
        .map(|ext| ext.to_string_lossy().to_lowercase() == "gguf")
        .unwrap_or(false);
    if !is_gguf {
        return Err("only downloaded .gguf files can be deleted".to_string());
    }

    let size = std::fs::metadata(&file)
        .map(|meta| meta.len())
        .unwrap_or(0);
    std::fs::remove_file(&file)
        .map_err(|error| format!("could not delete {}: {error}", file.display()))?;
    Ok(size)
}

/// Asks the streaming loop to stop. The partial file is removed by the download
/// task itself, so this is safe to call at any time.
#[tauri::command]
pub fn cancel_model_download() -> bool {
    let was_running = !CANCEL_DOWNLOAD.swap(true, Ordering::SeqCst);
    was_running
}

/// Downloads one GGUF file, streaming progress to the UI.
///
/// The payload is written to `<name>.part` and only renamed once complete and
/// verified as a real GGUF file, so an interrupted download can never be picked
/// up by the compiler. An existing `.part` file is continued with an HTTP
/// `Range` request, so a dropped connection resumes where it stopped.
#[tauri::command]
pub async fn download_model(
    app: AppHandle,
    repo: String,
    file: String,
    target_dir: String,
) -> Result<String, String> {
    let (owner, name) = validate_repo(&repo)?;
    let trimmed_file = file.trim();
    if trimmed_file.contains("..") || !trimmed_file.to_lowercase().ends_with(".gguf") {
        return Err("only .gguf files can be downloaded".to_string());
    }

    let directory = PathBuf::from(target_dir.trim());
    if directory.as_os_str().is_empty() {
        return Err("no download folder selected".to_string());
    }
    fs_create(&directory)
        .map_err(|error| format!("cannot create {}: {error}", directory.display()))?;

    let file_name = trimmed_file
        .rsplit('/')
        .next()
        .unwrap_or(trimmed_file)
        .to_string();
    let target = directory.join(&file_name);
    let temporary = directory.join(format!("{file_name}.part"));

    CANCEL_DOWNLOAD.store(false, Ordering::SeqCst);

    let url = resolve_url(&owner, &name, trimmed_file);

    // No global timeout: a 20 GB file on a slow line must be allowed to finish.
    let client = client(None)?;
    // A leftover `.part` is the resume point: its size becomes the `Range` start,
    // so an interrupted transfer only asks for the bytes it is missing.
    let mut start = std::fs::metadata(&temporary)
        .map(|meta| meta.len())
        .unwrap_or(0);
    let mut attempt = 0;
    use futures_util::StreamExt;

    let (total, downloaded) = loop {
        attempt += 1;
        let mut request = client.get(&url);
        if start > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={start}-"));
        }
        let response = request
            .send()
            .await
            .map_err(|error| format!("cannot start the download: {error}"))?;
        let status = response.status();

        // The partial file is longer than what the server has (or the file was
        // replaced), so it cannot be continued: discard it and start over once.
        if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE && start > 0 && attempt == 1 {
            std::fs::remove_file(&temporary).ok();
            start = 0;
            continue;
        }
        if !status.is_success() {
            return Err(http_error("downloading the model", status));
        }

        let resuming = status == reqwest::StatusCode::PARTIAL_CONTENT;
        if !resuming && start > 0 {
            // The server ignored the range request; rewrite from byte zero.
            std::fs::remove_file(&temporary).ok();
            start = 0;
        }

        let total = if resuming {
            content_range_total(&response)
                .unwrap_or_else(|| response.content_length().unwrap_or(0) + start)
        } else {
            response.content_length().unwrap_or(0)
        };

        // Appending keeps the bytes already on disk when the transfer resumes.
        let mut out = if resuming {
            std::fs::OpenOptions::new()
                .append(true)
                .open(&temporary)
                .map_err(|error| format!("cannot write to {}: {error}", temporary.display()))?
        } else {
            std::fs::File::create(&temporary)
                .map_err(|error| format!("cannot write to {}: {error}", temporary.display()))?
        };

        let mut stream = response.bytes_stream();
        let mut last_emit = Instant::now()
            .checked_sub(Duration::from_secs(1))
            .unwrap_or_else(Instant::now);
        let mut downloaded = start;

        while let Some(chunk) = stream.next().await {
            if CANCEL_DOWNLOAD.load(Ordering::SeqCst) {
                drop(out);
                std::fs::remove_file(&temporary).ok();
                return Err("download cancelled".to_string());
            }
            let chunk = chunk.map_err(|error| {
                format!(
                    "the download was interrupted after {downloaded} bytes: {error}. Retry to resume from there."
                )
            })?;
            out.write_all(&chunk)
                .map_err(|error| format!("cannot write the model to disk: {error}"))?;
            downloaded += chunk.len() as u64;

            if last_emit.elapsed() >= Duration::from_millis(120) {
                last_emit = Instant::now();
                let percent = if total > 0 {
                    (downloaded as f64 / total as f64) * 100.0
                } else {
                    0.0
                };
                let _ = app.emit(
                    "model:progress",
                    ModelProgress {
                        repo: repo.clone(),
                        file: file_name.clone(),
                        downloaded,
                        total,
                        percent,
                    },
                );
            }
        }

        out.flush()
            .map_err(|error| format!("cannot flush the downloaded model: {error}"))?;
        drop(out);
        break (total, downloaded);
    };

    // A stream can end cleanly without delivering every byte; keep the partial
    // file in that case so the next attempt resumes instead of restarting.
    if total > 0 && downloaded < total {
        return Err(format!(
            "the download stopped at {} of {} bytes. Retry to resume from there.",
            downloaded, total
        ));
    }

    // A GGUF file starts with the magic "GGUF"; anything else is an error page.
    // Only four bytes are read: resumed weights can be many gigabytes.
    let magic_ok = std::fs::File::open(&temporary)
        .and_then(|mut handle| {
            let mut magic = [0u8; 4];
            handle.read_exact(&mut magic)?;
            Ok(magic == *b"GGUF")
        })
        .unwrap_or(false);
    if !magic_ok {
        std::fs::remove_file(&temporary).ok();
        return Err("the downloaded file is not a GGUF model (bad magic)".to_string());
    }

    std::fs::rename(&temporary, &target)
        .map_err(|error| format!("cannot finalise {}: {error}", target.display()))?;

    // Emit a final 100% event so the UI settles even if the last chunk was small.
    let _ = app.emit(
        "model:progress",
        ModelProgress {
            repo,
            file: file_name,
            downloaded,
            total: total.max(downloaded),
            percent: 100.0,
        },
    );

    Ok(target.to_string_lossy().to_string())
}

/* ------------------------------------------------------------------ tests */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_quant_labels() {
        assert_eq!(quant_label("Qwen3-8B-Q4_K_M.gguf"), "Q4_K_M");
        assert_eq!(quant_label("Qwen3-8B-IQ4_XS.gguf"), "IQ4_XS");
        assert_eq!(quant_label("Llama-3.2-3B-BF16.gguf"), "BF16");
        assert_eq!(quant_label("stories15M-q4_0.gguf"), "Q4_0");
        assert_eq!(quant_label("Qwen3-8B-f16.gguf"), "F16");
        // No quant suffix: fall back to the stem rather than inventing "8B".
        assert_eq!(quant_label("Qwen3-8B.gguf"), "Qwen3-8B");
    }

    #[test]
    fn parses_parameter_counts() {
        assert_eq!(parse_params("unsloth/Qwen3-8B-GGUF"), "8B");
        assert_eq!(parse_params("Qwen3.8-27B-GGUF"), "27B");
        // "15M" really is the parameter count for the tiny test model.
        assert_eq!(parse_params("stories15M-q4_0"), "15M");
        assert_eq!(parse_params("gemma-3-1b-it-GGUF"), "1B");
        // Mixture-of-experts names carry two numbers; the total wins.
        assert_eq!(parse_params("Huihui-Qwen3.6-35B-A3B-GGUF"), "35B");
        assert_eq!(parse_params("no-size-here"), "");
    }

    #[test]
    fn detects_tool_calling_from_chat_template() {
        let template = "{%- if tools %}<|im_start|>system\\n<tools></tools><tool_call>{%- endif %}";
        let (badges, signals) = detect(
            "unsloth/Qwen3-8B-GGUF",
            &["gguf".to_string()],
            "text-generation",
            "qwen3",
            Some(template),
            1,
            false,
        );
        assert!(badges.tool_use, "tool calling should be detected");
        assert!(signals.iter().any(|s| s.contains("tool calls")));
        assert!(!badges.vision);
    }

    #[test]
    fn detects_reasoning_and_vision() {
        let (badges, _) = detect(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-8B-GGUF",
            &["gguf".to_string(), "reasoning".to_string()],
            "text-generation",
            "qwen2",
            Some("{%- if message.reasoning_content %}"),
            2,
            false,
        );
        assert!(badges.reasoning);
        assert!(badges.multilingual);

        let (vision, _) = detect(
            "org/llava-v1.6-GGUF",
            &[],
            "image-text-to-text",
            "llava",
            None,
            1,
            true,
        );
        assert!(vision.vision);
    }

    #[test]
    fn rejects_bad_repository_ids() {
        assert!(validate_repo("owner/name").is_ok());
        assert!(validate_repo("owner").is_err());
        assert!(validate_repo("owner/name/extra").is_err());
        assert!(validate_repo("owner/na me").is_err());
        assert!(validate_repo("../etc/passwd").is_err());
    }

    #[test]
    fn encodes_unsafe_characters() {
        assert_eq!(encode_segment("file name.gguf"), "file%20name.gguf");
        assert_eq!(encode_segment("Q4_K_M.gguf"), "Q4_K_M.gguf");
    }

    #[test]
    fn builds_download_urls_with_subfolders() {
        assert_eq!(
            resolve_url("ggml-org", "models", "tinyllamas/stories15M-q4_0.gguf"),
            "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf?download=true"
        );
        assert!(resolve_url("org", "repo", "my model.gguf").ends_with("my%20model.gguf?download=true"));
    }

    /// Live smoke test against the real Hub. Run with:
    /// `cargo test --manifest-path src-tauri/Cargo.toml hf::tests::live -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn live_search_and_detail() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let results = runtime
            .block_on(hf_search("qwen3".to_string(), "downloads".to_string(), 5))
            .expect("search should succeed");
        assert!(!results.is_empty());
        for model in &results {
            println!(
                "{} | arch={} ctx={} size={} params={} tool={} think={} vision={}",
                model.repo,
                model.architecture,
                model.context_length,
                model.repo_size,
                model.params,
                model.badges.tool_use,
                model.badges.reasoning,
                model.badges.vision
            );
        }
        // A stable repository that publishes single-file quantisations (some
        // repos only ship sharded files, which this app deliberately skips).
        let detail = runtime
            .block_on(hf_model("unsloth/Qwen3-8B-GGUF".to_string()))
            .expect("detail should succeed");
        println!(
            "detail: {} quants={} mmproj={} licence={} signals={:?}",
            detail.repo,
            detail.quants.len(),
            detail.mmproj.len(),
            detail.license,
            detail.signals
        );
        for quant in detail.quants.iter().take(5) {
            println!("  quant {} {} {} bytes", quant.label, quant.file, quant.size);
        }
        assert!(
            !detail.quants.is_empty(),
            "should list quantisations for a single-file GGUF repo"
        );
        assert!(detail.quants.iter().any(|quant| quant.label.starts_with('Q')));

        // The constructed download URL must resolve, including a sub-folder
        // path, without downloading the weights themselves.
        let url = resolve_url("ggml-org", "models", "tinyllamas/stories15M-q4_0.gguf");
        let head = runtime.block_on(async {
            reqwest::Client::builder()
                .user_agent(USER_AGENT)
                .build()
                .unwrap()
                .head(&url)
                .send()
                .await
        });
        match head {
            Ok(response) => {
                println!("HEAD {url} -> {}", response.status());
                assert!(
                    response.status().is_success() || response.status().is_redirection(),
                    "download URL should resolve, got {}",
                    response.status()
                );
            }
            Err(error) => println!("HEAD request skipped: {error}"),
        }
    }

    /// Confirms huggingface.co honours a resumed transfer, which is what the
    /// `.part` resume path relies on. Run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn live_range_resume_is_supported() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let url = resolve_url("ggml-org", "models", "tinyllamas/stories15M-q4_0.gguf");
        const OFFSET: u64 = 1_048_576;

        let response = runtime
            .block_on(async {
                client(Some(Duration::from_secs(60)))
                    .unwrap()
                    .get(&url)
                    .header(reqwest::header::RANGE, format!("bytes={OFFSET}-"))
                    .send()
                    .await
            })
            .expect("the range request should reach huggingface.co");

        println!("range status={}", response.status());
        assert_eq!(
            response.status(),
            reqwest::StatusCode::PARTIAL_CONTENT,
            "the Hub must answer a resumed request with 206"
        );
        let total = content_range_total(&response).expect("Content-Range must carry the total size");
        assert!(total > OFFSET, "the file must be longer than the resume offset");

        let body = runtime.block_on(async { response.bytes().await }).unwrap();
        println!("resumed body={} bytes of {total} total", body.len());
        assert_eq!(
            body.len() as u64,
            total - OFFSET,
            "only the missing bytes may be transferred"
        );
    }

    fn unique_dir(tag: &str) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("lms-{tag}-{}-{id}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn ranks_the_projector_of_the_same_family_first() {
        let dir = unique_dir("mmproj");
        for name in [
            "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
            "mmproj-model-f16.gguf",
            "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf",
            "notes.txt",
        ] {
            std::fs::write(dir.join(name), b"x").unwrap();
        }

        let candidates = mmproj_candidates(
            dir.join("Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf")
                .to_string_lossy()
                .to_string(),
        );
        assert_eq!(candidates.len(), 2, "only mmproj files are candidates");
        assert!(candidates[0].file.contains("Qwen2.5-VL-7B-Instruct"));
        assert!(candidates[1].file.starts_with("mmproj-model"));

        // A folder without projectors yields nothing instead of guessing.
        let empty = unique_dir("mmproj-none");
        std::fs::write(empty.join("model.gguf"), b"x").unwrap();
        assert!(mmproj_candidates(
            empty.join("model.gguf").to_string_lossy().to_string()
        )
        .is_empty());

        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_dir_all(&empty).ok();
    }

    #[test]
    fn parses_content_range_totals() {
        assert_eq!(parse_content_range_total("bytes 100-499/500"), Some(500));
        assert_eq!(parse_content_range_total("bytes 0-0/1234"), Some(1234));
        // `*` means the server does not know the final size.
        assert_eq!(parse_content_range_total("bytes 100-499/*"), None);
        assert_eq!(parse_content_range_total("garbage"), None);
        assert_eq!(parse_content_range_total("bytes 0-1/"), None);
    }

    #[test]
    fn deletes_only_downloaded_gguf_files() {
        let dir = unique_dir("delete");
        let inside = dir.join("Qwen3-8B-Q4_K_M.gguf");
        std::fs::write(&inside, vec![0u8; 32]).unwrap();
        let other_dir = unique_dir("delete-other");
        let outside = other_dir.join("elsewhere.gguf");
        std::fs::write(&outside, b"x").unwrap();
        let notes = dir.join("notes.txt");
        std::fs::write(&notes, b"x").unwrap();

        let freed = delete_cached_model(
            inside.to_string_lossy().to_string(),
            dir.to_string_lossy().to_string(),
        )
        .expect("a downloaded file is removable");
        assert_eq!(freed, 32);
        assert!(!inside.exists());

        // Anything outside the download folder is refused, even with a .gguf name.
        assert!(delete_cached_model(
            outside.to_string_lossy().to_string(),
            dir.to_string_lossy().to_string()
        )
        .is_err());
        assert!(outside.exists());

        assert!(delete_cached_model(
            notes.to_string_lossy().to_string(),
            dir.to_string_lossy().to_string()
        )
        .is_err());

        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_dir_all(&other_dir).ok();
    }
}
