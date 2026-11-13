//! Base runtime discovery and download.
//!
//! A `.llamafile` is an APE binary with a ZIP appended. To build one we need the
//! pristine runtime from a llamafile release, so this module talks to the GitHub
//! releases API and streams the asset to the app data directory.

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager};

/// Upstream moved from `Mozilla-Ocho` to `mozilla-ai`; try the new home first
/// and fall back so the app keeps working either way.
const REPOSITORIES: [&str; 2] = ["mozilla-ai/llamafile", "Mozilla-Ocho/llamafile"];
const USER_AGENT: &str = "llamafile-studio/1.0 (+https://github.com/llamafile-studio)";

/// One downloadable asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeAsset {
    pub name: String,
    pub url: String,
    pub size: u64,
    /// Human readable note shown in the UI (`full`, `thin`, ...).
    pub variant: String,
}

/// The asset the app recommends for this machine.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeOffer {
    pub repo: String,
    pub tag: String,
    pub published_at: String,
    pub recommended: RuntimeAsset,
    pub alternatives: Vec<RuntimeAsset>,
}

#[derive(Debug, Deserialize)]
struct GhRelease {
    tag_name: String,
    #[serde(default)]
    published_at: String,
    #[serde(default)]
    assets: Vec<GhAsset>,
}

#[derive(Debug, Deserialize)]
struct GhAsset {
    name: String,
    #[serde(default)]
    browser_download_url: String,
    #[serde(default)]
    size: u64,
}

/// Progress payload for `runtime:progress` events.
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct DownloadProgress {
    downloaded: u64,
    total: u64,
    percent: f64,
}

fn http_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .timeout(Duration::from_secs(120))
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| format!("cannot create the HTTP client: {e}"))
}

/// Client for release asset downloads.
///
/// Unlike the metadata client above there is no total timeout: the full runtime
/// is ~370 MB and legitimately takes minutes on a slow line. A whole-request
/// deadline would cut the transfer off halfway, and a half-written APE still
/// starts with `MZ`, which is why the completeness check below exists.
fn download_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .connect_timeout(Duration::from_secs(20))
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| format!("cannot create the HTTP client: {e}"))
}

/// Classifies a release asset. `None` means "not a runnable Linux runtime".
fn classify(name: &str, tag: &str) -> Option<(u8, String)> {
    let lower = name.to_lowercase();
    // Reject archives, checksums and the sibling tools shipped in the release.
    if lower.ends_with(".zip")
        || lower.ends_with(".zst")
        || lower.ends_with(".sha256")
        || lower.ends_with(".json")
        || lower.ends_with(".deb")
        || lower.ends_with(".rpm")
        || lower.contains("zipalign")
        || lower.contains("whisperfile")
        || lower.contains("transcribefile")
        || lower.contains("diffusionfile")
    {
        return None;
    }
    if !lower.starts_with("llamafile-") {
        return None;
    }
    let tag_lower = tag.to_lowercase();
    let exact = format!("llamafile-{tag_lower}");
    if lower == exact {
        // The full build: every GPU backend, largest download, most compatible.
        return Some((0, "full runtime (all GPU backends)".to_string()));
    }
    if lower == format!("{exact}-thin") {
        return Some((1, "thin runtime (smaller download, no bundled GPU libraries)".to_string()));
    }
    // Anything else (arch suffixes, older naming) is a last resort.
    Some((2, "alternative runtime build".to_string()))
}

/// Returns the newest release, preferring the runtime that best fits this host.
#[tauri::command]
pub async fn latest_runtime() -> Result<RuntimeOffer, String> {
    let client = http_client()?;
    let mut last_error = String::from("no llamafile release could be reached");

    for repo in REPOSITORIES {
        let url = format!("https://api.github.com/repos/{repo}/releases/latest");
        let response = match client
            .get(&url)
            .header("Accept", "application/vnd.github+json")
            .send()
            .await
        {
            Ok(response) => response,
            Err(error) => {
                last_error = format!("network error while contacting {repo}: {error}");
                continue;
            }
        };

        if !response.status().is_success() {
            last_error = format!(
                "GitHub returned HTTP {} for {repo} (rate limited? try again shortly)",
                response.status().as_u16()
            );
            continue;
        }

        let release: GhRelease = match response.json().await {
            Ok(release) => release,
            Err(error) => {
                last_error = format!("cannot parse the {repo} release metadata: {error}");
                continue;
            }
        };

        let mut assets: Vec<(u8, RuntimeAsset)> = release
            .assets
            .iter()
            .filter_map(|asset| {
                classify(&asset.name, &release.tag_name).map(|(rank, variant)| {
                    (
                        rank,
                        RuntimeAsset {
                            name: asset.name.clone(),
                            url: asset.browser_download_url.clone(),
                            size: asset.size,
                            variant,
                        },
                    )
                })
            })
            .collect();

        if assets.is_empty() {
            last_error = format!("release {} of {repo} has no downloadable runtime", release.tag_name);
            continue;
        }

        assets.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.size.cmp(&b.1.size)));
        let recommended = assets.remove(0).1;

        return Ok(RuntimeOffer {
            repo: repo.to_string(),
            tag: release.tag_name,
            published_at: release.published_at,
            recommended,
            alternatives: assets.into_iter().map(|(_, asset)| asset).collect(),
        });
    }

    Err(last_error)
}

/// Where downloaded runtimes and build reports live.
pub fn runtime_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let base = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("cannot resolve the application data directory: {e}"))?;
    let dir = base.join("runtimes");
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    Ok(dir)
}

fn sanitise_file_name(name: &str) -> String {
    let base = Path::new(name)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "llamafile".to_string());
    let cleaned: String = base
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') { c } else { '_' })
        .collect();
    if cleaned.is_empty() || cleaned == "." || cleaned == ".." {
        "llamafile".to_string()
    } else {
        cleaned
    }
}

/// Streams a release asset into the app data directory and marks it executable.
#[tauri::command]
pub async fn download_runtime(
    app: AppHandle,
    url: String,
    file_name: String,
) -> Result<String, String> {
    // Only ever fetch over HTTPS from GitHub's own hosts.
    if !url.starts_with("https://github.com/")
        && !url.starts_with("https://objects.githubusercontent.com/")
        && !url.starts_with("https://release-assets.githubusercontent.com/")
    {
        return Err(format!("refusing to download from an untrusted host: {url}"));
    }

    let client = download_client()?;
    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("cannot start the download: {e}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "download failed with HTTP {}",
            response.status().as_u16()
        ));
    }

    let total = response.content_length().unwrap_or(0);
    let dir = runtime_dir(&app)?;
    let safe_name = sanitise_file_name(&file_name);
    let target = dir.join(&safe_name);
    // Write to a sibling then rename, so an interrupted download never leaves a
    // truncated runtime that looks valid to the compiler.
    let temporary = dir.join(format!("{safe_name}.part"));

    let mut file = std::fs::File::create(&temporary)
        .map_err(|e| format!("cannot create {}: {e}", temporary.display()))?;

    let mut downloaded: u64 = 0;
    let mut stream = response.bytes_stream();
    use futures_util::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("the download was interrupted: {e}"))?;
        file.write_all(&chunk)
            .map_err(|e| format!("cannot write the runtime to disk: {e}"))?;
        downloaded += chunk.len() as u64;

        let percent = if total > 0 {
            (downloaded as f64 / total as f64) * 100.0
        } else {
            0.0
        };
        let _ = app.emit(
            "runtime:progress",
            DownloadProgress {
                downloaded,
                total,
                percent,
            },
        );
    }

    file.flush()
        .map_err(|e| format!("cannot flush the downloaded runtime: {e}"))?;
    drop(file);

    // A body can stop early without an error - a proxy or an expiring CDN token
    // closes the connection cleanly - and the resulting file is still a valid
    // ELF/PE header with the rest of the payload missing. Accepting it makes
    // every build from this runtime produce a binary that segfaults on start, so
    // a short transfer has to fail here instead.
    if total > 0 && downloaded != total {
        std::fs::remove_file(&temporary).ok();
        return Err(format!(
            "the download stopped at {downloaded} of {total} bytes and was discarded; try again"
        ));
    }

    if downloaded < 1024 {
        // Anything this small is an error page, not a runtime.
        std::fs::remove_file(&temporary).ok();
        return Err("the downloaded file is implausibly small; the release asset may have moved".to_string());
    }

    // A real runtime is an APE with a ZIP archive in its tail; an error page or
    // a partial file cannot be packaged, no matter what its size says.
    if let Err(reason) = crate::zipwriter::verify_trailing_archive(&temporary) {
        std::fs::remove_file(&temporary).ok();
        return Err(format!(
            "the downloaded file is not a complete llamafile runtime: {reason}"
        ));
    }

    std::fs::rename(&temporary, &target)
        .map_err(|e| format!("cannot finalise {}: {e}", target.display()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o755))
            .map_err(|e| format!("cannot mark {} executable: {e}", target.display()))?;
    }

    Ok(target.to_string_lossy().to_string())
}
