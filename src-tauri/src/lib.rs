//! Llamafile Studio - a visual compiler for portable llamafile executables.
//!
//! The whole packaging pipeline lives here: validation, the `.args` manifest, the
//! APE + ZIP assembly and the post-write verification. The frontend is a thin
//! view over these commands, so everything that touches the filesystem is
//! auditable from one place.
//!
//! ```text
//!  base runtime (APE)  ─┐
//!  .args               ─┼─▶  ZIP (stored + page aligned)  ─▶  chmod +x  ─▶  verify
//!  model.gguf          ─┤
//!  mmproj.gguf         ─┤   (optional vision projector)
//!  persona.txt         ─┘
//! ```

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::Serialize;
use tauri::{AppHandle, Emitter};

pub mod args;
pub mod hf;
mod runtime;
pub mod zipwriter;

use args::{ArgsPreview, CompileConfig};
use zipwriter::{verify_trailing_archive, StoredZipWriter, ZIP_ALIGNMENT};

/* --------------------------------------------------------------- system info */

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemInfo {
    logical_cores: u32,
    physical_cores: u32,
    /// Physical cores minus two, clamped to at least one: leaves headroom for
    /// the desktop session and the inference server loop itself.
    suggested_threads: u32,
    total_memory_gb: u64,
    architecture: String,
    os: String,
    page_size: u64,
}

/// Reads MemTotal from /proc/meminfo. Returns 0 when unavailable.
fn total_memory_gb() -> u64 {
    let Ok(text) = fs::read_to_string("/proc/meminfo") else {
        return 0;
    };
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb: u64 = rest
                .trim()
                .trim_end_matches("kB")
                .trim()
                .parse()
                .unwrap_or(0);
            return kb / 1024 / 1024;
        }
    }
    0
}

/// Hardware metrics for the "CPU threads" optimiser and the memory hints.
#[tauri::command]
fn system_info() -> SystemInfo {
    let logical = num_cpus::get().max(1) as u32;
    let physical = num_cpus::get_physical().max(1) as u32;
    SystemInfo {
        logical_cores: logical,
        physical_cores: physical,
        suggested_threads: physical.saturating_sub(2).max(1),
        total_memory_gb: total_memory_gb(),
        architecture: std::env::consts::ARCH.to_string(),
        os: format!("{} {}", std::env::consts::OS, std::env::consts::FAMILY),
        // The host's page size, which is not the archive alignment: this number
        // is only shown in the hardware line of the UI.
        page_size: 4096,
    }
}

/// Metadata about a path, used to render live badges in the UI without giving
/// the webview direct filesystem access.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PathInfo {
    exists: bool,
    is_file: bool,
    size: u64,
    extension: String,
    looks_like_llamafile: bool,
}

/// Stats a path and, for candidate runtimes, checks the APE signature.
#[tauri::command]
fn inspect_path(path: String) -> PathInfo {
    let target = PathBuf::from(&path);
    let metadata = fs::metadata(&target).ok();
    let is_file = metadata.as_ref().map(|m| m.is_file()).unwrap_or(false);
    let size = metadata.as_ref().map(|m| m.len()).unwrap_or(0);
    let extension = target
        .extension()
        .map(|ext| ext.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    let mut looks_like_llamafile = false;
    if is_file {
        let mut magic = [0u8; 2];
        if let Ok(mut file) = File::open(&target) {
            if file.read_exact(&mut magic).is_ok() {
                looks_like_llamafile = &magic == b"MZ" || &magic == b"#!";
            }
        }
    }

    PathInfo {
        exists: metadata.is_some(),
        is_file,
        size,
        extension,
        looks_like_llamafile,
    }
}

/* ------------------------------------------------------------------ dialogs */

/// Converts a dialog path into a plain string, ignoring non-local URLs.
fn file_path_to_string(path: tauri_plugin_dialog::FilePath) -> Option<String> {
    path.into_path()
        .ok()
        .map(|path| path.to_string_lossy().into_owned())
}

/// Opens the native GTK folder chooser (used for the model download target).
#[tauri::command]
async fn pick_folder(app: AppHandle, start_in: Option<String>) -> Result<Option<String>, String> {
    let handle = app.clone();
    let picked = tauri::async_runtime::spawn_blocking(move || {
        use tauri_plugin_dialog::DialogExt;

        let mut builder = handle.dialog().file();
        if let Some(dir) = start_in.as_deref() {
            let path = Path::new(dir);
            if path.is_dir() {
                builder = builder.set_directory(path);
            }
        }
        builder.blocking_pick_folder().and_then(file_path_to_string)
    })
    .await
    .map_err(|error| format!("the folder dialog could not be opened: {error}"))?;

    Ok(picked)
}

/// Opens the native GTK file chooser.
///
/// `kind` selects the filters: `runtime` (any file), `model` (`.gguf` only) or
/// `output` (`.llamafile`). `save` switches to the save dialog, in which case the
/// OS handles the "overwrite?" confirmation for us.
#[tauri::command]
async fn pick_path(
    app: AppHandle,
    kind: String,
    save: bool,
    suggested_name: Option<String>,
    start_in: Option<String>,
) -> Result<Option<String>, String> {
    let handle = app.clone();
    let picked = tauri::async_runtime::spawn_blocking(move || {
        use tauri_plugin_dialog::DialogExt;

        let dialog = handle.dialog();
        let mut builder = dialog.file();
        builder = match kind.as_str() {
            "model" => builder.add_filter("GGUF model", &["gguf"]),
            "output" => builder
                .add_filter("llamafile executable", &["llamafile"])
                .add_filter("All files", &["*"]),
            _ => builder
                .add_filter("llamafile runtime", &["*"])
                .add_filter("All files", &["*"]),
        };
        if let Some(name) = suggested_name.as_deref() {
            if !name.is_empty() {
                builder = builder.set_file_name(name);
            }
        }
        if let Some(dir) = start_in.as_deref() {
            let path = Path::new(dir);
            if path.is_dir() {
                builder = builder.set_directory(path);
            }
        }

        let result = if save {
            builder.blocking_save_file()
        } else {
            builder.blocking_pick_file()
        };
        result.and_then(file_path_to_string)
    })
    .await
    .map_err(|error| format!("the file dialog could not be opened: {error}"))?;

    Ok(picked)
}

/* ------------------------------------------------------------ args preview */

/// Pure preview of the `.args` file, called on every UI edit.
#[tauri::command]
fn preview_args(config: CompileConfig) -> ArgsPreview {
    args::build(&config)
}

/* ---------------------------------------------------------------- progress */

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompileProgress {
    stage: String,
    message: String,
    percent: f64,
    bytes_done: u64,
    bytes_total: u64,
}

/// Rate-limited progress emitter. A multi-gigabyte copy produces thousands of
/// chunks; flooding the IPC bridge would stall the webview, so updates are
/// coalesced to roughly eleven per second unless forced.
struct Reporter {
    app: AppHandle,
    total: u64,
    done: u64,
    last_emit: Instant,
    stage: &'static str,
}

impl Reporter {
    fn new(app: AppHandle, total: u64) -> Self {
        let last_emit = Instant::now()
            .checked_sub(Duration::from_secs(1))
            .unwrap_or_else(Instant::now);
        Self {
            app,
            total,
            done: 0,
            last_emit,
            stage: "validating",
        }
    }

    fn emit(&mut self, message: &str, force: bool) {
        let now = Instant::now();
        if !force && now.duration_since(self.last_emit) < Duration::from_millis(90) {
            return;
        }
        self.last_emit = now;
        let percent = if self.total == 0 {
            0.0
        } else {
            (self.done as f64 / self.total as f64) * 100.0
        };
        let _ = self.app.emit(
            "compile:progress",
            CompileProgress {
                stage: self.stage.to_string(),
                message: message.to_string(),
                percent: percent.clamp(0.0, 100.0),
                bytes_done: self.done,
                bytes_total: self.total,
            },
        );
    }

    fn stage(&mut self, stage: &'static str, message: &str) {
        self.stage = stage;
        self.emit(message, true);
    }

    fn advance(&mut self, bytes: u64) {
        self.done += bytes;
        self.emit("", false);
    }
}

/* ---------------------------------------------------------------- reporting */

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EntryReport {
    name: String,
    size: u64,
    data_offset: u64,
    page_aligned: bool,
    zip64: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CompileReport {
    output_path: String,
    output_size: u64,
    entries: Vec<EntryReport>,
    duration_ms: u64,
    args_text: String,
    warnings: Vec<String>,
    verified: bool,
    uses_zip64: bool,
    executable: bool,
    log: Vec<String>,
}

/* --------------------------------------------------------------- validation */

fn validate_paths(config: &CompileConfig) -> Result<Vec<String>, String> {
    let mut warnings = Vec::new();

    if config.base_runtime.trim().is_empty() {
        return Err("no base runtime selected".to_string());
    }
    if config.model.trim().is_empty() {
        return Err("no GGUF model selected".to_string());
    }
    if config.output.trim().is_empty() {
        return Err("no output file selected".to_string());
    }

    let base = PathBuf::from(config.base_runtime.trim());
    let model = PathBuf::from(config.model.trim());
    let output = PathBuf::from(config.output.trim());

    if !base.is_file() {
        return Err(format!(
            "the base runtime {} does not exist or is not a file",
            base.display()
        ));
    }
    let base_size = fs::metadata(&base)
        .map_err(|e| format!("cannot inspect {}: {e}", base.display()))?
        .len();
    if base_size < 1024 {
        return Err(format!(
            "{} is only {base_size} bytes - that is not a llamafile runtime",
            base.display()
        ));
    }

    if !model.is_file() {
        return Err(format!(
            "the model {} does not exist or is not a file",
            model.display()
        ));
    }
    let model_size = fs::metadata(&model)
        .map_err(|e| format!("cannot inspect {}: {e}", model.display()))?
        .len();
    if model_size < 1024 {
        return Err(format!(
            "{} is only {model_size} bytes - that is not a usable GGUF model",
            model.display()
        ));
    }
    if !config.model.trim().to_lowercase().ends_with(".gguf") {
        warnings.push(
            "the selected model does not use the .gguf extension; llamafile may refuse to load it"
                .to_string(),
        );
    }

    // Optional vision projector: validated exactly like the model, but never
    // required. It is embedded as a second archive entry and referenced from
    // `.args` with `--mmproj /zip/<name>`.
    let mmproj = config.mmproj.trim();
    let mmproj_path = if mmproj.is_empty() {
        None
    } else {
        let candidate = PathBuf::from(mmproj);
        if !candidate.is_file() {
            return Err(format!(
                "the vision projector {} does not exist or is not a file",
                candidate.display()
            ));
        }
        let size = fs::metadata(&candidate)
            .map_err(|e| format!("cannot inspect {}: {e}", candidate.display()))?
            .len();
        if size < 1024 {
            return Err(format!(
                "{} is only {size} bytes - that is not a usable mmproj file",
                candidate.display()
            ));
        }
        if !mmproj.to_lowercase().ends_with(".gguf") {
            warnings.push(
                "the vision projector does not use the .gguf extension; llamafile may refuse to load it"
                    .to_string(),
            );
        }
        Some(candidate)
    };

    if output.as_os_str().is_empty() {
        return Err("the output path is invalid".to_string());
    }
    if output.is_dir() {
        return Err(format!(
            "the output path {} is a directory",
            output.display()
        ));
    }
    // Refuse to clobber the inputs. Comparing canonicalised paths catches the
    // usual mistakes (symlinks, `./` prefixes, hard links in the same dir).
    let normalise = |path: &Path| -> PathBuf {
        fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
    };
    if normalise(&output) == normalise(&base) {
        return Err("the output file must differ from the base runtime".to_string());
    }
    if normalise(&output) == normalise(&model) {
        return Err("the output file must differ from the GGUF model".to_string());
    }
    if let Some(projector) = &mmproj_path {
        if normalise(&output) == normalise(projector) {
            return Err("the output file must differ from the vision projector".to_string());
        }
        if normalise(projector) == normalise(&model) {
            return Err("the vision projector must differ from the model".to_string());
        }
    }

    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
        }
    }

    // A llamafile runtime is an APE **with a PKZIP archive in its tail**, and
    // both halves are load-bearing: the APE header is what the kernel (or the
    // shell) executes, the archive is what makes it a llamafile. Checking only
    // the magic lets a truncated download through - the header survives a
    // cut-off transfer, the archive does not - and every binary built on top of
    // such a file segfaults before printing a single line.
    let mut magic = [0u8; 2];
    let magic_ok = File::open(&base)
        .and_then(|mut file| file.read_exact(&mut magic))
        .map(|_| &magic == b"MZ" || &magic == b"#!")
        .unwrap_or(false);
    if !magic_ok {
        return Err(format!(
            "{} is not a llamafile runtime: it does not start with an APE (MZ) or shell (#!) header",
            base.display()
        ));
    }
    verify_trailing_archive(&base).map_err(|reason| {
        format!(
            "{} is not a usable base runtime: {reason}. Download it again from the runtime panel, or pick the release binary itself",
            base.display()
        )
    })?;

    if !output
        .extension()
        .map(|ext| ext.eq_ignore_ascii_case("llamafile"))
        .unwrap_or(false)
    {
        warnings.push(
            "the output file has no .llamafile extension; you can still run it, but it is easier to recognise with one"
                .to_string(),
        );
    }

    Ok(warnings)
}

/* ------------------------------------------------------------ verification */

/// Cheap structural check: the file must end in an EOCD record whose entry count
/// matches what we wrote. Catches truncated writes and full disks immediately.
fn verify_archive(path: &Path, expected_entries: usize) -> Result<bool, String> {
    let mut file =
        File::open(path).map_err(|e| format!("cannot reopen the output for verification: {e}"))?;
    let length = file
        .metadata()
        .map_err(|e| format!("cannot stat the output file: {e}"))?
        .len();
    if length < 22 {
        return Ok(false);
    }
    file.seek(SeekFrom::End(-22))
        .map_err(|e| format!("cannot seek within the output file: {e}"))?;
    let mut footer = [0u8; 22];
    file.read_exact(&mut footer)
        .map_err(|e| format!("cannot read the end of the output file: {e}"))?;
    let signature = u32::from_le_bytes([footer[0], footer[1], footer[2], footer[3]]);
    if signature != 0x0605_4b50 {
        return Ok(false);
    }
    let count = u16::from_le_bytes([footer[10], footer[11]]) as usize;
    Ok(count == expected_entries)
}

/* -------------------------------------------------------------- the pipeline */

/// Copies the base runtime into the output path while reporting progress.
fn copy_runtime(base: &Path, output: &Path, reporter: &mut Reporter) -> Result<u64, String> {
    let mut source =
        File::open(base).map_err(|e| format!("cannot open {}: {e}", base.display()))?;
    let expected = source
        .metadata()
        .map_err(|e| format!("cannot stat {}: {e}", base.display()))?
        .len();
    let mut destination =
        File::create(output).map_err(|e| format!("cannot create {}: {e}", output.display()))?;

    let mut buffer = vec![0u8; 4 * 1024 * 1024];
    let mut copied = 0u64;
    loop {
        let read = source
            .read(&mut buffer)
            .map_err(|e| format!("read error while copying the runtime: {e}"))?;
        if read == 0 {
            break;
        }
        destination
            .write_all(&buffer[..read])
            .map_err(|e| format!("write error while copying the runtime: {e}"))?;
        copied += read as u64;
        reporter.advance(read as u64);
    }
    destination
        .flush()
        .map_err(|e| format!("cannot flush the output file: {e}"))?;
    if copied != expected {
        return Err(format!(
            "the base runtime changed while it was being copied (expected {expected} bytes, read {copied})"
        ));
    }
    Ok(copied)
}

fn run_pipeline(
    app: AppHandle,
    config: CompileConfig,
    preview: ArgsPreview,
    started: Instant,
) -> Result<CompileReport, String> {
    let base = PathBuf::from(config.base_runtime.trim());
    let model = PathBuf::from(config.model.trim());
    let output = PathBuf::from(config.output.trim());

    let base_size = fs::metadata(&base).map_err(|e| e.to_string())?.len();
    let model_size = fs::metadata(&model).map_err(|e| e.to_string())?.len();
    let mmproj = config.mmproj.trim();
    let mmproj_path = if mmproj.is_empty() {
        None
    } else {
        Some(PathBuf::from(mmproj))
    };
    let mmproj_size = match &mmproj_path {
        Some(path) => fs::metadata(path).map_err(|e| e.to_string())?.len(),
        None => 0,
    };
    let persona = config.system_prompt.trim().to_string();
    let persona_bytes_len = if preview.embeds_persona {
        persona.len() as u64 + 1
    } else {
        0
    };
    let total =
        base_size + model_size + mmproj_size + preview.text.len() as u64 + persona_bytes_len;

    let mut reporter = Reporter::new(app.clone(), total);
    let mut log: Vec<String> = Vec::new();
    let mut warnings = preview.warnings.clone();

    reporter.stage("copying", "Copying the base llamafile runtime…");
    log.push(format!("Base runtime: {}", base.display()));
    log.push(format!("Model: {} ({})", model.display(), human_size(model_size)));
    if let Some(path) = &mmproj_path {
        log.push(format!(
            "Vision projector: {} ({})",
            path.display(),
            human_size(mmproj_size)
        ));
    }
    let copied = copy_runtime(&base, &output, &mut reporter)?;
    log.push(format!("Copied {copied} bytes of runtime"));

    reporter.stage("packing", "Writing the .args manifest…");
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&output)
        .map_err(|e| format!("cannot reopen {} for appending: {e}", output.display()))?;
    file.seek(SeekFrom::End(0))
        .map_err(|e| format!("cannot seek to the end of the output file: {e}"))?;

    let mut zip = StoredZipWriter::new(&mut file, ZIP_ALIGNMENT)?;
    zip.add_bytes(args::ARGS_ENTRY, preview.text.as_bytes())?;
    reporter.advance(preview.text.len() as u64);
    log.push(format!(".args written ({} arguments)", preview.lines.len()));

    if preview.embeds_persona {
        let mut persona_payload = persona.clone();
        persona_payload.push('\n');
        zip.add_bytes(args::PERSONA_ENTRY, persona_payload.as_bytes())?;
        reporter.advance(persona_bytes_len);
        log.push(if preview.persona_applied {
            "persona.txt embedded and applied via -f (chat system prompt)".to_string()
        } else {
            "persona.txt embedded (server mode applies prompts per request)".to_string()
        });
    }

    let model_name = model
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .ok_or_else(|| "the model path has no file name".to_string())?;
    reporter.stage(
        "packing",
        "Embedding the model weights (stored, page aligned)…",
    );
    {
        let mut on_chunk = |bytes: u64| reporter.advance(bytes);
        zip.add_file(&model_name, &model, &mut on_chunk)?;
    }
    log.push(format!("{model_name} appended uncompressed"));

    if let Some(projector) = &mmproj_path {
        let projector_name = projector
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .ok_or_else(|| "the projector path has no file name".to_string())?;
        reporter.stage("packing", "Embedding the vision projector…");
        {
            let mut on_chunk = |bytes: u64| reporter.advance(bytes);
            zip.add_file(&projector_name, projector, &mut on_chunk)?;
        }
        log.push(format!("{projector_name} appended (vision projector)"));
    }

    reporter.stage("finalising", "Writing the ZIP central directory…");
    let summary = zip.finish()?;
    drop(file);
    log.push(format!(
        "Archive finalised: {} bytes, central directory at {}, ZIP64={}",
        summary.total_size, summary.central_directory_offset, summary.uses_zip64
    ));

    // --- permissions ------------------------------------------------------
    #[cfg(unix)]
    let executable = {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&output, fs::Permissions::from_mode(0o755))
            .map_err(|e| format!("cannot make {} executable: {e}", output.display()))?;
        true
    };
    #[cfg(not(unix))]
    let executable = false;
    log.push("Permissions set to 0755 (executable)".to_string());

    // --- verification -----------------------------------------------------
    reporter.stage("verifying", "Verifying the generated executable…");
    let verified = verify_archive(&output, summary.entries.len())?;
    if !verified {
        warnings.push(
            "the generated file did not pass the end-of-archive check - try rebuilding"
                .to_string(),
        );
    }
    log.push(format!(
        "Verification: {}",
        if verified { "end-of-archive record intact" } else { "FAILED" }
    ));

    let output_size = fs::metadata(&output)
        .map_err(|e| format!("cannot stat the generated file: {e}"))?
        .len();

    reporter.stage("done", "Llamafile compiled successfully");
    reporter.done = reporter.total;
    reporter.emit("Llamafile compiled successfully", true);

    let entries = summary
        .entries
        .iter()
        .map(|entry| EntryReport {
            name: entry.name.clone(),
            size: entry.size,
            data_offset: entry.data_offset,
            page_aligned: entry.data_offset % ZIP_ALIGNMENT == 0,
            zip64: entry.zip64,
        })
        .collect::<Vec<_>>();

    for entry in &entries {
        log.push(format!(
            "Entry {:>16} {} ({} bytes at offset {}, page aligned: {})",
            entry.name,
            human_size(entry.size),
            entry.size,
            entry.data_offset,
            entry.page_aligned
        ));
    }

    Ok(CompileReport {
        output_path: output.to_string_lossy().to_string(),
        output_size,
        entries,
        duration_ms: started.elapsed().as_millis() as u64,
        args_text: preview.text,
        warnings,
        verified,
        uses_zip64: summary.uses_zip64,
        executable,
        log,
    })
}

fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

/// The one command that produces a `.llamafile`.
///
/// Heavy work runs on Tauri's blocking pool so the UI thread (and the progress
/// bar) stay responsive while gigabytes are copied.
#[tauri::command]
async fn compile(app: AppHandle, config: CompileConfig) -> Result<CompileReport, String> {
    let preview = args::build(&config);
    if !preview.errors.is_empty() {
        return Err(preview.errors.join("\n"));
    }
    let mut warnings = validate_paths(&config)?;
    let started = Instant::now();

    let mut report = tauri::async_runtime::spawn_blocking(move || {
        run_pipeline(app, config, preview, started)
    })
    .await
    .map_err(|error| format!("the packaging task crashed: {error}"))??;

    // Validation warnings come first so the UI can show them in order.
    warnings.extend(report.warnings);
    report.warnings = warnings;
    Ok(report)
}

/// Reveals the generated file in the system file manager (xdg-open on Linux).
#[tauri::command]
fn reveal_in_file_manager(path: String) -> Result<(), String> {
    let target = PathBuf::from(&path);
    let directory = if target.is_dir() {
        target
    } else {
        target
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| "the path has no parent directory".to_string())?
    };
    std::process::Command::new("xdg-open")
        .arg(&directory)
        .spawn()
        .map_err(|e| format!("cannot open the file manager: {e}"))?;
    Ok(())
}

/// Persists a build report next to the executable. Handy for support tickets and
/// for keeping the exact `.args` used at build time.
#[tauri::command]
fn save_build_report(output_path: String, contents: String) -> Result<String, String> {
    let output = PathBuf::from(&output_path);
    let stem = output
        .file_stem()
        .map(|stem| stem.to_string_lossy().to_string())
        .unwrap_or_else(|| "llamafile".to_string());
    let report_path = output
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("{stem}.build.txt"));
    fs::write(&report_path, contents)
        .map_err(|e| format!("cannot write {}: {e}", report_path.display()))?;
    Ok(report_path.to_string_lossy().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("lms-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Writes a llamafile-shaped file: an APE header plus, optionally, the ZIP
    /// archive every complete runtime carries in its tail.
    fn fake_runtime(path: &Path, with_archive: bool) {
        let mut file = File::create(path).unwrap();
        file.write_all(b"MZqFpD='\n").unwrap();
        file.write_all(&vec![0x41u8; 4096]).unwrap();
        drop(file);
        if with_archive {
            let mut file = OpenOptions::new().read(true).write(true).open(path).unwrap();
            file.seek(SeekFrom::End(0)).unwrap();
            let mut zip = StoredZipWriter::new(&mut file, ZIP_ALIGNMENT).unwrap();
            zip.add_bytes(".cosmo", b"stub").unwrap();
            zip.finish().unwrap();
        }
    }

    fn config(runtime: &Path, model: &Path, output: &Path) -> CompileConfig {
        CompileConfig {
            base_runtime: runtime.to_string_lossy().into_owned(),
            model: model.to_string_lossy().into_owned(),
            output: output.to_string_lossy().into_owned(),
            execution_mode: "chat".to_string(),
            host: "127.0.0.1".to_string(),
            port: 8080,
            gpu_layers: 99,
            ctx_size: 4096,
            threads: 4,
            system_prompt: String::new(),
            mmproj: String::new(),
            stop_tokens: Vec::new(),
            extra_args: String::new(),
        }
    }

    #[test]
    fn rejects_a_truncated_base_runtime() {
        let dir = scratch_dir("validate");
        let runtime = dir.join("runtime.part");
        let model = dir.join("model.gguf");
        let output = dir.join("out.llamafile");
        std::fs::write(&model, vec![0u8; 4096]).unwrap();

        // A cut-off download keeps the MZ header but loses the archive tail; it
        // used to be accepted and produced a binary that segfaulted on start.
        fake_runtime(&runtime, false);
        let error = validate_paths(&config(&runtime, &model, &output)).unwrap_err();
        assert!(error.contains("not a usable base runtime"), "{error}");

        // The same file once its archive arrived is a valid runtime.
        fake_runtime(&runtime, true);
        assert!(validate_paths(&config(&runtime, &model, &output)).is_ok());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_a_base_runtime_that_is_not_an_ape() {
        let dir = scratch_dir("validate-magic");
        let runtime = dir.join("runtime.zip");
        let model = dir.join("model.gguf");
        let output = dir.join("out.llamafile");
        std::fs::write(&model, vec![0u8; 4096]).unwrap();
        // A release archive looks like a zip but cannot be executed.
        let mut bytes = b"PK\x03\x04 not an executable".to_vec();
        bytes.resize(4096, 0);
        std::fs::write(&runtime, &bytes).unwrap();
        let error = validate_paths(&config(&runtime, &model, &output)).unwrap_err();
        assert!(error.contains("does not start with an APE"), "{error}");
        std::fs::remove_dir_all(&dir).ok();
    }
}

/* --------------------------------------------------------------- app entry */

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        // Native file dialogs, filesystem access, opener and shell integration.
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            system_info,
            inspect_path,
            pick_path,
            pick_folder,
            preview_args,
            compile,
            reveal_in_file_manager,
            save_build_report,
            runtime::latest_runtime,
            runtime::download_runtime,
            hf::hf_search,
            hf::hf_model,
            hf::download_model,
            hf::cancel_model_download,
            hf::default_model_dir,
            hf::cached_models,
            hf::mmproj_candidates,
            hf::delete_cached_model,
        ])
        .run(tauri::generate_context!())
        .expect("failed to start Llamafile Studio");
}
