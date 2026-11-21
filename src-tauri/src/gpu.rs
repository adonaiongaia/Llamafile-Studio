//! Validation of GPU backend plugins.
//!
//! llamafile does not statically link GPU support. It looks for shared libraries
//! named `ggml-(cuda|rocm|vulkan).(so|dll)` in the directory of the executable,
//! inside the llamafile bundle, in `~/.llamafile/v/<version>` and in `$HOME`.
//! Upstream publishes the prebuilt ones by appending them to the archive root
//! (`zipalign -j0 llamafile ggml-cuda.so ggml-rocm.so`), which is what the
//! compiler does with the files selected here.
//!
//! This module only inspects and validates paths: it never copies anything and
//! never decides *which* backend to use. The runtime probes them in turn and
//! falls back to the CPU when a library finds no usable device.

use std::fs;
use std::path::Path;

use serde::Serialize;

/// Prefix every plugin llamafile looks for shares.
const LIB_PREFIX: &str = "ggml-";
const LINUX_EXT: &str = ".so";
const WINDOWS_EXT: &str = ".dll";

/// One selected library, as rendered by the UI.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuLibrary {
    pub path: String,
    /// Base name, which is also the ZIP entry name.
    pub file: String,
    pub size: u64,
    pub exists: bool,
    /// False when llamafile could never load the file.
    pub valid: bool,
    pub problem: Option<String>,
    /// `cuda`, `rocm`, `vulkan`, `metal`, `amx`, `sycl` or `other`.
    pub backend: String,
    /// `linux` for `.so`, `windows` for `.dll`.
    pub platform: String,
}

struct ParsedName {
    backend: &'static str,
    platform: &'static str,
}

/// Splits `ggml-cuda.so` into its backend and platform.
///
/// The loader builds its candidate names in lowercase, which is why a
/// mixed-case file name is rejected later on: it would be bundled but never
/// found.
fn parse_name(file: &str) -> Option<ParsedName> {
    let lower = file.to_ascii_lowercase();
    let (stem, platform) = if let Some(stem) = lower.strip_suffix(LINUX_EXT) {
        (stem, "linux")
    } else if let Some(stem) = lower.strip_suffix(WINDOWS_EXT) {
        (stem, "windows")
    } else {
        return None;
    };
    let backend = stem.strip_prefix(LIB_PREFIX)?;
    if backend.is_empty() {
        return None;
    }
    let backend = match backend {
        "cuda" => "cuda",
        "rocm" | "hip" => "rocm",
        "vulkan" => "vulkan",
        "metal" => "metal",
        "amx" => "amx",
        "sycl" => "sycl",
        _ => "other",
    };
    Some(ParsedName { backend, platform })
}

/// Backend implied by a library file name, or `None` when the name is unusable.
pub fn backend_of(file: &str) -> Option<&'static str> {
    parse_name(file).map(|parsed| parsed.backend)
}

/// Describes one path without failing: problems travel in the result so the UI
/// can show every row instead of only the first broken one.
fn describe(raw: &str) -> GpuLibrary {
    let path = Path::new(raw);
    let file = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| raw.to_string());

    let parsed = parse_name(&file);
    let metadata = fs::metadata(path).ok();
    let exists = metadata.is_some();
    let size = metadata.as_ref().map(|meta| meta.len()).unwrap_or(0);
    let is_file = metadata.as_ref().map(|meta| meta.is_file()).unwrap_or(false);

    let problem = if parsed.is_none() {
        Some(
            "the file name must be ggml-<backend>.so or ggml-<backend>.dll (for example ggml-cuda.so)"
                .to_string(),
        )
    } else if !exists {
        Some("the file does not exist any more".to_string())
    } else if !is_file {
        Some("the path is not a regular file".to_string())
    } else if size == 0 {
        Some("the file is empty".to_string())
    } else if file != file.to_ascii_lowercase() {
        Some("the file name must be lowercase for llamafile to find it".to_string())
    } else {
        None
    };

    GpuLibrary {
        path: raw.to_string(),
        file,
        size,
        exists,
        valid: problem.is_none(),
        problem,
        backend: parsed
            .as_ref()
            .map(|value| value.backend.to_string())
            .unwrap_or_default(),
        platform: parsed
            .map(|value| value.platform.to_string())
            .unwrap_or_default(),
    }
}

/// Describes every selected path, in selection order.
pub fn inspect(paths: &[String]) -> Vec<GpuLibrary> {
    paths.iter().map(|raw| describe(raw)).collect()
}

/// Validates the selection for the compiler and returns the resolved libraries.
///
/// Duplicate base names are refused rather than silently collapsed: a ZIP
/// archive with two entries of the same name is ambiguous and the loader would
/// pick whichever one it finds first.
pub fn validate(paths: &[String]) -> Result<Vec<GpuLibrary>, String> {
    let libraries = inspect(paths);
    let mut problems: Vec<String> = Vec::new();
    let mut reported_duplicates: Vec<String> = Vec::new();

    for library in &libraries {
        if let Some(problem) = &library.problem {
            problems.push(format!("{}: {problem}", library.file));
        }
        let key = library.file.to_ascii_lowercase();
        let duplicate = libraries
            .iter()
            .filter(|other| other.file.to_ascii_lowercase() == key)
            .count()
            > 1;
        if duplicate && !reported_duplicates.contains(&key) {
            reported_duplicates.push(key);
            problems.push(format!(
                "{} is selected more than once; a llamafile archive cannot hold two entries with the same name",
                library.file
            ));
        }
    }

    if problems.is_empty() {
        Ok(libraries)
    } else {
        Err(problems.join("\n"))
    }
}

/// Describes the selection for the UI (sizes, backend badges, problems).
///
/// Unlike [`validate`] this never fails, so a broken or missing file still
/// renders as a row the user can remove.
#[tauri::command]
pub fn inspect_gpu_libs(paths: Vec<String>) -> Vec<GpuLibrary> {
    inspect(&paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("lms-gpu-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn classifies_backend_and_platform() {
        assert_eq!(backend_of("ggml-cuda.so"), Some("cuda"));
        assert_eq!(backend_of("ggml-cuda.dll"), Some("cuda"));
        assert_eq!(backend_of("ggml-vulkan.so"), Some("vulkan"));
        assert_eq!(backend_of("ggml-rocm.so"), Some("rocm"));
        assert_eq!(backend_of("ggml-hip.dll"), Some("rocm"));
        assert_eq!(backend_of("ggml-amx.so"), Some("amx"));
        assert_eq!(backend_of("ggml-sycl.so"), Some("sycl"));
        // Unknown suffixes are still loadable plugins.
        assert_eq!(backend_of("ggml-rpc.so"), Some("other"));
        // Anything else is not a plugin llamafile would ever look for.
        assert_eq!(backend_of("libggml-cuda.so"), None);
        assert_eq!(backend_of("ggml-cuda.so.0"), None);
        assert_eq!(backend_of("ggml-.so"), None);
        assert_eq!(backend_of("ggml-cuda.zip"), None);
    }

    #[test]
    fn accepts_a_valid_selection() {
        let dir = scratch_dir("valid");
        let cuda = dir.join("ggml-cuda.so");
        let vulkan = dir.join("ggml-vulkan.so");
        fs::write(&cuda, vec![0u8; 4096]).unwrap();
        fs::write(&vulkan, vec![0u8; 2048]).unwrap();

        let libraries = validate(&[
            cuda.to_string_lossy().into_owned(),
            vulkan.to_string_lossy().into_owned(),
        ])
        .unwrap();
        assert_eq!(libraries.len(), 2);
        assert_eq!(libraries[0].backend, "cuda");
        assert_eq!(libraries[0].platform, "linux");
        assert_eq!(libraries[0].size, 4096);
        assert!(libraries.iter().all(|library| library.valid));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_names_llamafile_would_not_load() {
        let dir = scratch_dir("names");
        let wrong_prefix = dir.join("cudallamafile.so");
        let wrong_extension = dir.join("ggml-cuda.zip");
        fs::write(&wrong_prefix, vec![0u8; 64]).unwrap();
        fs::write(&wrong_extension, vec![0u8; 64]).unwrap();

        let error = validate(&[
            wrong_prefix.to_string_lossy().into_owned(),
            wrong_extension.to_string_lossy().into_owned(),
        ])
        .unwrap_err();
        assert!(error.contains("cudallamafile.so"), "{error}");
        assert!(error.contains("ggml-cuda.zip"), "{error}");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_duplicate_entry_names() {
        let first = scratch_dir("dupe-a");
        let second = scratch_dir("dupe-b");
        let a = first.join("ggml-cuda.so");
        let b = second.join("ggml-cuda.so");
        fs::write(&a, vec![0u8; 64]).unwrap();
        fs::write(&b, vec![0u8; 64]).unwrap();

        let error = validate(&[
            a.to_string_lossy().into_owned(),
            b.to_string_lossy().into_owned(),
        ])
        .unwrap_err();
        assert!(error.contains("selected more than once"), "{error}");
        fs::remove_dir_all(&first).ok();
        fs::remove_dir_all(&second).ok();
    }

    #[test]
    fn marks_missing_files_without_failing_the_inspection() {
        let missing = std::env::temp_dir().join("lms-gpu-missing/ggml-cuda.so");
        let info = inspect(&[missing.to_string_lossy().into_owned()]);
        assert_eq!(info.len(), 1);
        assert!(!info[0].valid);
        assert!(!info[0].exists);
        assert!(info[0]
            .problem
            .as_deref()
            .unwrap_or_default()
            .contains("does not exist"));
    }
}
