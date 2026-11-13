//! End-to-end packaging test against a *real* llamafile runtime and GGUF model.
//!
//! It is gated behind environment variables so `cargo test` stays hermetic and
//! fast:
//!
//! ```sh
//! LMS_E2E_RUNTIME=/tmp/lms-e2e/llamafile-thin \
//! LMS_E2E_MODEL=/tmp/lms-e2e/stories15M-q4_0.gguf \
//! LMS_E2E_OUT=/tmp/lms-e2e \
//! cargo test --manifest-path src-tauri/Cargo.toml --test packaging_e2e -- --nocapture
//! ```
//!
//! The resulting `/tmp/lms-e2e/stories15M.llamafile` is a real, runnable
//! executable; run it with `--server` and query `/v1/models` to confirm llamafile
//! itself can read the archive this crate produced.

use std::fs::{self, OpenOptions};
use std::io::Seek;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

use llamafile_studio_lib::args::{self, CompileConfig};
use llamafile_studio_lib::zipwriter::{StoredZipWriter, ZIP_ALIGNMENT};

fn var(key: &str) -> Option<PathBuf> {
    std::env::var_os(key)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

#[test]
fn packages_a_real_runtime_and_model() {
    let (Some(runtime), Some(model)) = (var("LMS_E2E_RUNTIME"), var("LMS_E2E_MODEL")) else {
        eprintln!("LMS_E2E_RUNTIME / LMS_E2E_MODEL not set - skipping the end-to-end packaging test");
        return;
    };

    let out_dir = var("LMS_E2E_OUT").unwrap_or_else(|| PathBuf::from("/tmp/lms-e2e"));
    fs::create_dir_all(&out_dir).expect("cannot create the output directory");
    let output = out_dir.join("stories15M.llamafile");

    fs::copy(&runtime, &output).expect("cannot copy the base runtime");
    fs::set_permissions(&output, fs::Permissions::from_mode(0o755)).unwrap();

    let prompt = "You are a terse narrator.";
    let config = CompileConfig {
        base_runtime: runtime.to_string_lossy().to_string(),
        model: model.to_string_lossy().to_string(),
        output: output.to_string_lossy().to_string(),
        execution_mode: "server".to_string(),
        host: "127.0.0.1".to_string(),
        // A dedicated port keeps the test from colliding with a real server.
        port: 18080,
        gpu_layers: 0,
        ctx_size: 2048,
        threads: 4,
        system_prompt: prompt.to_string(),
        mmproj: String::new(),
        stop_tokens: vec!["<|end|>".to_string()],
        extra_args: String::new(),
    };

    let preview = args::build(&config);
    assert!(preview.errors.is_empty(), "unexpected errors: {:?}", preview.errors);
    assert_eq!(preview.lines.last().map(String::as_str), Some("..."));
    assert!(preview.lines.iter().any(|line| line == "/zip/stories15M-q4_0.gguf"));

    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&output)
        .expect("cannot reopen the output");
    file.seek(std::io::SeekFrom::End(0)).unwrap();

    let mut zip = StoredZipWriter::new(&mut file, ZIP_ALIGNMENT).expect("writer init");
    zip.add_bytes(args::ARGS_ENTRY, preview.text.as_bytes()).unwrap();
    zip.add_bytes(args::PERSONA_ENTRY, format!("{prompt}\n").as_bytes())
        .unwrap();
    let model_name = model
        .file_name()
        .expect("model has a file name")
        .to_string_lossy()
        .to_string();
    zip.add_file(&model_name, &model, &mut |_: u64| {}).unwrap();
    let summary = zip.finish().expect("cannot finalise the archive");
    drop(file);

    assert_eq!(summary.entries.len(), 3, "expected .args, persona.txt and the model");
    assert!(!summary.uses_zip64, "a 19 MB model must not need ZIP64");
    for entry in &summary.entries {
        assert_eq!(
            entry.data_offset % ZIP_ALIGNMENT,
            0,
            "{} is not page aligned",
            entry.name
        );
    }

    // The file must still end in a valid EOCD record with three entries.
    let bytes = fs::read(&output).unwrap();
    let footer = &bytes[bytes.len() - 22..];
    assert_eq!(
        u32::from_le_bytes([footer[0], footer[1], footer[2], footer[3]]),
        0x0605_4b50
    );
    assert_eq!(u16::from_le_bytes([footer[10], footer[11]]), 3);

    let mode = fs::metadata(&output).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o755, "the output must be executable");

    println!(
        "packaged {} ({} bytes, {} entries)",
        output.display(),
        summary.total_size,
        summary.entries.len()
    );
}

/// Verifies that a vision projector becomes its own archive entry and is wired
/// up from `.args`.
///
/// The projector used here is synthetic: this test asserts archive structure and
/// argument plumbing, not that a real vision model loads.
#[test]
fn packages_a_vision_projector_alongside_weights() {
    let (Some(runtime), Some(model)) = (var("LMS_E2E_RUNTIME"), var("LMS_E2E_MODEL")) else {
        eprintln!("LMS_E2E_RUNTIME / LMS_E2E_MODEL not set - skipping the projector test");
        return;
    };

    let out_dir = var("LMS_E2E_OUT").unwrap_or_else(|| PathBuf::from("/tmp/lms-e2e"));
    fs::create_dir_all(&out_dir).unwrap();

    // Larger than the 1 KiB sanity threshold used by the compiler.
    let projector = out_dir.join("mmproj-test-f16.gguf");
    fs::write(&projector, vec![0u8; 4096]).unwrap();

    let output = out_dir.join("stories15M-vision.llamafile");
    fs::copy(&runtime, &output).unwrap();
    fs::set_permissions(&output, fs::Permissions::from_mode(0o755)).unwrap();

    let config = CompileConfig {
        base_runtime: runtime.to_string_lossy().to_string(),
        model: model.to_string_lossy().to_string(),
        output: output.to_string_lossy().to_string(),
        execution_mode: "server".to_string(),
        host: "127.0.0.1".to_string(),
        // Distinct from the other test so both can run in one session.
        port: 18081,
        gpu_layers: 0,
        ctx_size: 2048,
        threads: 4,
        system_prompt: String::new(),
        mmproj: projector.to_string_lossy().to_string(),
        stop_tokens: Vec::new(),
        extra_args: String::new(),
    };

    let preview = args::build(&config);
    assert!(preview.errors.is_empty(), "unexpected errors: {:?}", preview.errors);
    assert!(preview.embeds_mmproj);
    assert!(preview.lines.iter().any(|line| line == "--mmproj"));
    assert!(preview
        .lines
        .iter()
        .any(|line| line == "/zip/mmproj-test-f16.gguf"));
    assert_eq!(preview.lines.last().map(String::as_str), Some("..."));

    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&output)
        .unwrap();
    file.seek(std::io::SeekFrom::End(0)).unwrap();

    let mut zip = StoredZipWriter::new(&mut file, ZIP_ALIGNMENT).unwrap();
    zip.add_bytes(args::ARGS_ENTRY, preview.text.as_bytes()).unwrap();
    let model_name = model
        .file_name()
        .expect("model has a file name")
        .to_string_lossy()
        .to_string();
    zip.add_file(&model_name, &model, &mut |_: u64| {}).unwrap();
    zip.add_file("mmproj-test-f16.gguf", &projector, &mut |_: u64| {})
        .unwrap();
    let summary = zip.finish().unwrap();
    drop(file);

    assert_eq!(summary.entries.len(), 3, "args + weights + projector");
    let projector_entry = &summary.entries[2];
    assert_eq!(projector_entry.name, "mmproj-test-f16.gguf");
    assert_eq!(projector_entry.size, 4096);
    assert_eq!(projector_entry.data_offset % ZIP_ALIGNMENT, 0);
    assert!(projector_entry.data_offset > summary.entries[1].data_offset);

    let bytes = fs::read(&output).unwrap();
    let footer = &bytes[bytes.len() - 22..];
    assert_eq!(u16::from_le_bytes([footer[10], footer[11]]), 3);

    println!(
        "packaged projector into {} ({} bytes, {} entries)",
        output.display(),
        summary.total_size,
        summary.entries.len()
    );
}
