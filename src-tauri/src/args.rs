//! `.args` generation and validation.
//!
//! llamafile reads a file literally named `.args` from the root of the embedded
//! ZIP archive. It is *not* a shell command line: it contains exactly one
//! argument per line, and the special token `...` marks where arguments passed
//! at runtime are spliced in. That placement is what makes a baked configuration
//! still overridable from the terminal:
//!
//! ```text
//! --chat
//! -m
//! /zip/tinyllama.gguf
//! -ngl
//! 99
//! ...
//! ```
//!
//! Running `./model.llamafile -c 8192` therefore appends `-c 8192` *after* the
//! baked flags, and llama.cpp's last-wins parsing honours the override. The `...`
//! line is mandatory and is always emitted last.

use serde::{Deserialize, Serialize};

/// Files packaged inside a llamafile are addressed through this virtual prefix.
pub const ZIP_PREFIX: &str = "/zip/";
/// Name of the generated argument file inside the archive.
pub const ARGS_ENTRY: &str = ".args";
/// Name of the embedded persona/system prompt.
pub const PERSONA_ENTRY: &str = "persona.txt";

/// Upper bound from the hardware panel: 120 layers covers every current model.
pub const MAX_GPU_LAYERS: u32 = 120;

/// The full configuration coming from the UI. `camelCase` on the wire keeps the
/// JavaScript side idiomatic.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CompileConfig {
    pub base_runtime: String,
    pub model: String,
    pub output: String,
    /// `"server"` or `"chat"`.
    pub execution_mode: String,
    pub host: String,
    pub port: u16,
    pub gpu_layers: u32,
    pub ctx_size: u32,
    pub threads: u32,
    pub system_prompt: String,
    /// Optional multimodal projector (`mmproj`) GGUF used by vision models.
    #[serde(default)]
    pub mmproj: String,
    #[serde(default)]
    pub stop_tokens: Vec<String>,
    /// Free-form advanced flags, one argument per line.
    #[serde(default)]
    pub extra_args: String,
}

/// What the backend produced from a config, before anything touches the disk.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ArgsPreview {
    pub lines: Vec<String>,
    pub text: String,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    /// Whether `persona.txt` should be embedded in the archive.
    pub embeds_persona: bool,
    /// Whether the persona is actually applied at launch (chat mode only).
    pub persona_applied: bool,
    /// Whether a vision projector should be embedded alongside the weights.
    pub embeds_mmproj: bool,
}

/// Sanitises a user supplied stop token.
///
/// llama.cpp's `-r/--reverse-prompt` needs a literal string, and since `.args`
/// is line separated a token containing a newline cannot be represented at all.
/// We report those instead of silently writing a corrupted argument file.
fn normalise_stop_token(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    // Users naturally type the two characters `\n`; expand escapes that map to
    // printable ASCII but never allow real newlines into the file.
    let expanded = trimmed
        .replace("\\t", "\t")
        .replace("\\r", "\r")
        .replace("\\n", "\n");
    // Newlines are reported by the caller (they cannot be encoded) instead of
    // being swallowed here, even when the token is nothing but a newline.
    Some(expanded)
}

/// Builds the `.args` payload plus validation diagnostics.
///
/// This is pure: it never reads or writes files, so the UI can call it on every
/// keystroke to show a live preview of what will be baked in.
pub fn build(config: &CompileConfig) -> ArgsPreview {
    let mut lines: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let mut errors: Vec<String> = Vec::new();

    let mode = config.execution_mode.trim().to_lowercase();
    let is_server = match mode.as_str() {
        "server" => true,
        "chat" => false,
        other => {
            errors.push(format!(
                "unknown execution mode \"{other}\" (expected \"server\" or \"chat\")"
            ));
            false
        }
    };

    // --- model ------------------------------------------------------------
    let model_path = config.model.trim();
    let model_name = if model_path.is_empty() {
        errors.push("select a GGUF model to embed".to_string());
        String::new()
    } else {
        let path = std::path::Path::new(model_path);
        match path.file_name() {
            Some(name) => name.to_string_lossy().to_string(),
            None => {
                errors.push(format!("the model path \"{model_path}\" has no file name"));
                String::new()
            }
        }
    };
    if !model_name.is_empty() && !model_name.to_lowercase().ends_with(".gguf") {
        warnings.push(format!(
            "\"{model_name}\" does not end in .gguf - llamafile will likely refuse to load it"
        ));
    }

    // --- mode -------------------------------------------------------------
    lines.push(if is_server { "--server" } else { "--chat" }.to_string());

    if !model_name.is_empty() {
        lines.push("-m".to_string());
        lines.push(format!("{ZIP_PREFIX}{model_name}"));
    }

    // --- vision projector -------------------------------------------------
    // `-mm/--mmproj` is registered by every llamafile mode (server, chat and the
    // default combined mode), so it is safe to bake in regardless of the
    // execution mode chosen above.
    let mmproj_path = config.mmproj.trim();
    let mut embeds_mmproj = false;
    if !mmproj_path.is_empty() {
        match std::path::Path::new(mmproj_path).file_name() {
            Some(name) => {
                let mmproj_name = name.to_string_lossy().to_string();
                if !mmproj_name.to_lowercase().ends_with(".gguf") {
                    warnings.push(format!(
                        "\"{mmproj_name}\" does not end in .gguf - llamafile may refuse to load the projector"
                    ));
                }
                if !model_name.is_empty() && mmproj_name == model_name {
                    errors.push(
                        "the vision projector and the model must be different files".to_string()
                    );
                }
                lines.push("--mmproj".to_string());
                lines.push(format!("{ZIP_PREFIX}{mmproj_name}"));
                embeds_mmproj = true;
            }
            None => errors.push(format!(
                "the projector path \"{mmproj_path}\" has no file name"
            )),
        }
    }

    // --- network (server mode only) ---------------------------------------
    let host = config.host.trim();
    if is_server {
        if host.is_empty() {
            errors.push("the server bind address must not be empty".to_string());
        } else {
            // A host/port mismatch is the single most common "why can't I reach
            // it" issue, so flag the loopback default explicitly.
            if host == "127.0.0.1" || host == "localhost" {
                warnings.push(
                    "bound to loopback only: the server will not be reachable from other machines"
                        .to_string(),
                );
            }
            lines.push("--host".to_string());
            lines.push(host.to_string());
        }
        if config.port == 0 {
            errors.push("the server port must be between 1 and 65535".to_string());
        } else {
            lines.push("--port".to_string());
            lines.push(config.port.to_string());
        }
    } else if !host.is_empty() && host != "127.0.0.1" {
        warnings.push(
            "--host and --port are server options and are ignored in chat mode".to_string(),
        );
    }

    // --- hardware ---------------------------------------------------------
    let gpu_layers = config.gpu_layers;
    if gpu_layers > MAX_GPU_LAYERS {
        warnings.push(format!(
            "GPU layers clamped from {gpu_layers} to {MAX_GPU_LAYERS}, the maximum this slider allows"
        ));
    }
    lines.push("-ngl".to_string());
    lines.push(gpu_layers.min(MAX_GPU_LAYERS).to_string());

    if config.ctx_size == 0 {
        warnings.push("a context size of 0 makes the runtime use the model default".to_string());
    }
    lines.push("-c".to_string());
    lines.push(config.ctx_size.to_string());

    if config.threads == 0 {
        errors.push("CPU threads must be at least 1".to_string());
    } else {
        lines.push("-t".to_string());
        lines.push(config.threads.to_string());
    }

    // --- persona ----------------------------------------------------------
    let prompt = config.system_prompt.trim();
    let embeds_persona = !prompt.is_empty();
    let mut persona_applied = false;
    if embeds_persona {
        // The persona is always embedded so it travels with the binary, but only
        // chat/CLI mode has a flag that applies it: `-sys/--system-prompt-file`
        // is registered for those modes only, and llama-server rejects unknown
        // arguments outright. Server mode takes the prompt per API request.
        if is_server {
            warnings.push(
                "persona embedded as /zip/persona.txt, but llamafile's server takes the system prompt per request - use Chat mode to bake it in"
                    .to_string(),
            );
        } else {
            lines.push("-sysf".to_string());
            lines.push(format!("{ZIP_PREFIX}{PERSONA_ENTRY}"));
            persona_applied = true;
        }
    }

    // --- stop tokens ------------------------------------------------------
    let mut stop_tokens: Vec<String> = Vec::new();
    for token in &config.stop_tokens {
        match normalise_stop_token(token) {
            None => {}
            Some(expanded) => {
                if expanded.contains('\n') || expanded.contains('\r') {
                    warnings.push(format!(
                        "stop token \"{token}\" contains a newline; .args holds one argument per line so it cannot be baked in - newline stopping is active by default"
                    ));
                    continue;
                }
                if expanded.trim().is_empty() {
                    continue;
                }
                stop_tokens.push(expanded);
            }
        }
    }
    if !stop_tokens.is_empty() {
        if is_server {
            warnings.push(
                "stop tokens are per-request in server mode and are not baked into the server"
                    .to_string(),
            );
        } else {
            for token in &stop_tokens {
                lines.push("-r".to_string());
                lines.push(token.clone());
            }
        }
    }

    // --- advanced ---------------------------------------------------------
    for raw in config.extra_args.lines() {
        let line = raw.trim_end_matches('\r');
        if line.trim().is_empty() {
            continue;
        }
        if line.trim() == "..." {
            warnings.push(
                "the trailing \"...\" terminator is added automatically; the extra copy was ignored"
                    .to_string(),
            );
            continue;
        }
        // One argument per line: preserve interior spaces verbatim.
        lines.push(line.trim().to_string());
    }

    // --- runtime passthrough ----------------------------------------------
    // Mandatory: everything the end user types lands after the baked flags.
    lines.push("...".to_string());

    let mut text = lines.join("\n");
    text.push('\n');

    ArgsPreview {
        lines,
        text,
        warnings,
        errors,
        embeds_persona,
        persona_applied,
        embeds_mmproj,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_config() -> CompileConfig {
        CompileConfig {
            base_runtime: "/tmp/llamafile".to_string(),
            model: "/tmp/TinyLlama.Q4_K_M.gguf".to_string(),
            output: "/tmp/out.llamafile".to_string(),
            execution_mode: "chat".to_string(),
            host: "127.0.0.1".to_string(),
            port: 8080,
            gpu_layers: 99,
            ctx_size: 4096,
            threads: 6,
            system_prompt: String::new(),
            mmproj: String::new(),
            stop_tokens: vec![],
            extra_args: String::new(),
        }
    }

    #[test]
    fn ends_with_terminator_and_uses_zip_prefix() {
        let preview = build(&base_config());
        assert!(preview.errors.is_empty());
        assert_eq!(preview.lines.last().unwrap(), "...");
        assert_eq!(preview.lines[0], "--chat");
        assert!(preview.lines.contains(&"/zip/TinyLlama.Q4_K_M.gguf".to_string()));
    }

    #[test]
    fn embeds_vision_projector() {
        let mut config = base_config();
        config.mmproj = "/tmp/mmproj-model-f16.gguf".to_string();
        let preview = build(&config);
        assert!(preview.errors.is_empty(), "{:?}", preview.errors);
        assert!(preview.embeds_mmproj);
        let index = preview
            .lines
            .iter()
            .position(|line| line == "--mmproj")
            .expect("--mmproj must be emitted");
        assert_eq!(preview.lines[index + 1], "/zip/mmproj-model-f16.gguf");
        assert_eq!(preview.lines.last().unwrap(), "...");
    }

    #[test]
    fn server_mode_emits_network_flags_but_not_persona() {
        let mut config = base_config();
        config.execution_mode = "server".to_string();
        config.system_prompt = "You are terse.".to_string();
        let preview = build(&config);
        assert!(preview.lines.contains(&"--host".to_string()));
        assert!(preview.lines.contains(&"8080".to_string()));
        assert!(!preview.lines.contains(&"-sysf".to_string()));
        assert!(preview.embeds_persona);
        assert!(!preview.persona_applied);
    }

    #[test]
    fn chat_mode_applies_persona_and_stops() {
        let mut config = base_config();
        config.system_prompt = "Answer in JSON.".to_string();
        config.stop_tokens = vec!["<|end|>".to_string(), "\\n".to_string()];
        let preview = build(&config);
        assert!(preview.lines.contains(&"-sysf".to_string()));
        assert!(preview.lines.contains(&"/zip/persona.txt".to_string()));
        assert_eq!(preview.lines.iter().filter(|l| *l == "-r").count(), 1);
        assert!(preview
            .warnings
            .iter()
            .any(|w| w.contains("contains a newline")));
    }
}
