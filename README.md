# Llamafile Studio

A native Linux desktop application (Rust + Tauri 2 + Tailwind) that turns a GGUF
model into **one standalone, double-clickable `.llamafile` executable** — no
command line, no shell scripts, no `zipalign` invocation by hand.

![tabs](https://img.shields.io/badge/UI-Files%20%C2%B7%20Persona%20%C2%B7%20Settings-ff3ec8)
![backend](https://img.shields.io/badge/backend-Rust%20%2B%20Tauri%202-2ee6a8)

---

## What it does

A `.llamafile` is an **APE** (Actually Portable Executable) with a ZIP archive
appended to it. Studio performs the whole packaging pipeline natively:

```
 base runtime (APE) ─┐
 .args              ─┼─▶  ZIP: stored, page-aligned, ZIP64 when needed  ─▶  chmod 0755  ─▶  verify
 model.gguf         ─┤
 mmproj.gguf        ─┤   (optional vision projector)
 persona.txt        ─┘
```

1. **Base runtime** — pick a local `llamafile` binary or download the latest
   release from GitHub (the app asks the releases API and prefers the full
   multi-backend runtime, falling back to `-thin`).
2. **`.args`** — generated with *exactly one argument per line*, always ending
   with the `...` terminator so runtime flags still override the baked ones.
3. **ZIP append** — entries are written **uncompressed (method 0)** and each
   payload starts on a **16384-byte boundary**, the alignment upstream
   `zipalign` uses and the one the runtime checks before mapping weights. Sizes
   above 4 GiB automatically switch to **ZIP64**, which matters for essentially
   every 7B+ Q8 model.
4. **Permissions & verification** — the result is `chmod 0755`, then re-opened to
   confirm the end-of-central-directory record and entry count survived.

Everything is implemented in `src-tauri/src/zipwriter.rs` (~400 lines of Rust),
so no external `zipalign` binary is required at build time or run time.

## Features

| Area | What you get |
| --- | --- |
| **Files** | Runtime picker + one-click GitHub download with live progress, `.gguf`-filtered model picker, optional `mmproj` vision projector picker, optional `ggml-*.so` / `ggml-*.dll` GPU plugin picker, save dialog for the output |
| **Models** | Hugging Face search over GGUF repositories, capability badges (tool calling, thinking, vision, MoE, code, multilingual), per-quantisation sizes with a RAM fit hint, a cancellable download with a progress bar, and a downloads manager that lists cached files with sizes and deletes them |
| **Persona** | 8 000-character prompt canvas with live line/character count, six pre-baked archetypes (Software Architect, Creative Storyteller, Strict JSON Extractor, Medical Scribe, Socratic Tutor, Linux Sysadmin), dynamic stop-token chips |
| **Settings** | `--server` / `--chat` toggle, host + port binder, GPU offload slider 0–120, context selector (2K–16K) with memory-aware warnings, CPU threads defaulting to *physical cores − 2* detected via `num_cpus` |
| **Feedback** | Real-time progress bar with byte counters, live `.args` preview with validation warnings, post-build report (entries, offsets, alignment, ZIP64, log) that can be copied or saved |

The UI uses a neon-circuitry dark theme drawn from the llamafile artwork:
near-black navy surfaces, cyan (`#3fe3ff`) and magenta (`#ff3ec8`) accents,
violet mid-tones, mono HUD labels and glowing hairlines. It ships no web fonts
and works fully offline.

## Build

### Requirements

```bash
sudo apt install build-essential pkg-config libwebkit2gtk-4.1-dev \
    libgtk-3-dev librsvg2-dev libayatana-appindicator3-dev libssl-dev
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Node 18+
```

### One command

```bash
./scripts/build-appimage.sh
```

The AppImage lands in `src-tauri/target/release/bundle/appimage/`.

### Manual

```bash
npm install
node scripts/make-icons.mjs   # generates src-tauri/icons/*.png
npm run build:css             # Tailwind -> ui/styles.css
npm run tauri dev             # develop
npm run tauri build           # .deb + AppImage
```

Unit tests cover the ZIP writer (alignment, end-of-archive integrity) and the
`.args` generator (mode flags, persona wiring, stop-token handling):

```bash
cargo test --manifest-path src-tauri/Cargo.toml
```

## Model browser

Search talks to `https://huggingface.co/api/models?filter=gguf` and shows, per
repository: parameter count (from the name), GGUF architecture, context length,
total repository size, downloads, likes, licence and the last update.

Capability badges are **inferred**, not read from a normalised field, because the
Hub does not expose one. The strongest signal is the repository's Jinja
`chat_template`, which is fetched server-side and discarded after inspection:

| Badge | Signal |
| --- | --- |
| Tool calling | template mentions `tool_call`, `<tools>`, `tool_response` or `function_call` |
| Thinking | template mentions `reasoning_content`, `enable_thinking`, `reasoning_effort` or ` thinking`; otherwise known reasoning families/tags |
| Vision | a multimodal projector (`mmproj`) is published, or an `image-text-to-text` pipeline / vision tag |
| MoE / Code / Multilingual | tags, architecture and model-card languages |

Expanding a repository lists every single-file quantisation with its real size
(LFS metadata), sorted smallest first, and a RAM fit hint based on the detected
system memory. Split (`-00001-of-…`) shards are skipped because a llamafile
embeds one model file.

Downloads stream to the chosen folder (`~/Models` by default), are written to a
`.part` file and only renamed after the `GGUF` magic is verified, so an
interrupted transfer can never reach the compiler. Cancelling deletes the
partial file.

Interrupted downloads resume. The size of the existing `.part` file is sent as an
HTTP `Range` start, so a dropped connection, a failed transfer or a closed app
continues from the bytes already on disk instead of downloading gigabytes again —
the progress bar picks up at that offset. If the server ignores the range (HTTP
200), or reports the offset as unsatisfiable (HTTP 416), Studio discards the
partial file and starts that download from zero once. A stream that ends before
the expected total is treated as interrupted and its partial file is kept.

### Vision models

A multimodal GGUF only works together with its matching projection file. Pick
one in the **Files** tab (or download it from the browser's projector list) and
Studio embeds it as a second archive entry plus a baked `--mmproj` argument:

```text
-m
/zip/Qwen2.5-VL-3B-Q4_K_M.gguf
--mmproj
/zip/mmproj-Qwen2.5-VL-3B-f16.gguf
...
```

`--mmproj` is registered by every llamafile mode (server, chat and the default
combined mode), which is why it is emitted regardless of the execution mode.
The projector is optional and never required for text-only models.

Projectors are paired automatically. When a model is selected or downloaded,
Studio looks beside the weights for `mmproj*.gguf`, prefers a file that shares
the model family and then the `f16` flavour, and attaches it with a note in the
meta line. A projector you picked yourself is never overwritten, the pairing is
retried whenever the model changes, and a dead automatic path is dropped rather
than kept. Downloading a vision model that has no projector on disk surfaces
that repository's projector list directly in the Files tab, so the download
becomes an attached projector without going back to the browser.

### GPU libraries

llamafile does not link GPU support statically: it loads CUDA, ROCm and Vulkan
as plugins named `ggml-(cuda|rocm|vulkan).(so|dll)`, searching the executable's
directory, its own bundle, `~/.llamafile/v/<version>` and `$HOME`. The **Files**
tab accepts any number of `ggml-*.so` / `ggml-*.dll` files and appends each one
at the archive root — the layout `zipalign -j0 llamafile ggml-cuda.so` produces —
page aligned like the weights. The selection is validated (prefix, extension,
lowercase loadable name, no duplicate entry names), the library sizes join the
progress total, and every plugin appears in the build report. Leave the list
empty to ship the runtime exactly as downloaded.

### Downloaded cache

The download folder doubles as the cache. Every time results or a repository are
rendered, Studio lists that folder and marks any quantisation whose **file name
and byte size** match as `Downloaded`, replacing the Download button with **Use**
(the same applies to projectors). No database is involved, so moving or deleting
a file updates the badges immediately. Matching on size as well as name means a
truncated or differently built file is never mistaken for a cached download.

The **Models** tab also lists the folder itself under *Downloaded models*: every
file with its size plus the folder total, projectors tagged separately, the file
currently selected in the Files tab marked *In use*, and a Delete button per
file. Deleting asks for confirmation, clears the selection if that file was in
use, and the backend refuses any path that is not a `.gguf` directly inside the
download folder.

## Baked arguments

Chat mode, persona and two stop tokens:

```text
--chat
-m
/zip/TinyLlama-1.1B.Q4_K_M.gguf
-ngl
99
-c
4096
-t
6
-f
/zip/persona.txt
-r
<|im_end|>
...
```

Server mode adds `--host 127.0.0.1 --port 8080` and omits the chat-only flags.

### How the persona is wired up

`--chat` runs llamafile's terminal chatbot, which builds the conversation's
system message from the **prompt** (`-p`, labelled "system prompt" in
`llamafile --chat --help`) and never reads `-sys`/`-sysf`. Studio therefore
binds the persona with `-f /zip/persona.txt`, which loads the embedded file
into that prompt. (`-sysf` parses without error in chat mode but is silently
ignored, so the persona would never reach the model.)

llamafile's HTTP server takes the **system prompt and stop sequences per
request** (`llama-server` rejects `-sys`/`-sysf` as unknown arguments, and
upstream removed the server-wide prompt). Studio therefore:

* always embeds the persona as `/zip/persona.txt` so it travels with the binary,
* wires it up automatically with `-f` in **Chat mode**,
* shows an explicit warning in the UI when **Server mode** is selected.

That keeps generated binaries runnable instead of shipping flags the runtime
would refuse.

## Project layout

```
ui/                     index.html, main.js, Tailwind source + built CSS
src-tauri/src/lib.rs    Tauri commands: system info, dialogs, validation, pipeline
src-tauri/src/args.rs   .args generation + validation (unit tested)
src-tauri/src/gpu.rs    GPU backend plugin validation (unit tested)
src-tauri/src/zipwriter.rs  stored/page-aligned/ZIP64 ZIP appender (unit tested)
src-tauri/src/runtime.rs    GitHub release discovery + streaming download
scripts/                icon generation, AppImage build
```

## License

Apache-2.0. llamafile itself is Apache-2.0 and remains the property of its
authors; Studio only packages its runtime with your own model.
