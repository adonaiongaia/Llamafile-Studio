// Hide the extra console window on Windows in release builds. On Linux this
// attribute is inert, but it keeps the crate cross-platform clean.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

/// Thin binary wrapper: all logic lives in the library so the same code can be
/// reused by tests and by the mobile entry point Tauri generates.
fn main() {
    llamafile_studio_lib::run()
}
