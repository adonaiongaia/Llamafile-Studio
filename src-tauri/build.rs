fn main() {
    // Generates the permission schemas and embeds the application metadata that
    // `tauri::generate_context!()` reads at compile time.
    tauri_build::build()
}
