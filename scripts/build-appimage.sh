#!/usr/bin/env bash
#
# Builds "Llamafile Studio" as a Linux AppImage.
#
# Requirements (Debian/Ubuntu):
#   sudo apt install build-essential pkg-config libwebkit2gtk-4.1-dev \
#       libgtk-3-dev librsvg2-dev libayatana-appindicator3-dev libssl-dev
#   rustup (https://rustup.rs)  ·  node >= 18  ·  npm
#
# Usage:
#   ./scripts/build-appimage.sh            # release AppImage
#   ./scripts/build-appimage.sh --debug    # debug build (faster, for iteration)
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PROFILE="release"
TAURI_FLAGS=()
if [[ "${1:-}" == "--debug" ]]; then
  PROFILE="debug"
  TAURI_FLAGS+=("--debug")
fi

echo "==> Checking the toolchain"
command -v node >/dev/null || { echo "node is missing (install Node.js 18+)"; exit 1; }
command -v npm >/dev/null || { echo "npm is missing"; exit 1; }
if ! command -v cargo >/dev/null; then
  if [[ -x "$HOME/.cargo/bin/cargo" ]]; then
    export PATH="$HOME/.cargo/bin:$PATH"
  else
    echo "cargo is missing (install Rust from https://rustup.rs)"; exit 1
  fi
fi
pkg-config --exists webkit2gtk-4.1 || {
  echo "libwebkit2gtk-4.1-dev is missing; see the header of this script."
  exit 1
}

echo "==> Installing npm dependencies"
if [[ -f package-lock.json ]]; then
  npm ci --no-audit --no-fund
else
  npm install --no-audit --no-fund
fi

echo "==> Generating the icon set"
node scripts/make-icons.mjs

echo "==> Compiling the Tailwind stylesheet"
npm run build:css

echo "==> Compiling the Tauri application and bundling the AppImage"
# `tauri build` runs `beforeBuildCommand` (CSS) again, which is idempotent.
npx tauri build --bundles appimage "${TAURI_FLAGS[@]}"

BUNDLE_DIR="src-tauri/target/${PROFILE}/bundle/appimage"
echo
if compgen -G "${BUNDLE_DIR}/*.AppImage" > /dev/null; then
  echo "==> AppImage ready:"
  ls -lh "${BUNDLE_DIR}"/*.AppImage
else
  echo "!! No AppImage was produced in ${BUNDLE_DIR}" >&2
  exit 1
fi
