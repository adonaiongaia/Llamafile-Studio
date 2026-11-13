#!/usr/bin/env node
/**
 * Generates the PNG icon set used by the Tauri bundle without pulling in any
 * image library. Everything is drawn into a raw RGBA buffer and encoded with
 * Node's built-in zlib.
 *
 * Run with:  node scripts/make-icons.mjs
 */
import { deflateSync } from "node:zlib";
import { mkdirSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const outDir = join(root, "src-tauri", "icons");
mkdirSync(outDir, { recursive: true });

/* ------------------------------------------------------------------ helpers */

const CRC_TABLE = (() => {
  const table = new Int32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c;
  }
  return table;
})();

function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length, 0);
  const body = Buffer.concat([Buffer.from(type, "ascii"), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body), 0);
  return Buffer.concat([len, body, crc]);
}

/** Encode an RGBA pixel buffer (width*height*4) as a PNG file. */
function encodePng(width, height, rgba) {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // colour type: RGBA
  ihdr[10] = 0; // deflate
  ihdr[11] = 0; // adaptive filtering
  ihdr[12] = 0; // no interlace

  // Each scanline is prefixed with filter type 0 (None).
  const stride = width * 4;
  const raw = Buffer.alloc((stride + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (stride + 1)] = 0;
    rgba.copy(raw, y * (stride + 1) + 1, y * stride, y * stride + stride);
  }

  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", ihdr),
    chunk("IDAT", deflateSync(raw, { level: 9 })),
    chunk("IEND", Buffer.alloc(0))
  ]);
}

/* ------------------------------------------------------------------- drawing */

const lerp = (a, b, t) => a + (b - a) * t;
const clamp01 = (v) => Math.min(1, Math.max(0, v));

/** Signed distance to a rounded rectangle, used for antialiased edges. */
function roundedRectDistance(x, y, cx, cy, halfW, halfH, radius) {
  const dx = Math.abs(x - cx) - (halfW - radius);
  const dy = Math.abs(y - cy) - (halfH - radius);
  const outside = Math.hypot(Math.max(dx, 0), Math.max(dy, 0));
  return outside + Math.min(Math.max(dx, dy), 0) - radius;
}

/** Blend `src` over `dst` with coverage `a` (all arrays are RGBA floats 0-1). */
function blend(dst, i, r, g, b, a) {
  if (a <= 0) return;
  const inv = 1 - a;
  dst[i] = r * a + dst[i] * inv;
  dst[i + 1] = g * a + dst[i + 1] * inv;
  dst[i + 2] = b * a + dst[i + 2] * inv;
  dst[i + 3] = a + dst[i + 3] * inv;
}

/**
 * Draws the app icon: a rounded "window" with an indigo→teal gradient, a
 * stylised llama head silhouette in white and a small spark. All coordinates
 * are expressed in a normalised 0..1 space so every size scales cleanly.
 */
function drawIcon(size) {
  const px = new Float32Array(size * size * 4);
  const aa = 1.25 / size; // one and a bit pixels of antialiasing coverage

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      const u = (x + 0.5) / size;
      const v = (y + 0.5) / size;

      // Background: transparent outside the rounded square.
      const bg = roundedRectDistance(u, v, 0.5, 0.5, 0.5, 0.5, 0.22);
      const bgCover = clamp01(0.5 - bg / aa);
      if (bgCover > 0) {
        // Diagonal gradient: deep purple -> Adwaita blue -> teal.
        const t = clamp01((u * 0.55 + v * 0.45 - 0.05) / 0.95);
        let r, g, b;
        if (t < 0.55) {
          const k = t / 0.55;
          r = lerp(0.36, 0.16, k);
          g = lerp(0.16, 0.42, k);
          b = lerp(0.62, 0.82, k);
        } else {
          const k = (t - 0.55) / 0.45;
          r = lerp(0.16, 0.11, k);
          g = lerp(0.42, 0.72, k);
          b = lerp(0.82, 0.62, k);
        }
        blend(px, i, r, g, b, bgCover);
      }

      // Soft inner glow at the top-left for depth.
      const glowDist = Math.hypot(u - 0.18, v - 0.12);
      const glow = clamp01(1 - glowDist / 0.62) ** 3 * 0.28 * bgCover;
      blend(px, i, 1, 1, 1, glow);

      // --- Llama head silhouette -----------------------------------------
      // Ears (two small rounded triangles approximated by capsules).
      const earLeft = Math.hypot(u - 0.335, v - 0.245) - 0.043;
      const earRight = Math.hypot(u - 0.545, v - 0.245) - 0.043;
      const ears = Math.min(earLeft, earRight);
      const earsCover = clamp01(0.5 - ears / aa);

      // Head: rounded rectangle.
      const head = roundedRectDistance(u, v, 0.5, 0.545, 0.155, 0.175, 0.075);
      const headCover = clamp01(0.5 - head / aa);

      // Snout: rounded rectangle pushed slightly to the right/bottom.
      const snout = roundedRectDistance(u, v, 0.53, 0.585, 0.115, 0.072, 0.055);
      const snoutCover = clamp01(0.5 - snout / aa);

      const silhouette = Math.max(earsCover, headCover, Math.max(snoutCover, 0));

      // Eyes and nose are punched out of the white silhouette.
      const eye = Math.hypot(u - 0.455, v - 0.5) - 0.019;
      const eyeCover = clamp01(0.5 - eye / aa);
      const nose = Math.hypot(u - 0.6, v - 0.565) - 0.014;
      const noseCover = clamp01(0.5 - nose / aa);
      const punch = Math.max(eyeCover, noseCover);

      const white = clamp01(silhouette - punch * silhouette);
      blend(px, i, 1, 1, 1, white * 0.96 * bgCover);

      // --- Spark / terminal prompt chevron --------------------------------
      const chevronA = roundedRectDistance(u, v, 0.665, 0.762, 0.052, 0.0125, 0.0125);
      const chevronACover = clamp01(0.5 - chevronA / aa) * (u < 0.665 ? 1 : 0);
      const chevronB = roundedRectDistance(u, v, 0.722, 0.742, 0.0125, 0.052, 0.0125);
      const chevronBCover = clamp01(0.5 - chevronB / aa);
      blend(px, i, 0.13, 0.13, 0.15, Math.max(chevronACover, chevronBCover) * 0.55 * bgCover);
    }
  }

  // Convert float RGBA to bytes.
  const out = Buffer.alloc(size * size * 4);
  for (let p = 0; p < px.length; p++) {
    out[p] = Math.round(clamp01(px[p]) * 255);
  }
  return encodePng(size, size, out);
}

/* ---------------------------------------------------------------------- main */

const targets = [
  ["32x32.png", 32],
  ["128x128.png", 128],
  ["128x128@2x.png", 256],
  ["icon.png", 512]
];

for (const [name, size] of targets) {
  const png = drawIcon(size);
  writeFileSync(join(outDir, name), png);
  console.log(`wrote src-tauri/icons/${name} (${size}x${size}, ${png.length} bytes)`);
}
