/**
 * Tailwind configuration for Llamafile Studio.
 *
 * The palette is a neon circuitry theme drawn from the llamafile artwork:
 * near-black navy surfaces, cyan and magenta neon accents, violet mid-tones.
 * Only the utilities that are actually used by `ui/**` are emitted, which keeps
 * the shipped stylesheet tiny (the app is fully offline at runtime).
 */
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./ui/**/*.{html,js}"],
  theme: {
    extend: {
      colors: {
        // Deep-space surface ramp (higher elevation = lighter navy).
        window: "#04060e",
        panel: "#060b16",
        view: "#080f1e",
        card: "#0a1222",
        cardhover: "#0e1830",
        raised: "#101c33",
        line: "#1b2942",
        linestrong: "#27395c",
        // Text ramp, cooled towards blue.
        ink: "#f3f8ff",
        inksoft: "#c8d6ec",
        inkmuted: "#8fa4c2",
        inkfaint: "#637c9e",
        // Neon accents.
        accent: "#3fe3ff",
        accenthover: "#7beeff",
        accentsoft: "rgba(63,227,255,0.12)",
        accent2: "#ff3ec8",
        accent2hover: "#ff6fdc",
        accent2soft: "rgba(255,62,200,0.12)",
        violet: "#8b5cf6",
        violethover: "#a78bfa",
        violetsoft: "rgba(139,92,246,0.14)",
        success: "#2ee6a8",
        successsoft: "rgba(46,230,168,0.12)",
        warning: "#ffc857",
        warningsoft: "rgba(255,200,87,0.12)",
        danger: "#ff4d6d",
        dangersoft: "rgba(255,77,109,0.12)",
        purge: "#a855f7",
        // Readable ink for tinted surfaces.
        infoink: "#8be9ff",
        warnink: "#ffd98a",
        dangerink: "#ff8fa3",
        violetink: "#d8c2ff"
      },
      fontFamily: {
        sans: [
          "Cantarell",
          "Inter",
          "Ubuntu",
          "Segoe UI",
          "system-ui",
          "-apple-system",
          "sans-serif"
        ],
        mono: ["Source Code Pro", "JetBrains Mono", "Ubuntu Mono", "monospace"]
      },
      borderRadius: {
        adwaita: "9px",
        pill: "999px"
      },
      boxShadow: {
        window: "0 24px 70px rgba(0, 0, 0, 0.72)",
        card: "inset 0 1px 0 rgba(255, 255, 255, 0.03), 0 16px 40px -28px rgba(0, 0, 0, 0.95)",
        cta: "0 0 0 1px rgba(63, 227, 255, 0.22), 0 10px 30px -10px rgba(255, 62, 200, 0.55)",
        "glow-cyan": "0 0 24px -4px rgba(63, 227, 255, 0.6)",
        "glow-pink": "0 0 28px -6px rgba(255, 62, 200, 0.6)",
        "glow-danger": "0 0 18px -4px rgba(255, 77, 109, 0.65)"
      },
      transitionTimingFunction: {
        adwaita: "cubic-bezier(0.2, 0.8, 0.2, 1)"
      },
      keyframes: {
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" }
        },
        shimmer: {
          "0%": { backgroundPosition: "200% 0" },
          "100%": { backgroundPosition: "-200% 0" }
        },
        pulsebar: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.55" }
        }
      },
      animation: {
        "fade-up": "fade-up 180ms cubic-bezier(0.2, 0.8, 0.2, 1) both",
        shimmer: "shimmer 1.6s linear infinite",
        pulsebar: "pulsebar 1.4s ease-in-out infinite"
      }
    }
  },
  plugins: []
};
