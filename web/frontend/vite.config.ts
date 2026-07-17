import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";
import { buildWorkboxOptions } from "./pwaWorkbox";

// Palette from src/styles.css :root — dark terminal aesthetic.
const THEME_BG = "#05070d";

// Dev-server proxy: the frontend calls /api/* and Vite forwards to the
// FastAPI backend, so no CORS pain in development.
export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      // New shells install automatically (skipWaiting). ReloadPrompt shows a
      // non-intrusive "update available — refresh" banner and does NOT call
      // location.reload() on its own — user chooses when to apply.
      registerType: "autoUpdate",
      // Committed static icons (public/icons/*) — no build-time render step.
      includeAssets: ["icons/apple-touch-icon.png"],
      manifest: {
        name: "Evolve",
        short_name: "Evolve",
        description:
          "Evolve: quant research, AI scoring, backtesting, and a guided analyst — self-hosted.",
        theme_color: THEME_BG,
        background_color: THEME_BG,
        display: "standalone",
        start_url: "/",
        lang: "en",
        icons: [
          { src: "icons/pwa-192.png", sizes: "192x192", type: "image/png", purpose: "any" },
          { src: "icons/pwa-512.png", sizes: "512x512", type: "image/png", purpose: "any" },
          {
            src: "icons/pwa-maskable-512.png",
            sizes: "512x512",
            type: "image/png",
            purpose: "maskable",
          },
        ],
      },
      workbox: buildWorkboxOptions(),
      devOptions: {
        // Keep SW off in Vite dev — localhost still works; avoids fighting HMR.
        enabled: false,
      },
    }),
  ],
  server: {
    proxy: { "/api": "http://localhost:8000" },
  },
});
