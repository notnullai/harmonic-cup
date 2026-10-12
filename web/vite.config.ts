import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the app runs on :5173 and the API on :8000; these paths are
// forwarded so the code can use the same relative URLs it uses in production,
// where the API serves the built app itself.
const API = (globalThis as { process?: { env: Record<string, string | undefined> } })
  .process?.env.HARMONIC_API_URL ?? "http://localhost:8000";
const apiPaths = ["/health", "/sensors", "/history", "/events", "/kg", "/chat",
  "/inject", "/reset", "/sources", "/report.pdf", "/machines", "/recordings", "/uploads",
  "/try"];

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      ...Object.fromEntries(apiPaths.map((p) => [p, { target: API, changeOrigin: true }])),
      "/ws": { target: API.replace("http", "ws"), ws: true },
    },
  },
  build: {
    outDir: "dist",
    chunkSizeWarningLimit: 1600,
    rollupOptions: {
      // Two pages, one build: the console (index.html) and the standalone
      // "try Harmonic on your own data" wizard (web/try/index.html). Vite
      // preserves the input's own path under dist/, so this lands at
      // dist/try/index.html — exactly where api/main.py's `try_page()`
      // route looks for it. Each gets its own entry chunk; common ones
      // (React, theme.css) are shared under dist/assets/.
      input: { main: "index.html", try: "try/index.html" },
      output: {
        manualChunks: { echarts: ["echarts"], graph: ["cytoscape", "cytoscape-dagre"] },
      },
    },
  },
});
