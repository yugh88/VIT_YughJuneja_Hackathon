import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Built into dist/ and served by FastAPI at "/"; in dev, API calls are proxied to :8000.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist", chunkSizeWarningLimit: 1500 },
  server: {
    proxy: {
      "/v1": { target: "http://localhost:8000", ws: true },
    },
  },
});
