import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev-server proxy: the frontend calls /api/* and Vite forwards to the
// FastAPI backend, so no CORS pain in development.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": "http://localhost:8000" },
  },
});
