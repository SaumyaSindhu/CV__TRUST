import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// base '/static/' + outDir '../backend/static': the production build lands
// exactly where FastAPI already serves it (mount /static + index at /),
// so the backend needs zero changes and the app runs fully air-gapped.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "/static/",
  build: { outDir: "../backend/static", emptyOutDir: true },
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
