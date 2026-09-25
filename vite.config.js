import { defineConfig } from 'vite';

export default defineConfig({
  // '/' for Vercel/Netlify/Cloudflare; '/<repo>/' for GitHub Pages (set by the workflow)
  base: process.env.BASE_PATH || '/',
  // MediaPipe ships its own wasm + loader; keep it out of dependency pre-bundling
  optimizeDeps: { exclude: ['@mediapipe/tasks-vision'] },
  worker: { format: 'es' },
  build: {
    target: 'es2022',
    chunkSizeWarningLimit: 1200,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules/three')) return 'three';
          if (id.includes('@mediapipe')) return 'mediapipe';
        },
      },
    },
  },
  server: { host: true },
});
