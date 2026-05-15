import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/predictions': 'http://localhost:8000',
      '/elo-standings': 'http://localhost:8000',
      '/history': 'http://localhost:8000',
      '/game': 'http://localhost:8000',
    }
  }
})
