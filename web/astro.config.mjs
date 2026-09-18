import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://matheuspavaneli.github.io',
  base: '/rag-eval',
  trailingSlash: 'always',
  output: 'static',
  build: { format: 'directory', inlineStylesheets: 'always' },
  devToolbar: { enabled: false },
});
