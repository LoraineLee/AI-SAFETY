import { defineConfig } from 'tsup';

export default defineConfig([
  // Universal APIs
  {
    entry: ['streams/index.ts'],
    format: ['cjs', 'esm'],
    external: ['react', 'svelte', 'vue', 'solid-js'],
    dts: true,
    sourcemap: true,
  },
  {
    entry: ['function/index.ts'],
    format: ['cjs', 'esm'],
    external: ['react', 'svelte', 'vue', 'solid-js'],
    outDir: 'function/dist',
    dts: true,
    sourcemap: true,
  },
  {
    entry: ['provider/index.ts'],
    format: ['cjs', 'esm'],
    external: ['react', 'svelte', 'vue', 'solid-js'],
    outDir: 'provider/dist',
    dts: true,
    sourcemap: true,
  },
  {
    entry: ['prompts/index.ts'],
    format: ['cjs', 'esm'],
    external: ['react', 'svelte', 'vue', 'solid-js'],
    outDir: 'prompts/dist',
    dts: true,
    sourcemap: true,
  },
  // React APIs
  {
    entry: ['react/index.ts'],
    outDir: 'react/dist',
    banner: {
      js: "'use client'",
    },
    format: ['cjs', 'esm'],
    external: ['react', 'svelte', 'vue', 'solid-js'],
    dts: true,
    sourcemap: true,
  },
  {
    entry: ['react/index.server.ts'],
    outDir: 'react/dist',
    format: ['cjs', 'esm'],
    external: ['react', 'svelte', 'vue', 'solid-js'],
    dts: true,
    sourcemap: true,
  },
  // Svelte APIs
  {
    entry: ['svelte/index.ts'],
    outDir: 'svelte/dist',
    banner: {},
    format: ['cjs', 'esm'],
    external: ['react', 'svelte', 'vue', 'solid-js'],
    dts: true,
    sourcemap: true,
    // `sswr` has some issue with `.es.js` that can't be resolved correctly by
    // vite so we have to bundle it here.
    noExternal: ['sswr'],
  },
  // Vue APIs
  {
    entry: ['vue/index.ts'],
    outDir: 'vue/dist',
    banner: {},
    format: ['cjs', 'esm'],
    external: ['react', 'svelte', 'vue', 'solid-js'],
    dts: true,
    sourcemap: true,
  },
  // Solid APIs
  {
    entry: ['solid/index.ts'],
    outDir: 'solid/dist',
    banner: {},
    format: ['cjs', 'esm'],
    external: ['react', 'svelte', 'vue', 'solid-js'],
    dts: true,
    sourcemap: true,
  },
]);
