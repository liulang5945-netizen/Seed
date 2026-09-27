import { defineConfig } from 'tsdown'

export default defineConfig([
  {
    entry: ['lib/types/main.js'],
    outDir: 'lib',
    format: ['esm'],
    platform: 'node',
    target: 'es2024',
    fixedExtension: false,
    dts: false,
    clean: false,
    // The main process must be self-contained: electron-builder collects only
    // `dependencies` (not `peerDependencies`) into the packaged top-level
    // node_modules, while several `@taiji/*` packages are runtime-imported but
    // only declared as peers (e.g. `@taiji/cordis` via
    // `@taiji/dsh-typert-protocol`). Bundling the whole `@taiji/*` closure
    // removes that class of "ERR_MODULE_NOT_FOUND" at startup.
    deps: { neverBundle: ['electron'], alwaysBundle: [/^@taiji\//] },
  },
  ...(['preload-app', 'preload-platform-account', 'preload-mandatory', 'preload-update-dialog'] as const).map(name => ({
    // Sandboxed Electron preloads run as CommonJS even though the application package is ESM.
    entry: { [name]: `lib/types/${name}.js` },
    outDir: 'lib',
    format: 'cjs' as const,
    codeSplitting: false,
    platform: 'node' as const,
    target: 'es2024',
    fixedExtension: false,
    dts: false,
    clean: false,
    deps: { neverBundle: ['electron'] },
  })),
])