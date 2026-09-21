module.exports = {
  root: true,
  // 生成物：由 openapi-typescript 从冻结快照产出，逐字节受 check:api-types 守护。
  // 对它做 lint/格式化会同时制造假阳性并破坏那份逐字节比对。
  ignorePatterns: ['src/api/schema.d.ts', 'dist'],
  env: {
    browser: true,
    es2022: true,
    node: true,
  },
  extends: [
    'eslint:recommended',
    'plugin:vue/vue3-recommended',
    'plugin:import/recommended',
    'prettier',
  ],
  parser: 'vue-eslint-parser',
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
  },
  settings: {
    'import/resolver': {
      // F14: 一旦显式声明 resolver，import 插件就只用列出的这些。
      // 必须同时保留 node —— alias resolver 不识别 `node:fs` 这类内置模块协议前缀，
      // 只配 alias 会让 `import { readFileSync } from 'node:fs'` 被误判 no-unresolved。
      node: {
        extensions: ['.js', '.mjs', '.cjs', '.ts', '.vue', '.json'],
      },
      alias: {
        map: [['@', './src']],
        extensions: ['.js', '.ts', '.vue', '.json'],
      },
    },
    'import/core-modules': ['node:fs', 'node:path', 'node:url', 'node:os'],
  },
  rules: {
    // R5: 只限制 log/info/debug——warn/error 是应用正常运行时诊断通道，允许保留
    'no-console': ['warn', { allow: ['warn', 'error'] }],
    'no-debugger': 'warn',
    'no-unused-vars': ['warn', { argsIgnorePattern: '^_' }],
    'vue/multi-word-component-names': 'off',
    // 有意为之的模式降级为 warning，避免阻断 CI
    'no-empty': ['warn', { allowEmptyCatch: true }],
    'no-constant-condition': ['warn', { checkLoops: false }],
    'no-inner-declarations': 'warn',
    'no-case-declarations': 'warn',
    // F14: 启用 import 路径解析（配合 eslint-import-resolver-alias）
    'import/no-unresolved': 'error',
  },
  overrides: [
    {
      // TS 源文件：root parser 是 vue-eslint-parser，它不解析 TS 语法
      // （`import type`、`<const T>` 会直接报 Parsing error），必须换 parser。
      // 注意：将来 .vue 的 <script lang="ts"> 还需要在 root 的 parserOptions
      // 里补 `parser: '@typescript-eslint/parser'`，本次尚无 TS 版 SFC，故不加。
      files: ['**/*.ts'],
      parser: '@typescript-eslint/parser',
      parserOptions: { ecmaVersion: 'latest', sourceType: 'module' },
    },
    {
      // 测试文件：注入 vitest 全局变量
      files: ['**/__tests__/**/*.{js,ts}', '**/*.{test,spec}.{js,ts}'],
      globals: {
        describe: 'readonly',
        it: 'readonly',
        expect: 'readonly',
        beforeEach: 'readonly',
        afterEach: 'readonly',
        beforeAll: 'readonly',
        afterAll: 'readonly',
        vi: 'readonly',
      },
    },
  ],
}
