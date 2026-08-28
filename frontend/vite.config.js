import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueDevTools from 'vite-plugin-vue-devtools'
import AutoImport from 'unplugin-auto-import/vite'
import Components from 'unplugin-vue-components/vite'
import { NaiveUiResolver } from 'unplugin-vue-components/resolvers'

// https://vite.dev/config/
export default defineConfig(({ mode }) => ({
  test: {
    environment: 'jsdom',
    globals: true,
    include: ['src/**/*.{test,spec}.{js,ts}'],
    // jsdom 不校验标签名，emoji 等非法名在此静默通过、在 Blink 里渲染期抛异常。
    // 这层 setup 把校验补回来，让白屏级失效在单测阶段就暴露。
    setupFiles: ['src/__tests__/setup/blinkDom.js'],
  },
  base: '/',
  build: {
    modulePreload: false,
    cssCodeSplit: false,
  },
  plugins: [
    // 移除 crossorigin 属性（QWebEngineView 兼容）
    {
      name: 'strip-crossorigin',
      transformIndexHtml(html) {
        return html.replace(/ crossorigin/g, '')
      },
    },
    vue(),
    // vitest 模式下不加载 devtools，避免 vite-plugin-inspect 兼容性崩溃
    ...(mode !== 'test' ? [vueDevTools()] : []),
    AutoImport({
      imports: [
        'vue',
        {
          'naive-ui': [
            'useDialog',
            'useMessage',
            'useNotification',
            'useLoadingBar',
          ],
        },
      ],
    }),
    Components({
      resolvers: [NaiveUiResolver()],
    }),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    },
  },
  server: {
    watch: {
      usePolling: true,
      interval: 300,
    },
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://127.0.0.1:8000',
        ws: true,
      },
    },
  },
}))
