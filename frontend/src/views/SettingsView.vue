<template>
  <section class="dedicated-view settings-view">
    <!-- 顶栏：居中标题 + 副标题 -->
    <header class="settings-topbar">
      <span style="width:0"></span>
      <div class="topbar-title-wrap">
        <span class="topbar-title">设置</span>
        <span class="topbar-sub">Seed系统配置与偏好</span>
      </div>
      <span class="topbar-spacer"></span>
    </header>

    <!-- 滚动内容区 -->
    <div class="settings-scroll">
      <div class="settings-wrap">
        <RuntimeEvidenceStrip context="settings" compact />
        <div class="settings-layout">

          <!-- ═══ 左侧：设置导航 ═══ -->
          <nav class="settings-nav" aria-label="设置导航">
            <button
              class="sn-item"
              :class="{ active: activeSection === 'general' }"
              @click="activeSection = 'general'"
            >
              <svg class="sn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M12 3v2.5M12 18.5V21M21 12h-2.5M5.5 12H3M18.5 5.5l-1.8 1.8M7.3 16.7l-1.8 1.8M18.5 18.5l-1.8-1.8M7.3 7.3 5.5 5.5"/></svg>
              通用设置
            </button>
            <button
              class="sn-item"
              :class="{ active: activeSection === 'neuron' }"
              @click="activeSection = 'neuron'"
            >
              <svg class="sn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="7" r="2.2"/><circle cx="6" cy="17" r="2.2"/><circle cx="18" cy="12" r="2.2"/><circle cx="12" cy="4" r="1.5"/><circle cx="12" cy="20" r="1.5"/><path d="M8 7.5 12 5M8 16.5 12 19M8 16l7-4M8 8l7 3"/></svg>
              Taiji 设置
            </button>
            <button
              class="sn-item"
              :class="{ active: activeSection === 'runtime' }"
              @click="activeSection = 'runtime'"
            >
              <svg class="sn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v4M12 18v4M4.9 4.9l2.9 2.9M16.2 16.2l2.9 2.9M2 12h4M18 12h4M4.9 19.1l2.9-2.9M16.2 7.8l2.9-2.9"/><circle cx="12" cy="12" r="3.2"/></svg>
              运行环境
            </button>
            <button
              class="sn-item"
              :class="{ active: activeSection === 'privacy' }"
              @click="activeSection = 'privacy'"
            >
              <svg class="sn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="10" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
              数据与隐私
            </button>
            <button
              class="sn-item"
              :class="{ active: activeSection === 'about' }"
              @click="activeSection = 'about'"
            >
              <svg class="sn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 16v-4M12 8h.01"/></svg>
              关于
            </button>
          </nav>

          <!-- ═══ 右侧：设置内容 ═══ -->
          <div class="settings-content">

            <!-- ── 1. 通用设置 ── -->
            <section v-if="activeSection === 'general'" class="settings-section">
              <h2>通用设置</h2>

              <!-- 外观主题 -->
              <div class="setting-row setting-row--col setting-row--first">
                <span class="setting-label">外观主题</span>
                <div class="theme-previews">
                  <div
                    v-for="th in appStore.themes"
                    :key="th.id"
                    class="theme-preview-card"
                    :class="{ active: appStore.currentTheme === th.id }"
                    :title="th.desc"
                    @click="appStore.setTheme(th.id)"
                  >
                    <span class="theme-swatch" :style="{ background: th.gradient }"></span>
                    <span class="theme-name">{{ th.name }}</span>
                  </div>
                </div>
              </div>

              <!-- 默认语言 -->
              <div class="setting-row">
                <div class="setting-left">
                  <span class="setting-label">默认语言</span>
                  <p class="setting-desc">界面与交互的显示语言</p>
                </div>
                <div class="setting-right">
                  <select v-model="uiLanguage" aria-label="默认语言" :disabled="savingSettings" @change="onUiLanguageChange">
                    <option value="zh-CN">简体中文</option>
                    <option value="zh-TW">繁體中文</option>
                    <option value="en">English</option>
                    <option value="ja">日本語</option>
                    <option value="ko">한국어</option>
                  </select>
                </div>
              </div>

              <!-- 时区 -->
              <div class="setting-row">
                <div class="setting-left">
                  <span class="setting-label">时区</span>
                  <p class="setting-desc">用于定时任务、日志时间戳等</p>
                </div>
                <div class="setting-right">
                  <select v-model="timezone" aria-label="时区" :disabled="savingSettings" @change="onTimezoneChange">
                    <option value="Asia/Shanghai">Asia/Shanghai (UTC+8)</option>
                    <option value="Asia/Tokyo">Asia/Tokyo (UTC+9)</option>
                    <option value="Asia/Seoul">Asia/Seoul (UTC+9)</option>
                    <option value="Asia/Singapore">Asia/Singapore (UTC+8)</option>
                    <option value="America/Los_Angeles">America/Los_Angeles (UTC-8)</option>
                    <option value="America/New_York">America/New_York (UTC-5)</option>
                    <option value="Europe/London">Europe/London (UTC+0)</option>
                    <option value="Europe/Berlin">Europe/Berlin (UTC+1)</option>
                  </select>
                </div>
              </div>

              <!-- 界面密度 -->
              <div class="setting-row setting-row--last">
                <div class="setting-left">
                  <span class="setting-label">界面密度</span>
                  <p class="setting-desc">调整元素间距与信息密度</p>
                </div>
                <div class="setting-right">
                  <div class="radio-group" role="radiogroup" aria-label="界面密度">
                    <label class="radio-chip">
                      <input v-model="uiDensity" type="radio" name="density" value="compact" :disabled="savingSettings" @change="onDensityChange">
                      <span class="rc-label">紧凑</span>
                    </label>
                    <label class="radio-chip">
                      <input v-model="uiDensity" type="radio" name="density" value="default" :disabled="savingSettings" @change="onDensityChange">
                      <span class="rc-label">默认</span>
                    </label>
                    <label class="radio-chip">
                      <input v-model="uiDensity" type="radio" name="density" value="comfortable" :disabled="savingSettings" @change="onDensityChange">
                      <span class="rc-label">宽松</span>
                    </label>
                  </div>
                </div>
              </div>
            </section>

            <!-- ── 2. Taiji 运行设置 ── -->
            <section v-else-if="activeSection === 'neuron'" class="settings-section">
              <h2>Taiji 运行设置</h2>

              <!-- 局部激活阈值 -->
              <div class="setting-row setting-row--first">
                <div class="setting-left">
                  <span class="setting-label">局部激活阈值</span>
                  <p class="setting-desc">控制单步局部状态更新的激活门槛，不代表全局同步或相位共振</p>
                </div>
                <div class="setting-right">
                  <div class="range-wrap">
                    <input v-model.number="activationThreshold" type="range" min="0" max="1" step="0.01" aria-label="局部激活阈值" :disabled="savingSettings" @change="onThresholdChange" />
                    <span class="range-value">{{ Number(activationThreshold).toFixed(2) }}</span>
                  </div>
                </div>
              </div>

              <!-- 响应超时 -->
              <div class="setting-row">
                <div class="setting-left">
                  <span class="setting-label">响应超时</span>
                  <p class="setting-desc">一次状态推进等待后端返回的最长时间（毫秒）</p>
                </div>
                <div class="setting-right">
                  <input v-model.number="responseTimeoutMs" type="number" min="10" max="10000" aria-label="响应超时" :disabled="savingSettings" @change="onResponseTimeoutChange" />
                </div>
              </div>

              <!-- 自动巩固 -->
              <div class="setting-row">
                <div class="setting-left">
                  <span class="setting-label">自动巩固</span>
                  <p class="setting-desc">高频突触自动强化并写入持久记忆</p>
                </div>
                <div class="setting-right">
                  <label class="toggle" aria-label="自动巩固开关">
                    <input v-model="autoConsolidation" type="checkbox" :disabled="savingSettings" @change="onAutoConsolidationChange" />
                    <span class="track"><span class="thumb"></span></span>
                  </label>
                </div>
              </div>

              <!-- 睡眠模式 -->
              <div class="setting-row setting-row--last">
                <div class="setting-left">
                  <span class="setting-label">睡眠模式</span>
                  <p class="setting-desc">非活跃时段暂停后台状态推进以节省算力，唤醒后恢复 Taiji 持续状态</p>
                </div>
                <div class="setting-right">
                  <label class="toggle" aria-label="睡眠模式开关">
                    <input v-model="sleepMode" type="checkbox" :disabled="savingSettings" @change="onSleepModeChange" />
                    <span class="track"><span class="thumb"></span></span>
                  </label>
                </div>
              </div>
            </section>

            <!-- ── 2.5 运行环境（认知主体切换）── -->
            <section v-else-if="activeSection === 'runtime'" class="settings-section">
              <h2>运行环境</h2>

              <!-- 终端访问（安全） -->
              <div class="setting-row">
                <div class="setting-left">
                  <span class="setting-label">允许未认证终端访问</span>
                  <p class="setting-desc">认证未启用时，允许工作台终端直接连接。开启会降低本地安全性，仅建议在受信任的本机环境使用。</p>
                </div>
                <div class="setting-right">
                  <label class="toggle" aria-label="允许未认证终端访问开关">
                    <input
                      v-model="terminalAllowUnauth"
                      type="checkbox"
                      :disabled="savingTerminalSetting"
                      @change="onTerminalUnauthChange"
                    />
                    <span class="track"><span class="thumb"></span></span>
                  </label>
                </div>
              </div>

              <!-- 切换状态 -->
              <div class="setting-row setting-row--last">
                <div class="setting-left">
                  <span class="setting-label">当前状态</span>
                  <p class="setting-desc">{{ runtimeStatusText }}</p>
                </div>
              </div>
            </section>

            <!-- ── 3. 数据与隐私 ── -->
            <section v-else-if="activeSection === 'privacy'" class="settings-section">
              <h2>数据与隐私</h2>

              <!-- 对话保留 -->
              <div class="setting-row setting-row--first">
                <div class="setting-left">
                  <span class="setting-label">对话保留</span>
                  <p class="setting-desc">历史对话的自动保留时长</p>
                </div>
                <div class="setting-right">
                  <select v-model="chatRetentionDays" aria-label="对话保留" :disabled="savingSettings" @change="onRetentionChange">
                    <option value="30">30 天</option>
                    <option value="90">90 天</option>
                    <option value="180">180 天</option>
                    <option value="365">365 天</option>
                    <option value="forever">永久保留</option>
                  </select>
                </div>
              </div>

              <!-- 自动清理 -->
              <div class="setting-row">
                <div class="setting-left">
                  <span class="setting-label">自动清理</span>
                  <p class="setting-desc">过期对话与缓存文件到达保留期限后自动移除</p>
                </div>
                <div class="setting-right">
                  <label class="toggle" aria-label="自动清理开关">
                    <input v-model="chatAutoCleanup" type="checkbox" :disabled="savingSettings" @change="onAutoCleanupChange" />
                    <span class="track"><span class="thumb"></span></span>
                  </label>
                </div>
              </div>

              <!-- 导出数据 -->
              <div class="setting-row">
                <div class="setting-left">
                  <span class="setting-label">导出数据</span>
                  <p class="setting-desc">导出所有对话记录、配置快照与 Taiji 状态</p>
                </div>
                <div class="setting-right">
                  <button class="btn-sm btn-outline" :disabled="exporting" @click="onExportData">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M7 10l5 5 5-5"/><path d="M12 15V3"/></svg>
                    {{ exporting ? '导出中…' : '导出' }}
                  </button>
                </div>
              </div>

              <!-- 危险区 -->
              <div class="setting-row setting-row--last">
                <div class="danger-zone">
                  <h3>
                    <svg class="dz-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z"/><path d="M12 9v4M12 17h.01"/></svg>
                    危险操作
                  </h3>
                  <p>重置将清空所有本地对话会话记录；不会删除模型权重、检查点、Taiji 状态与配置项。此操作不可撤销，建议先导出数据再进行重置。</p>
                  <button class="btn-destructive" :disabled="resetting" @click="onResetSeed">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M9.5 9.5h5v6h-5z"/><path d="M14.5 9.5 13 7h-2l-1.5 2.5"/></svg>
                    {{ resetting ? '重置中…' : '重置Seed' }}
                  </button>
                </div>
              </div>
            </section>

            <!-- ── 4. 关于 ── -->
            <section v-else-if="activeSection === 'about'" class="settings-section">
              <h2>关于</h2>

              <!-- 版本元信息 -->
              <div class="setting-row setting-row--first">
                <div class="about-meta">
                  <div class="meta-line">
                    <span style="font-weight:600;">Seed神经元系统</span>
                    <span class="meta-tag">v{{ appVersion }}</span>
                  </div>
                  <div class="meta-line meta-line--muted">
                    原生基底
                    <span class="meta-tag">seed-native-v1</span>
                  </div>
                  <div class="meta-line meta-line--muted">
                    状态通路
                    <span class="meta-tag">Taiji runtime → native capabilities</span>
                  </div>
                </div>
              </div>

              <!-- 开源许可 -->
              <div class="setting-row setting-row--last">
                <div class="setting-left">
                  <span class="setting-label">开源许可</span>
                  <p class="setting-desc">查看本系统使用的第三方组件许可协议</p>
                </div>
                <div class="setting-right">
                  <button class="btn-sm btn-ghost" @click="showLicense = true">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6"/><path d="M12 18v-6"/><path d="M9 15h6"/></svg>
                    查看许可
                  </button>
                </div>
              </div>
            </section>

          </div><!-- /.settings-content -->
        </div><!-- /.settings-layout -->
      </div><!-- /.settings-wrap -->
    </div><!-- /.settings-scroll -->

    <!-- 开源许可弹窗（内嵌摘要，与根目录 LICENSE 一致：Apache-2.0） -->
    <div
      v-if="showLicense"
      class="license-overlay"
      role="dialog"
      aria-modal="true"
      aria-label="开源许可"
      @click.self="showLicense = false"
    >
      <div class="license-panel">
        <header class="license-head">
          <span class="license-title">开源许可</span>
          <button class="license-close" aria-label="关闭许可弹窗" @click="showLicense = false">✕</button>
        </header>
        <div class="license-body">
          <p class="license-main">本项目基于 <strong>Apache License 2.0</strong> 开源。</p>
          <p class="license-copy">Copyright 2026 NeuroPlex Contributors</p>
          <div class="license-cols">
            <div class="license-col">
              <h4>允许</h4>
              <ul>
                <li>商业使用、修改、分发</li>
                <li>私有使用、专利授权</li>
              </ul>
            </div>
            <div class="license-col">
              <h4>条件</h4>
              <ul>
                <li>保留许可证与版权声明</li>
                <li>修改文件需携带变更说明</li>
              </ul>
            </div>
            <div class="license-col">
              <h4>限制</h4>
              <ul>
                <li>不提供任何担保</li>
                <li>不承担使用责任</li>
              </ul>
            </div>
          </div>
          <p class="license-note">完整许可文本见项目根目录 LICENSE 文件，或访问
            <a href="http://www.apache.org/licenses/LICENSE-2.0" target="_blank" rel="noopener">apache.org/licenses/LICENSE-2.0</a>。</p>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { ref, reactive, inject } from 'vue';
import { useAppStore } from '../stores/appStore.js';
import { useChatStore } from '../stores/chatStore.js';
import { nativeApi } from '../composables/nativeApi.js';
import RuntimeEvidenceStrip from '../components/RuntimeEvidenceStrip.vue';

const toast = inject('toast');
const $confirm = inject('$confirm', () => Promise.resolve(false));
const appStore = useAppStore();
const chatStore = useChatStore();

// 左侧设置导航当前激活分区（v-if 切换右侧内容）
const activeSection = ref('general');

const runtimeStatusText = ref('检测中…');

const refreshRuntime = async () => {
  try {
    const d = await nativeApi.systemHealth();
    runtimeStatusText.value = d.seed_active
      ? 'Seed 原生运行时激活中（检查点：checkpoints/seed_corpus.pt，对话即持续学习）'
      : 'Seed 原生运行时未激活；请检查本地服务状态';
  } catch (e) {
    runtimeStatusText.value = `状态检测失败：${e.message}`;
  }
};

refreshRuntime();

// ── 持久化设置组（经 /api/settings 通道，范式对齐终端开关）──
// 初值读 localStorage（App.vue 启动时已将服务端值镜像到 taiji_<key>），
// 进入页面时用 GET /api/settings 校正；变更时 POST /api/settings 持久化，
// 成功回写缓存，失败回滚 + toast；userModifiedKeys 防 GET/POST 竞态。
// 键名约定：小写下划线，语义与后端 app_settings.json 一致。
const SETTINGS_DEFAULTS = {
  ui_language: 'zh-CN',                    // 默认语言（通用）
  timezone: 'Asia/Shanghai',               // 时区（通用）
  ui_density: 'default',                   // 界面密度（通用）
  taiji_activation_threshold: 0.72,        // 局部激活阈值（Taiji）
  taiji_response_timeout_ms: 100,          // 响应超时毫秒（Taiji）
  taiji_auto_consolidation: true,          // 自动巩固（Taiji）
  taiji_sleep_mode: false,                 // 睡眠模式（Taiji）
  chat_retention_days: '90',               // 对话保留（数据与隐私）
  chat_auto_cleanup: true,                 // 自动清理（数据与隐私）
};

// 从 localStorage 读初值，按默认值类型自动解析（镜像值由 App.vue 写入）
function _readLocal(key, fallback) {
  const raw = localStorage.getItem(`taiji_${key}`);
  if (raw === null || raw === '') return fallback;
  if (typeof fallback === 'boolean') return raw === 'true';
  if (typeof fallback === 'number') {
    const n = Number(raw);
    return Number.isFinite(n) ? n : fallback;
  }
  return raw;
}

const uiLanguage = ref(_readLocal('ui_language', SETTINGS_DEFAULTS.ui_language));
const timezone = ref(_readLocal('timezone', SETTINGS_DEFAULTS.timezone));
const uiDensity = ref(_readLocal('ui_density', SETTINGS_DEFAULTS.ui_density));
const activationThreshold = ref(
  _readLocal('taiji_activation_threshold', SETTINGS_DEFAULTS.taiji_activation_threshold)
);
const responseTimeoutMs = ref(
  _readLocal('taiji_response_timeout_ms', SETTINGS_DEFAULTS.taiji_response_timeout_ms)
);
const autoConsolidation = ref(
  _readLocal('taiji_auto_consolidation', SETTINGS_DEFAULTS.taiji_auto_consolidation)
);
const sleepMode = ref(_readLocal('taiji_sleep_mode', SETTINGS_DEFAULTS.taiji_sleep_mode));
const chatRetentionDays = ref(_readLocal('chat_retention_days', SETTINGS_DEFAULTS.chat_retention_days));
const chatAutoCleanup = ref(_readLocal('chat_auto_cleanup', SETTINGS_DEFAULTS.chat_auto_cleanup));

// 键 → ref 映射，供 GET 校正与失败回滚使用；confirmed 记录最后一次已确认值（回滚目标）
const settingRefs = {
  ui_language: uiLanguage,
  timezone,
  ui_density: uiDensity,
  taiji_activation_threshold: activationThreshold,
  taiji_response_timeout_ms: responseTimeoutMs,
  taiji_auto_consolidation: autoConsolidation,
  taiji_sleep_mode: sleepMode,
  chat_retention_days: chatRetentionDays,
  chat_auto_cleanup: chatAutoCleanup,
};
const confirmed = reactive(Object.fromEntries(
  Object.entries(settingRefs).map(([k, r]) => [k, r.value])
));

const savingSettings = ref(false);
// 用户已手动改过的键：迟到的 GET 响应不得用旧值覆盖（竞态防护）
const userModifiedKeys = new Set();

// 语言选项 → appStore 本地化语言（locales 仅支持 zh/en）双向映射
const _mapLangToStore = (lang) => (lang === 'en' ? 'en' : 'zh');
// 初始化同步：若 store 语言与已持久化偏好不一致，以持久化值为准（仅 zh/en 可映射）
if (uiLanguage.value === 'en') appStore.currentLang = 'en';

// 进入页面时拉取服务端设置并校正所有控件（含终端开关）
const refreshPersistedSettings = async () => {
  try {
    const d = await nativeApi.settingsGet();
    if (!d || typeof d !== 'object') return;
    for (const [key, target] of Object.entries(settingRefs)) {
      const v = d[key];
      const expectedType = typeof SETTINGS_DEFAULTS[key];
      if (userModifiedKeys.has(key) || v === undefined || typeof v !== expectedType) continue;
      target.value = v;
      confirmed[key] = v;
      localStorage.setItem(`taiji_${key}`, expectedType === 'string' ? v : JSON.stringify(v));
    }
    if (!userModifiedKeys.has('ui_language') && typeof d.ui_language === 'string') {
      appStore.currentLang = _mapLangToStore(d.ui_language);
    }
    if (
      !userModifiedKeys.has('terminal_allow_unauthenticated') &&
      typeof d.terminal_allow_unauthenticated === 'boolean'
    ) {
      terminalAllowUnauth.value = d.terminal_allow_unauthenticated;
      localStorage.setItem(
        'taiji_terminal_allow_unauthenticated',
        JSON.stringify(d.terminal_allow_unauthenticated)
      );
    }
  } catch (e) {
    // 拉取失败时沿用 localStorage 缓存值，不打断设置页
  }
};
refreshPersistedSettings();

// 通用保存：POST 单键，成功回写缓存，失败回滚 ref 并 toast
const saveSetting = async (key, next, prev) => {
  if (savingSettings.value) return;
  savingSettings.value = true;
  userModifiedKeys.add(key);
  try {
    await nativeApi.settingsSave({ [key]: next });
    confirmed[key] = next;
    localStorage.setItem(`taiji_${key}`, typeof next === 'string' ? next : JSON.stringify(next));
    toast('✅ 设置已保存', 'success');
  } catch (e) {
    settingRefs[key].value = prev;
    toast(`❌ 保存设置失败：${e.message}`, 'error');
  } finally {
    savingSettings.value = false;
  }
};

// ── 各控件变更处理（prev 均取自最后一次已确认值）──
const onUiLanguageChange = () => {
  const prev = confirmed.ui_language;
  appStore.currentLang = _mapLangToStore(uiLanguage.value);
  saveSetting('ui_language', uiLanguage.value, prev);
};
const onTimezoneChange = () => saveSetting('timezone', timezone.value, confirmed.timezone);
const onDensityChange = () => saveSetting('ui_density', uiDensity.value, confirmed.ui_density);
const onThresholdChange = () => {
  let v = Number(activationThreshold.value);
  if (!Number.isFinite(v)) v = SETTINGS_DEFAULTS.taiji_activation_threshold;
  v = Math.min(1, Math.max(0, v));
  activationThreshold.value = v;
  saveSetting('taiji_activation_threshold', v, confirmed.taiji_activation_threshold);
};
const onResponseTimeoutChange = () => {
  let v = Math.round(Number(responseTimeoutMs.value));
  if (!Number.isFinite(v)) v = SETTINGS_DEFAULTS.taiji_response_timeout_ms;
  v = Math.min(10000, Math.max(10, v));
  responseTimeoutMs.value = v;
  saveSetting('taiji_response_timeout_ms', v, confirmed.taiji_response_timeout_ms);
};
const onAutoConsolidationChange = () => {
  const next = autoConsolidation.value;
  saveSetting('taiji_auto_consolidation', next, !next);
};
const onSleepModeChange = () => {
  const next = sleepMode.value;
  saveSetting('taiji_sleep_mode', next, !next);
};
const onRetentionChange = () =>
  saveSetting('chat_retention_days', chatRetentionDays.value, confirmed.chat_retention_days);
const onAutoCleanupChange = () => {
  const next = chatAutoCleanup.value;
  saveSetting('chat_auto_cleanup', next, !next);
};

// ── 导出数据：聚合会话列表（chatStore）+ 服务端设置，Blob 下载 ──
const exporting = ref(false);
const onExportData = async () => {
  if (exporting.value) return;
  exporting.value = true;
  try {
    let settings = {};
    try {
      settings = await nativeApi.settingsGet();
    } catch (e) { /* 设置拉取失败不阻断导出，降级为空对象 */ }
    const payload = {
      app: 'Seed',
      exported_at: new Date().toISOString(),
      settings,
      chat_sessions: chatStore.sessions,
    };
    const d = new Date();
    const dateStr = `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}`;
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `seed-export-${dateStr}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    toast('✅ 数据已导出', 'success');
  } catch (e) {
    toast(`❌ 导出失败：${e.message}`, 'error');
  } finally {
    exporting.value = false;
  }
};

// ── 重置 Seed：二次确认后调用 POST /api/system/reset（仅清空对话会话）──
// 后端语义边界：不动模型权重 / checkpoints / Taiji 状态 / 配置项（见 routes_system.py）
const resetting = ref(false);
const onResetSeed = async () => {
  if (resetting.value) return;
  const ok = await $confirm({
    title: '⚠️ 重置 Seed',
    message:
      '将清空所有本地对话会话记录，此操作不可撤销。\n\n模型权重、检查点、Taiji 状态与配置项不受影响。\n建议先通过“导出数据”备份后再继续。',
    type: 'danger',
  });
  if (!ok) return;
  resetting.value = true;
  try {
    const d = await nativeApi.systemReset({ scope: 'chat_sessions' });
    if (d.status !== 'ok') throw new Error(d.detail || d.message || '重置失败');
    toast(`✅ 重置完成：${d.message || '已清空对话会话'}`, 'success');
    // 重置后刷新前端会话列表并重建新会话，避免引用已删除的会话
    await chatStore.loadSessions();
    if (chatStore.sessions.length === 0) chatStore.createNewSession();
  } catch (e) {
    toast(`❌ 重置失败：${e.message}`, 'error');
  } finally {
    resetting.value = false;
  }
};

// ── 开源许可弹窗（内嵌摘要，与根目录 LICENSE 一致：Apache-2.0）──
const showLicense = ref(false);

// ── 安全：允许未认证终端访问（经 /api/settings 通道持久化）──
// 读写模式对齐上方持久化设置组：初值读 localStorage（taiji_ 前缀缓存），
// 进入页面时用 GET /api/settings 校正（见 refreshPersistedSettings）；
// 切换时 POST /api/settings 持久化，成功后回写缓存，失败则 toast 并回滚。
const terminalAllowUnauth = ref(
  localStorage.getItem('taiji_terminal_allow_unauthenticated') === 'true'
);
const savingTerminalSetting = ref(false);

const onTerminalUnauthChange = async () => {
  userModifiedKeys.add('terminal_allow_unauthenticated');
  const next = terminalAllowUnauth.value;
  const prev = !next;
  if (savingTerminalSetting.value) return;
  savingTerminalSetting.value = true;
  try {
    await nativeApi.settingsSave({ terminal_allow_unauthenticated: next });
    localStorage.setItem('taiji_terminal_allow_unauthenticated', JSON.stringify(next));
    toast(next ? '✅ 已允许未认证终端访问' : '✅ 已关闭未认证终端访问', 'success');
  } catch (e) {
    terminalAllowUnauth.value = prev;
    toast(`❌ 保存终端设置失败：${e.message}`, 'error');
  } finally {
    savingTerminalSetting.value = false;
  }
};

// R5: systemPrompt/updateChecking/updateAvailable/updateMsg 仅被已移除的函数引用，
// 模板未使用，一并移除（需要时从 git 历史恢复）。
const appVersion = ref('1.0.0');


// R5: onBgImageSelect/saveSettings/checkUpdate/applyUpdate 未被模板引用，已移除；
// 需要时从 git 历史恢复并接入对应按钮。

// Load app version
(async () => {
  try {
    const d = await nativeApi.systemVersion();
    if (d.version) appVersion.value = d.version;
  } catch (e) {}
})();
</script>

<style scoped>
/* ═══ Seed控制台 · 设置页专属样式（对齐画布 settings.html · 豆包设计 token）═══ */

/* --- 视图外壳：覆盖 dedicated-view 的内边距，改为顶栏 + 滚动区 --- */
.settings-view {
  padding: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* --- 顶栏 --- */
/* 不画 border-bottom：外围边框由 .router-wrapper 独占（见 styles/shell.css） */
.settings-topbar {
  flex: none;
  height: 52px;
  padding: 0 18px;
  display: flex;
  align-items: center;
  gap: 12px;
  background: transparent;
}
.topbar-title-wrap {
  display: flex;
  flex-direction: column;
  gap: 0;
  line-height: 1.2;
}
.topbar-title {
  font-size: 0.94rem;
  font-weight: 650;
  color: var(--foreground);
}
.topbar-sub {
  font-size: 0.72rem;
  color: var(--muted-foreground);
  margin-top: 1px;
}
.topbar-spacer {
  flex: 1;
}

/* --- 滚动区 + 容器 --- */
.settings-scroll {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
}
.settings-wrap {
  width: 100%;
  max-width: 760px;
  margin: 0 auto;
  padding: 28px 24px 48px;
}

/* --- 双栏布局：左侧导航 + 右侧内容 --- */
.settings-layout {
  display: grid;
  grid-template-columns: 168px minmax(0, 1fr);
  gap: 32px;
  align-items: start;
}

/* --- 左侧 sticky 导航 --- */
.settings-nav {
  position: sticky;
  top: 16px;
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding-top: 6px;
}
.sn-item {
  width: 100%;
  border: 0;
  background: transparent;
  color: var(--muted-foreground);
  border-radius: calc(var(--radius) * 0.45);
  padding: 9px 14px;
  text-align: left;
  font-size: 0.88rem;
  font-weight: 450;
  line-height: 1.35;
  cursor: pointer;
  transition: background 150ms ease, color 150ms ease, font-weight 150ms ease;
  display: flex;
  align-items: center;
  gap: 9px;
}
.sn-item:hover {
  background: var(--muted);
  color: var(--foreground);
}
.sn-item.active {
  background: color-mix(in srgb, var(--primary) 12%, transparent);
  color: var(--primary);
  font-weight: 600;
}
.sn-icon {
  width: 17px;
  height: 17px;
  flex: none;
  stroke: currentColor;
  fill: none;
  stroke-width: 1.7;
  stroke-linecap: round;
  stroke-linejoin: round;
  opacity: 0.7;
}
.sn-item.active .sn-icon {
  opacity: 1;
}

/* --- 右侧内容区 --- */
.settings-content {
  min-width: 0;
}
.settings-section {
  display: flex;
  flex-direction: column;
}
.settings-section h2 {
  font-size: 1.15rem;
  font-weight: 650;
  margin: 0 0 18px;
  color: var(--foreground);
  letter-spacing: -0.01em;
}

/* --- 设置行 --- */
.setting-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 0;
  border-bottom: 1px solid var(--border);
  gap: 16px;
}
.setting-row--first {
  padding-top: 0;
}
.setting-row--last {
  border-bottom: none;
  padding-bottom: 0;
}
.setting-row--col {
  flex-direction: column;
  align-items: stretch;
  gap: 12px;
}
.setting-left {
  min-width: 0;
  flex: 1;
}
.setting-right {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: 8px;
}
.setting-label {
  font-size: 0.88rem;
  font-weight: 500;
  color: var(--foreground);
  line-height: 1.3;
}
.setting-desc {
  font-size: 0.76rem;
  color: var(--muted-foreground);
  margin: 3px 0 0;
  line-height: 1.4;
}

/* --- 表单控件基础 --- */
.setting-right select,
.setting-right input[type="number"],
.setting-right input[type="text"] {
  background: var(--background);
  color: var(--foreground);
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) * 0.45);
  padding: 7px 12px;
  font-size: 0.86rem;
  font-family: var(--font-sans);
  outline: none;
  transition: border-color 160ms ease, box-shadow 160ms ease;
}
.setting-right select:focus,
.setting-right input[type="number"]:focus,
.setting-right input[type="text"]:focus {
  border-color: var(--ring);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--ring) 16%, transparent);
}
.setting-right select {
  padding-right: 32px;
  appearance: none;
  -webkit-appearance: none;
  color: var(--muted-foreground);
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%237f8d9f' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 10px center;
  background-size: 15px;
  cursor: pointer;
}
.setting-right input[type="number"] {
  width: 90px;
  text-align: center;
  font-variant-numeric: tabular-nums;
}

/* --- Radio 组（界面密度）--- */
.radio-group {
  display: flex;
  gap: 4px;
  background: var(--muted);
  border-radius: calc(var(--radius) * 0.5);
  padding: 3px;
}
.radio-chip {
  position: relative;
  cursor: pointer;
}
.radio-chip input {
  position: absolute;
  opacity: 0;
  width: 0;
  height: 0;
}
.radio-chip .rc-label {
  display: block;
  padding: 5px 13px;
  border-radius: calc(var(--radius) * 0.4);
  font-size: 0.8rem;
  font-weight: 450;
  color: var(--muted-foreground);
  transition: background 150ms ease, color 150ms ease, font-weight 150ms ease;
  user-select: none;
  white-space: nowrap;
}
.radio-chip input:checked + .rc-label {
  background: var(--background);
  color: var(--foreground);
  font-weight: 600;
  box-shadow: 0 1px 2px color-mix(in srgb, var(--foreground) 8%, transparent);
}
.radio-chip:hover .rc-label {
  color: var(--foreground);
}

/* --- Range 滑块 --- */
.range-wrap {
  display: flex;
  align-items: center;
  gap: 10px;
}
.range-value {
  font-size: 0.82rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  color: var(--primary);
  min-width: 42px;
  text-align: right;
}
.setting-right input[type="range"] {
  -webkit-appearance: none;
  appearance: none;
  width: 140px;
  height: 6px;
  background: var(--border);
  border-radius: 999px;
  border: 0;
  padding: 0;
  outline: none;
  cursor: pointer;
}
.setting-right input[type="range"]:focus {
  box-shadow: none;
}
.setting-right input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--primary);
  cursor: pointer;
  border: 2px solid var(--background);
  box-shadow: 0 1px 4px color-mix(in srgb, var(--foreground) 18%, transparent);
  transition: transform 120ms ease;
}
.setting-right input[type="range"]::-webkit-slider-thumb:hover {
  transform: scale(1.15);
}
.setting-right input[type="range"]::-moz-range-thumb {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--primary);
  cursor: pointer;
  border: 2px solid var(--background);
  box-shadow: 0 1px 4px color-mix(in srgb, var(--foreground) 18%, transparent);
}

/* --- Toggle 开关 --- */
.toggle {
  position: relative;
  display: inline-block;
  width: 42px;
  height: 24px;
  flex-shrink: 0;
  cursor: pointer;
}
.toggle input {
  opacity: 0;
  width: 0;
  height: 0;
  position: absolute;
  border: 0;
  padding: 0;
}
.toggle .track {
  position: absolute;
  inset: 0;
  background: var(--border);
  border-radius: 999px;
  cursor: pointer;
  transition: background 200ms ease;
}
.toggle .thumb {
  position: absolute;
  top: 3px;
  left: 3px;
  width: 18px;
  height: 18px;
  background: var(--background);
  border-radius: 50%;
  transition: transform 200ms cubic-bezier(0.34, 1.56, 0.64, 1);
  box-shadow: 0 1px 3px color-mix(in srgb, var(--foreground) 16%, transparent);
}
.toggle input:checked + .track {
  background: var(--primary);
}
.toggle input:checked + .track .thumb {
  transform: translateX(18px);
}
.toggle input:focus-visible + .track {
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--ring) 20%, transparent);
}

/* --- 主题预览网格 --- */
.theme-previews {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 10px;
}
.theme-preview-card {
  border: 2px solid var(--border);
  border-radius: 12px;
  padding: 8px 8px 10px;
  cursor: pointer;
  background: var(--background);
  text-align: center;
  transition: border-color 160ms ease, box-shadow 160ms ease, transform 140ms ease;
}
.theme-preview-card:hover {
  border-color: color-mix(in srgb, var(--primary) 35%, var(--border));
  transform: translateY(-1px);
}
.theme-preview-card.active {
  border-color: var(--primary);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--ring) 18%, transparent);
}
.theme-swatch {
  width: 100%;
  height: 36px;
  border-radius: 8px;
  display: block;
}
.theme-name {
  font-size: 0.72rem;
  margin-top: 7px;
  color: var(--muted-foreground);
  font-weight: 450;
  transition: color 150ms ease;
}
.theme-preview-card.active .theme-name {
  color: var(--foreground);
  font-weight: 600;
}

/* --- 小号按钮（导出 / 查看许可）--- */
.btn-sm {
  height: 32px;
  padding: 0 13px;
  font-size: 0.8rem;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: 1px solid transparent;
  cursor: pointer;
  font-weight: 500;
  transition: background 150ms ease, border-color 150ms ease, transform 120ms ease;
}
.btn-sm:active {
  transform: translateY(1px);
}
.btn-sm:focus-visible {
  outline: 2px solid var(--ring);
  outline-offset: 2px;
}
.btn-sm.btn-outline {
  background: var(--background);
  color: var(--foreground);
  border-color: var(--border);
}
.btn-sm.btn-outline:hover {
  background: var(--muted);
}
.btn-sm.btn-ghost {
  background: var(--muted);
  color: var(--foreground);
}
.btn-sm.btn-ghost:hover {
  background: color-mix(in srgb, var(--muted) 80%, var(--foreground) 14%);
}

/* --- 危险区 --- */
.danger-zone {
  width: 100%;
  border: 1px solid var(--destructive);
  border-radius: 13px;
  padding: 20px;
  margin-top: 8px;
  background: color-mix(in srgb, var(--destructive) 4%, transparent);
}
.danger-zone h3 {
  color: var(--destructive);
  margin: 0 0 6px;
  font-size: 0.94rem;
  font-weight: 650;
  display: flex;
  align-items: center;
  gap: 7px;
}
.danger-zone .dz-icon {
  width: 18px;
  height: 18px;
  flex: none;
}
.danger-zone p {
  font-size: 0.78rem;
  color: var(--muted-foreground);
  margin: 0 0 14px;
  line-height: 1.5;
}
.btn-destructive {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  height: 36px;
  padding: 0 16px;
  border: 1px solid transparent;
  border-radius: 999px;
  background: var(--destructive);
  color: var(--destructive-foreground);
  font-size: 0.85rem;
  font-weight: 500;
  cursor: pointer;
  transition: background 150ms ease, transform 120ms ease;
}
.btn-destructive:hover {
  background: color-mix(in srgb, var(--destructive) 88%, var(--foreground) 12%);
}
.btn-destructive:active {
  transform: translateY(1px);
}
.btn-destructive:focus-visible {
  outline: 2px solid var(--ring);
  outline-offset: 2px;
}

/* --- 关于区块元信息 --- */
.about-meta {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 6px 0;
}
.about-meta .meta-line {
  font-size: 0.86rem;
  color: var(--foreground);
  display: flex;
  align-items: center;
  gap: 6px;
}
.about-meta .meta-line--muted {
  color: var(--muted-foreground);
  font-size: 0.8rem;
}
.about-meta .meta-tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 6px;
  font-size: 0.72rem;
  font-weight: 600;
  background: var(--muted);
  color: var(--muted-foreground);
  font-family: var(--font-mono);
}

/* --- 开源许可弹窗 --- */
.license-overlay {
  position: fixed;
  inset: 0;
  z-index: 1000;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  background: color-mix(in srgb, var(--foreground) 32%, transparent);
  backdrop-filter: blur(2px);
  animation: license-fade 160ms ease;
}
@keyframes license-fade {
  from { opacity: 0; }
  to { opacity: 1; }
}
.license-panel {
  width: min(560px, 100%);
  max-height: min(72vh, 640px);
  display: flex;
  flex-direction: column;
  background: var(--background);
  border: 1px solid var(--border);
  border-radius: 14px;
  box-shadow: 0 12px 40px color-mix(in srgb, var(--foreground) 18%, transparent);
  overflow: hidden;
  animation: license-pop 180ms cubic-bezier(0.34, 1.4, 0.64, 1);
}
@keyframes license-pop {
  from { opacity: 0; transform: translateY(8px) scale(0.98); }
  to { opacity: 1; transform: translateY(0) scale(1); }
}
.license-head {
  flex: none;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 18px;
  border-bottom: 1px solid var(--border);
}
.license-title {
  font-size: 0.92rem;
  font-weight: 650;
  color: var(--foreground);
}
.license-close {
  width: 28px;
  height: 28px;
  border: 0;
  border-radius: 8px;
  background: transparent;
  color: var(--muted-foreground);
  font-size: 0.8rem;
  cursor: pointer;
  transition: background 140ms ease, color 140ms ease;
}
.license-close:hover {
  background: var(--muted);
  color: var(--foreground);
}
.license-body {
  padding: 18px;
  overflow-y: auto;
  font-size: 0.82rem;
  color: var(--foreground);
  line-height: 1.6;
}
.license-main {
  margin: 0 0 4px;
  font-size: 0.88rem;
}
.license-copy {
  margin: 0 0 16px;
  color: var(--muted-foreground);
  font-family: var(--font-mono);
  font-size: 0.74rem;
}
.license-cols {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  margin-bottom: 16px;
}
.license-col {
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px;
  background: color-mix(in srgb, var(--muted) 50%, transparent);
}
.license-col h4 {
  margin: 0 0 8px;
  font-size: 0.78rem;
  font-weight: 650;
  color: var(--primary);
}
.license-col ul {
  margin: 0;
  padding-left: 16px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  color: var(--muted-foreground);
  font-size: 0.76rem;
}
.license-note {
  margin: 0;
  color: var(--muted-foreground);
  font-size: 0.76rem;
}
.license-note a {
  color: var(--primary);
  text-decoration: none;
}
.license-note a:hover {
  text-decoration: underline;
}
@media (max-width: 560px) {
  .license-cols {
    grid-template-columns: 1fr;
  }
}

/* --- 响应式 --- */
@media (max-width: 880px) {
  .settings-layout {
    grid-template-columns: 1fr;
    gap: 0;
  }
  .settings-nav {
    position: static;
    flex-direction: row;
    flex-wrap: wrap;
    gap: 4px;
    padding: 0 0 16px;
    margin-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }
  .sn-item {
    width: auto;
    padding: 7px 12px;
    font-size: 0.8rem;
  }
  .theme-previews {
    grid-template-columns: repeat(3, 1fr);
  }
}

@media (max-width: 560px) {
  .theme-previews {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 500px) {
  .settings-wrap {
    padding: 16px 12px 32px;
  }
  .setting-row:not(.setting-row--col) {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }
  .setting-right {
    width: 100%;
    justify-content: flex-start;
  }
}
</style>
