<template>
  <section class="native-taiji-status">
    <div class="native-status-head">
      <div>
        <span class="eyebrow">TAIJI NATIVE SUBSTRATE</span>
        <h1>原生状态通路</h1>
        <p>当前客户端展示的是持续状态与局部可塑性运行态，不是 Transformer 的 token 统计面板。</p>
      </div>
      <span class="native-status-pill">{{ connectionStatus }}</span>
    </div>

    <div class="native-contract-grid">
      <article class="native-contract-card">
        <span class="native-contract-label">运行时</span>
        <strong>{{ modelName || 'Seed native' }}</strong>
        <span>当前 Taiji 原生运行时身份</span>
      </article>
      <article class="native-contract-card">
        <span class="native-contract-label">语言器官</span>
        <strong>{{ languageProviderState || 'unknown' }}</strong>
        <span>{{ languageProviderBackend || '未提供 provider artifact' }}</span>
      </article>
      <article class="native-contract-card">
        <span class="native-contract-label">工作台</span>
        <strong>{{ toolCount }} 项能力</strong>
        <span>{{ workbenchDetail || '来自当前 capability snapshot' }}</span>
      </article>
      <article class="native-contract-card">
        <span class="native-contract-label">生命状态</span>
        <strong>{{ hasNeedsData ? '已上报' : '未上报' }}</strong>
        <span>needs 是否由当前运行时提供</span>
      </article>
    </div>

    <div class="native-pipeline" aria-label="Taiji 原生状态推进链路">
      <div class="native-pipeline-step"><span>01</span><strong>runtime</strong><small>连接与身份</small></div>
      <span class="native-pipeline-arrow">→</span>
      <div class="native-pipeline-step"><span>02</span><strong>input frame</strong><small>带来源输入</small></div>
      <span class="native-pipeline-arrow">→</span>
      <div class="native-pipeline-step"><span>03</span><strong>state update</strong><small>持续状态</small></div>
      <span class="native-pipeline-arrow">→</span>
      <div class="native-pipeline-step"><span>04</span><strong>language organ</strong><small>可读表达</small></div>
    </div>

    <div class="native-status-note">
      <strong>当前运行说明</strong>
      <p>{{ healthMessage || 'Taiji 原生运行时已连接；当前页面只显示已由状态接口上报的事实。' }}</p>
      <p>需求、学习细节与结构规模尚未通过公开状态合同上报，因此这里不推测突触、神经元数量或内部器官名称。</p>
    </div>
  </section>
</template>

<script setup>
defineOptions({ name: 'LifeNativeStatusPanel' })

defineProps({
  connectionStatus: { type: String, default: '' },
  modelName: { type: String, default: '' },
  languageProviderState: { type: String, default: '' },
  languageProviderBackend: { type: String, default: '' },
  toolCount: { type: Number, default: 0 },
  workbenchDetail: { type: String, default: '' },
  hasNeedsData: { type: Boolean, default: false },
  healthMessage: { type: String, default: '' },
})
</script>

<style scoped>
.native-taiji-status { max-width: 1080px; margin: 0 auto; padding: 28px 30px 40px; }
.native-status-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 24px; margin-bottom: 24px; }
.native-status-head .eyebrow { color: var(--primary); font-size: 0.7rem; font-weight: 700; letter-spacing: 0.16em; }
.native-status-head h1 { margin: 8px 0 6px; color: var(--foreground); font-size: 1.7rem; }
.native-status-head p { margin: 0; color: var(--muted-foreground); line-height: 1.6; }
.native-status-pill { flex: none; padding: 7px 12px; border: 1px solid color-mix(in srgb, var(--chart-2) 32%, var(--border)); border-radius: 999px; color: var(--chart-2); background: color-mix(in srgb, var(--chart-2) 9%, var(--card)); font-size: 0.78rem; }
.native-contract-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.native-contract-card, .native-status-note { padding: 18px; border: 1px solid var(--border); border-radius: 14px; background: var(--card); }
.native-contract-card { display: flex; min-height: 118px; flex-direction: column; gap: 7px; }
.native-contract-label { color: var(--muted-foreground); font-size: 0.74rem; }
.native-contract-card strong { color: var(--foreground); font-size: 1rem; }
.native-contract-card > span:last-child { color: var(--muted-foreground); font-size: 0.78rem; line-height: 1.5; }
.native-pipeline { display: flex; align-items: stretch; gap: 10px; margin: 18px 0; padding: 18px; border: 1px solid var(--border); border-radius: 14px; background: color-mix(in srgb, var(--accent) 35%, var(--card)); }
.native-pipeline-step { display: flex; flex: 1; min-width: 0; flex-direction: column; gap: 5px; }
.native-pipeline-step > span { color: var(--primary); font-size: 0.68rem; font-weight: 700; }
.native-pipeline-step strong { overflow-wrap: anywhere; color: var(--foreground); font-size: 0.84rem; }
.native-pipeline-step small { color: var(--muted-foreground); font-size: 0.72rem; }
.native-pipeline-arrow { align-self: center; color: var(--muted-foreground); font-size: 1.1rem; }
.native-status-note strong { color: var(--foreground); font-size: 0.88rem; }
.native-status-note p { margin: 8px 0 0; color: var(--muted-foreground); font-size: 0.8rem; line-height: 1.6; }
@media (max-width: 900px) { .native-contract-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 620px) {
  .native-taiji-status { padding: 20px 16px 30px; }
  .native-status-head { flex-direction: column; }
  .native-contract-grid { grid-template-columns: 1fr; }
  .native-pipeline { flex-direction: column; }
  .native-pipeline-arrow { transform: rotate(90deg); align-self: flex-start; }
}
</style>
