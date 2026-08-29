<template>
  <section class="dedicated-view kb-view">
    <div class="kb-page">
      <header class="page-header">
        <div>
          <p class="eyebrow">TAIJI NATIVE SURFACE</p>
          <h1>知识库</h1>
          <p class="subtitle">知识输入将在原生观察与记忆能力接入后，通过同一条 Taiji capability 边界进入运行时。</p>
        </div>
        <button class="btn btn-outline" :disabled="refreshing" @click="refreshRuntime">
          <span class="refresh-icon" :class="{ spin: refreshing }">↻</span>
          {{ refreshing ? '刷新中…' : '刷新状态' }}
        </button>
      </header>

      <div class="boundary-card" :class="{ ready: knowledgeCapabilities.length }">
        <div class="boundary-icon">◎</div>
        <div class="boundary-copy">
          <span class="status-chip" :class="knowledgeCapabilities.length ? 'ok' : 'run'">
            {{ knowledgeCapabilities.length ? '原生能力已接入' : '原生能力待接入' }}
          </span>
          <h2>{{ knowledgeCapabilities.length ? '知识输入可以进入 Taiji 工作流' : '当前阶段暂不启用知识库操作' }}</h2>
          <p v-if="knowledgeCapabilities.length">客户端会依据运行时返回的 capability 显示可用操作。</p>
          <p v-else>旧 RAG 接口已从默认客户端路径移除，当前不会执行 Legacy 写入、修改索引或伪造检索结果。</p>
        </div>
      </div>

      <div class="evidence-grid">
        <article class="evidence-card">
          <span class="evidence-label">来源</span>
          <strong>Runtime capability snapshot</strong>
          <small>{{ snapshotId || '尚未读取' }}</small>
        </article>
        <article class="evidence-card">
          <span class="evidence-label">知识能力</span>
          <strong>{{ knowledgeCapabilities.length ? `${knowledgeCapabilities.length} 项已发现` : '0 项已发现' }}</strong>
          <small>仅显示运行时真实上报的能力</small>
        </article>
        <article class="evidence-card">
          <span class="evidence-label">当前边界</span>
          <strong>Taiji Native</strong>
          <small>Legacy RAG 仅保留为显式离线对照</small>
        </article>
      </div>

      <div v-if="knowledgeCapabilities.length" class="capability-list">
        <div class="section-heading">
          <h2>已发现的知识能力</h2>
          <span>{{ runtimeStore.toolError || '来自当前 runtime snapshot' }}</span>
        </div>
        <article v-for="capability in knowledgeCapabilities" :key="capability.name" class="capability-row">
          <div>
            <strong>{{ capability.name }}</strong>
            <p>{{ capability.description || '运行时未提供描述' }}</p>
          </div>
          <span class="status-chip" :class="capability.enabled ? 'ok' : 'run'">{{ capability.enabled ? '可用' : '未启用' }}</span>
        </article>
      </div>

      <div v-else class="next-boundary">
        <span class="next-marker">NEXT NATIVE BOUNDARY</span>
        <p>下一阶段应先定义 knowledge ingest / retrieval 的输入、来源证明、索引状态与删除语义，再恢复客户端操作面。</p>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed, ref } from 'vue'
import { useRuntimeStore } from '../stores/runtimeStore.js'

defineOptions({ name: 'KBView' })

const runtimeStore = useRuntimeStore()
const refreshing = ref(false)
const snapshotId = computed(() => runtimeStore.runtimeSnapshot?.tools?.snapshot_id
  || runtimeStore.runtimeSnapshot?.tools?.tools?.[0]?.source_id
  || '')
const knowledgeCapabilities = computed(() => runtimeStore.tools.filter(tool => (
  String(tool.name || '').startsWith('knowledge.')
)))

async function refreshRuntime() {
  refreshing.value = true
  try {
    await runtimeStore.refreshRuntime()
  } finally {
    refreshing.value = false
  }
}
</script>

<style scoped>
.kb-view { min-height: 100%; }
.kb-page { max-width: 940px; margin: 0 auto; padding: 34px; }
.page-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 24px; margin-bottom: 28px; }
.eyebrow, .next-marker, .evidence-label { color: var(--muted-foreground); font-size: .68rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
.eyebrow { margin: 0 0 8px; color: var(--primary); }
h1, h2, p { margin-top: 0; }
h1 { margin-bottom: 7px; color: var(--foreground); font-size: 1.65rem; letter-spacing: -.03em; }
.subtitle { max-width: 660px; margin-bottom: 0; color: var(--muted-foreground); line-height: 1.6; }
.refresh-icon { display: inline-block; margin-right: 5px; font-size: 1rem; }
.spin { animation: kb-spin .8s linear infinite; }
.boundary-card { display: flex; gap: 18px; align-items: center; padding: 25px; border: 1px solid color-mix(in srgb, var(--primary) 24%, var(--border)); border-radius: 18px; background: linear-gradient(135deg, color-mix(in srgb, var(--primary) 10%, var(--card)), var(--card)); }
.boundary-card.ready { border-color: color-mix(in srgb, var(--chart-2) 42%, var(--border)); }
.boundary-icon { display: grid; width: 48px; height: 48px; flex: 0 0 48px; place-items: center; border-radius: 50%; color: var(--primary); background: color-mix(in srgb, var(--primary) 14%, transparent); font-size: 1.8rem; }
.boundary-copy h2 { margin: 10px 0 6px; color: var(--foreground); font-size: 1.06rem; }
.boundary-copy p { margin-bottom: 0; color: var(--muted-foreground); line-height: 1.55; }
.status-chip { display: inline-flex; align-items: center; padding: 4px 8px; border-radius: 999px; color: var(--muted-foreground); background: color-mix(in srgb, var(--muted) 36%, transparent); font-size: .72rem; }
.status-chip.ok { color: var(--chart-2); background: color-mix(in srgb, var(--chart-2) 12%, transparent); }
.status-chip.run { color: var(--primary); background: color-mix(in srgb, var(--primary) 11%, transparent); }
.evidence-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 14px; }
.evidence-card, .next-boundary, .capability-list { border: 1px solid var(--border); border-radius: 14px; background: var(--card); }
.evidence-card { display: flex; min-height: 98px; flex-direction: column; gap: 7px; padding: 17px; }
.evidence-card strong { overflow: hidden; color: var(--foreground); font-size: .91rem; text-overflow: ellipsis; white-space: nowrap; }
.evidence-card small { overflow: hidden; color: var(--muted-foreground); text-overflow: ellipsis; white-space: nowrap; }
.next-boundary { margin-top: 14px; padding: 19px; }
.next-boundary p { margin: 8px 0 0; color: var(--muted-foreground); line-height: 1.6; }
.capability-list { margin-top: 14px; padding: 20px; }
.section-heading { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-bottom: 12px; }
.section-heading h2 { margin-bottom: 0; color: var(--foreground); font-size: 1rem; }
.section-heading span { color: var(--muted-foreground); font-size: .78rem; }
.capability-row { display: flex; align-items: center; justify-content: space-between; gap: 18px; padding: 13px 0; border-top: 1px solid var(--border); }
.capability-row strong { color: var(--foreground); font-size: .9rem; }
.capability-row p { margin: 4px 0 0; color: var(--muted-foreground); font-size: .8rem; }
@keyframes kb-spin { to { transform: rotate(360deg); } }
@media (max-width: 680px) {
  .kb-page { padding: 22px 16px; }
  .page-header { flex-direction: column; }
  .evidence-grid { grid-template-columns: 1fr; }
}
</style>
