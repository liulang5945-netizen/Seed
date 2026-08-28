<template>
  <div v-if="trainState === 'running' || trainState === 'paused'" class="train-ctrl-bar">
    <div class="ctrl-row">
      <n-button v-if="trainState === 'running'" type="warning" round @click="emit('pause')">
        <template #icon><Pause :size="14" /></template>{{ t('pause_training') }}
      </n-button>
      <n-button v-if="trainState === 'paused'" type="primary" round @click="emit('resume')">
        <template #icon><Play :size="14" /></template>{{ t('resume_training') }}
      </n-button>
      <n-button type="error" round @click="emit('stop')">
        <template #icon><StopCircle :size="14" /></template>{{ t('stop_training') }}
      </n-button>
    </div>
  </div>
</template>

<script setup>
import { Pause, Play, StopCircle } from 'lucide-vue-next'

defineProps({
  trainState: { type: String, default: 'idle' },
  t: { type: Function, required: true },
})

const emit = defineEmits(['pause', 'resume', 'stop'])
</script>

<style scoped>
.train-ctrl-bar { position: sticky; bottom: 0; margin-top: 24px; padding: 14px 18px; background: var(--card, var(--bg-card)); border: 1px solid var(--border); border-radius: calc(var(--radius) * 0.7); box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.06); z-index: 10; }
.ctrl-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
</style>
