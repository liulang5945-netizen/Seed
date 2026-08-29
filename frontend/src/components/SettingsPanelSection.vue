<template>
  <section class="settings-section">
    <h2>{{ title }}</h2>
    <slot />
  </section>
</template>

<script setup>
defineOptions({ name: 'SettingsPanelSection' })

defineProps({
  title: { type: String, required: true },
})
</script>
