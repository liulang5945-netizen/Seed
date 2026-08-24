<template>
  <img
    :width="size"
    :height="size"
    src="/seed-taiji-network.png"
    alt="Seed Taiji"
    role="img"
    class="taiji-logo"
    :style="{ width: size + 'px', height: size + 'px' }"
    :class="{ 'is-thinking': thinking, 'is-idle': !thinking }"
  />
</template>

<script setup>
defineProps({
  size: { type: [Number, String], default: 40 },
  thinking: { type: Boolean, default: false },
})
</script>

<style scoped>
.taiji-logo {
  display: block;
  flex: none;
  object-fit: contain;
  border-radius: 50%;
  background: #f5f5f0;
  box-shadow: 0 4px 14px rgba(26, 26, 26, 0.12);
  transform-origin: center;
  animation: taiji-logo-flow 36s linear infinite;
}

.taiji-logo.is-thinking {
  animation-duration: 24s;
}

@keyframes taiji-logo-flow {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

@media (prefers-reduced-motion: reduce) {
  .taiji-logo,
  .taiji-logo.is-thinking {
    animation: none;
  }
}
</style>
