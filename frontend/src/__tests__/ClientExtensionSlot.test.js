import { nextTick, ref } from 'vue'
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import ClientExtensionSlot from '../components/ClientExtensionSlot.vue'

function extensionContext(manifests, lifecycle = []) {
  const active = ref(manifests)
  return {
    snapshot: ref({ snapshot_id: 'snapshot-1' }),
    lifecycle: ref(lifecycle),
    slotManifests: (slotName) => active.value.filter((manifest) => manifest.slots?.includes(slotName)),
    replace(next) {
      active.value = next
    },
  }
}

const previewManifest = {
  plugin_id: 'seed.preview',
  plugin_digest: 'digest-preview-1',
  plugin_version: '1.0.0',
  slots: ['route'],
  metadata: { label: '只读预览', description: '由 Seed snapshot 投影' },
}

describe('ClientExtensionSlot', () => {
  it('mounts a declarative snapshot entry into the requested slot', () => {
    const clientExtensions = extensionContext([previewManifest], [
      { plugin_id: 'seed.preview', state: 'active' },
    ])
    const wrapper = mount(ClientExtensionSlot, {
      props: { slotName: 'route' },
      global: { provide: { clientExtensions } },
    })

    expect(wrapper.find('[data-extension-slot="route"]').exists()).toBe(true)
    expect(wrapper.find('[data-plugin-id="seed.preview"]').attributes('data-extension-state')).toBe('active')
    expect(wrapper.text()).toContain('只读预览')
    expect(wrapper.text()).toContain('已挂载')
  })

  it('unmounts removed entries and remounts a new content-addressed version', async () => {
    const clientExtensions = extensionContext([previewManifest])
    const wrapper = mount(ClientExtensionSlot, {
      props: { slotName: 'route' },
      global: { provide: { clientExtensions } },
    })

    clientExtensions.replace([])
    await nextTick()
    expect(wrapper.find('.client-extension-slot').exists()).toBe(false)

    clientExtensions.replace([{
      ...previewManifest,
      plugin_digest: 'digest-preview-2',
      plugin_version: '2.0.0',
      metadata: { label: '只读预览 2' },
    }])
    await nextTick()
    expect(wrapper.find('[data-plugin-id="seed.preview"]').exists()).toBe(true)
    expect(wrapper.text()).toContain('v2.0.0')
  })

  it('surfaces failure state without executing a plugin renderer', () => {
    const clientExtensions = extensionContext([previewManifest], [
      { plugin_id: 'seed.preview', state: 'quarantined' },
    ])
    const wrapper = mount(ClientExtensionSlot, {
      props: { slotName: 'route' },
      global: { provide: { clientExtensions } },
    })

    expect(wrapper.find('[data-extension-state="quarantined"]').exists()).toBe(true)
    expect(wrapper.text()).toContain('已隔离')
    expect(wrapper.find('[data-executable-source]').exists()).toBe(false)
  })
})
