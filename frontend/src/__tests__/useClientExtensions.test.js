import { nextTick } from 'vue'
import { useClientExtensions } from '../composables/useClientExtensions.js'
import { nativeApi } from '../composables/nativeApi.js'

vi.mock('../composables/nativeApi.js', () => ({
  nativeApi: {
    clientExtensions: vi.fn(),
    clientExtensionsPrepare: vi.fn(),
    clientExtensionsCommit: vi.fn(),
    clientExtensionsDependency: vi.fn(),
    clientExtensionsRollback: vi.fn(),
    clientExtensionsBeginCall: vi.fn(),
    clientExtensionsEndCall: vi.fn(),
    clientExtensionsRetire: vi.fn(),
    clientExtensionsQuarantine: vi.fn(),
  },
}))

const payload = {
  snapshot: {
    snapshot_id: 'client-snapshot-1',
    manifests: [
      { plugin_id: 'seed.preview', slots: ['ide.panel', 'route'] },
      { plugin_id: 'seed.metrics', slots: ['visualization'] },
    ],
  },
  policy: { policy_digest: 'policy-1' },
  lifecycle: [{ plugin_id: 'seed.preview', state: 'active' }],
  dependency_health: { workbench: true },
}

beforeEach(() => {
  vi.clearAllMocks()
  nativeApi.clientExtensions.mockResolvedValue(payload)
})

it('refreshes the native snapshot and exposes declarative stable slots', async () => {
  const extensions = useClientExtensions()
  await extensions.refresh()
  await nextTick()

  expect(extensions.snapshot.value.snapshot_id).toBe('client-snapshot-1')
  expect(extensions.activeManifests.value).toHaveLength(2)
  expect(extensions.slotManifests('ide.panel')).toHaveLength(1)
  expect(extensions.slotManifests('route')[0].plugin_id).toBe('seed.preview')
  expect(extensions.slotManifests('desktop.root_shell')).toEqual([])
})

it('keeps prepare and commit on the native two-phase boundary', async () => {
  const extensions = useClientExtensions()
  nativeApi.clientExtensionsPrepare.mockResolvedValue({
    status: 'prepared',
    prepared_id: 'prepared-1',
  })
  nativeApi.clientExtensionsCommit.mockResolvedValue({
    status: 'committed',
    snapshot: payload.snapshot,
  })

  await expect(extensions.prepare([{ plugin_id: 'seed.preview' }], {
    capabilitySnapshotId: 'capability-1',
    states: { 'seed.preview': { open_count: 1 } },
  })).resolves.toEqual({ status: 'prepared', prepared_id: 'prepared-1' })
  expect(nativeApi.clientExtensionsPrepare).toHaveBeenCalledWith({
    capability_snapshot_id: 'capability-1',
    manifests: [{ plugin_id: 'seed.preview' }],
    dependency_health: undefined,
    states: { 'seed.preview': { open_count: 1 } },
  })

  await extensions.commit('prepared-1')
  expect(nativeApi.clientExtensionsCommit).toHaveBeenCalledWith({ prepared_id: 'prepared-1' })
  expect(nativeApi.clientExtensions).toHaveBeenCalledTimes(1)
})
