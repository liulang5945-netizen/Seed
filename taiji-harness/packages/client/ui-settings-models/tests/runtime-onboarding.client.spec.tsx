// @vitest-environment jsdom
/** First-run local-runtime step over the shared Models join. */
import type { GlobalStandardProps } from '@taiji/dsh-client-ui-slots'
import type { SettingsNamespaceView } from '@taiji/dsh-api-remotes/client'
import type { JsonValue } from '@taiji/dsh-util-values'
import { bindSnapshotSelector } from '@taiji/dsh-client-test-runtime'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import Schema from '@taiji/schemastery'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { SettingsDescribeMirror } from '@taiji/dsh-client-ui-settings/src/client/settings-mirror.ts'
import { en } from '../src/client/locales.ts'
import { ModelsSettingsStore } from '../src/client/store.ts'
import { RuntimeOnboardingDialog } from '../src/client/RuntimeOnboardingDialog.tsx'
import type { RuntimeOnboardingDialogProps } from '../src/client/RuntimeOnboardingDialog.tsx'
import { settingsSchema } from './settings-schema.client.ts'

const useResource = (() => ({ status: 'none' as const, value: undefined, failure: undefined, reload: () => {} })) as GlobalStandardProps['useResource']
const usePanelInfo: GlobalStandardProps['usePanelInfo'] = selector => selector({ activePanelId: null })
const unusedHook = (() => { throw new Error('unused standard hook') }) as never
type AttentionSnapshot = Parameters<Parameters<RuntimeOnboardingDialogProps['useSessionStatus']>[0]>[0]
const useSessionStatus: RuntimeOnboardingDialogProps['useSessionStatus'] = selector => selector(new Map() as AttentionSnapshot)

const TaijiConfig = Schema.object({
  baseURL: Schema.string().pattern(/^https?:\/\//),
  models: Schema.array(Schema.object({ id: Schema.string().required(), name: Schema.string() })),
})

/** The Taiji settings document: a route with no credential reference at all. */
function taijiNamespace(): SettingsNamespaceView {
  return {
    ns: 'llm-taiji',
    schema: JSON.parse(JSON.stringify(TaijiConfig.toJSON())) as JsonValue,
    value: {},
    base: {},
    user: {},
    autoGenerate: true, applies: 'live',
    secrets: [],
    revision: 0,
  }
}

function remoteOk<T>(value: T) {
  return { ok: true as const, value }
}

/** Boot the shared join over a Taiji route whose adapter is present only when `active`. */
function bench(active: boolean) {
  const face = {
    llm: {
      listProviders: () => Promise.resolve(remoteOk(
        active ? [{ id: 'taiji-local', name: 'Taiji' }] : [],
      )),
      listConfigurableProviders: () => Promise.resolve(remoteOk([{
        provider: 'taiji-local',
        displayName: 'Taiji（本地运行时）',
        settingsNs: 'llm-taiji',
        settingsPath: [],
      }])),
      discoverModels: () => Promise.resolve(remoteOk([])),
    },
    settings: {
      describe: () => Promise.resolve(remoteOk({
        writable: true, hasDocument: false, namespaces: [taijiNamespace()],
      })),
      mutate: () => Promise.resolve(remoteOk(false)),
    },
    credentials: {
      describe: () => Promise.resolve(remoteOk({})),
      set: () => Promise.resolve(remoteOk(undefined)),
    },
  }
  const ctx = { remote: face } as never
  const controller = new ModelsSettingsStore(ctx, settingsSchema, new SettingsDescribeMirror(ctx))
  const complete = vi.fn()
  const props: RuntimeOnboardingDialogProps = {
    automatic: true,
    explicit: false,
    stepId: 'taiji-local',
    complete,
    openSection: vi.fn(),
    useSessions: unusedHook,
    useSessionStatus,
    usePanelInfo, useSessionRetainInfo: () => undefined, useResource,
    useWorkspaces: unusedHook,
    controller,
    useModels: bindSnapshotSelector(controller.store),
    t: key => en[key],
  }
  return { controller, complete, props }
}

afterEach(() => {
  cleanup()
  document.getElementById('root')?.remove()
})

describe('RuntimeOnboardingDialog', () => {
  it('names the runtime as the blocker and ends the step on dismissal', async () => {
    const { controller, complete, props } = bench(false)
    await controller.load()
    render(<div id="root"><RuntimeOnboardingDialog {...props} /></div>)
    const dialog = await screen.findByRole('dialog', { name: en.runtimeOnboardingTitle })
    expect(dialog.textContent).toContain(en.runtimeOnboardingDescription)
    fireEvent.click(screen.getByRole('button', { name: en.runtimeOnboardingDismiss }))
    expect(complete).toHaveBeenCalled()
  })

  it('stays silent while some provider can already serve requests', async () => {
    const { controller, complete, props } = bench(true)
    await controller.load()
    render(<div id="root"><RuntimeOnboardingDialog {...props} /></div>)
    await waitFor(() => { expect(complete).toHaveBeenCalled() })
    expect(screen.queryByRole('dialog', { name: en.runtimeOnboardingTitle })).toBeNull()
  })

  it('completes without rendering when the shell never asked for the step', async () => {
    const { controller, complete, props } = bench(false)
    await controller.load()
    render(<div id="root"><RuntimeOnboardingDialog {...props} automatic={false} /></div>)
    await waitFor(() => { expect(complete).toHaveBeenCalled() })
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('opens on an explicit request even where the shell does not offer it automatically', async () => {
    const { controller, complete, props } = bench(false)
    await controller.load()
    render(<div id="root"><RuntimeOnboardingDialog {...props} automatic={false} explicit /></div>)
    await screen.findByRole('dialog', { name: en.runtimeOnboardingTitle })
    expect(complete).not.toHaveBeenCalled()
  })

  it('waits for the first join instead of declaring the runtime unreachable', async () => {
    const { complete, props } = bench(false)
    act(() => {
      render(<div id="root"><RuntimeOnboardingDialog {...props} /></div>)
    })
    await Promise.resolve()
    expect(complete).not.toHaveBeenCalled()
    expect(screen.queryByRole('dialog')).toBeNull()
  })
})
