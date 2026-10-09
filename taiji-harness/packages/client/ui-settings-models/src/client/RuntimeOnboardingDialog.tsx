/**
 * Local-runtime first-run step: the shipped composition's only model route is the
 * Taiji runtime, so when that route is declared but nobody answers it, the step
 * says what to start rather than asking for a credential.
 */

import { useEffect } from 'react'
import type { ReactNode } from 'react'
import { Button } from '@taiji/dsh-client-ui-primitives'
import type { SnapshotStore } from '@taiji/dsh-client-store'
import type { InjectFace, PropsRuntime } from '@taiji/dsh-client-ui-slots'
import type { ModelsSettingsState, ModelsSettingsStore, OnboardingTarget } from './store.ts'
import { onboardingReadiness } from './store.ts'
import type {} from './slot-contract.ts'
import type { en } from './locales.ts'
import { OnboardingModal } from './OnboardingModal.tsx'
import styles from './RuntimeOnboardingDialog.module.css'

/** The route this step speaks for: the fork's own local runtime. */
const RUNTIME_TARGET: OnboardingTarget = { provider: 'taiji-local', settingsNs: 'llm-taiji' }

/** Registration-side dependencies of {@link RuntimeOnboardingDialog}. */
export interface RuntimeOnboardingInjected {
  /** Whether the step may show on its own; an explicit request always opens it. */
  automatic: boolean
  hooks: {
    /** Shared Models-page join state, bound by the slot renderer. */
    models: SnapshotStore<ModelsSettingsState>
  }
  /** Shared Models-page join controller, loaded once to read the join. */
  controller: ModelsSettingsStore
  /** Feature copy. */
  t: (key: keyof typeof en) => string
}

/** Slot owner props plus the feature's injected dependencies. */
export type RuntimeOnboardingDialogProps =
  PropsRuntime<'settings.onboarding'> & InjectFace<RuntimeOnboardingInjected>

/**
 * Tell a first-run user that the local runtime is not answering, while its route is
 * declared and no provider can serve a request. A route that needs a credential is
 * left to the credential step; a route that needs none can only be started.
 * @param props - settings-shell owner state and Models feature dependencies.
 * @returns the onboarding modal, or null when the runtime is not the blocker.
 */
export function RuntimeOnboardingDialog(props: RuntimeOnboardingDialogProps): ReactNode {
  const { complete, controller, useModels, t, automatic, explicit = false } = props
  const state = useModels(snapshot => snapshot)
  const readiness = onboardingReadiness(state, RUNTIME_TARGET)

  useEffect(() => {
    if ((automatic || explicit) && state.status === 'idle') void controller.load()
  }, [controller, state.status, automatic, explicit])

  useEffect(() => {
    if (
      (!automatic && !explicit)
      || (readiness.kind !== 'runtime-unreachable' && readiness.kind !== 'loading')
    ) complete()
  }, [complete, readiness.kind, explicit, automatic])

  if (!automatic && !explicit) return null
  if (readiness.kind !== 'runtime-unreachable') return null

  return (
    <OnboardingModal title={t('runtimeOnboardingTitle')} focusTitle>
      <p className={styles.description}>{t('runtimeOnboardingDescription')}</p>
      <div className={styles.actions}>
        <Button variant="outline" onClick={() => { complete() }}>{t('runtimeOnboardingDismiss')}</Button>
      </div>
    </OnboardingModal>
  )
}
