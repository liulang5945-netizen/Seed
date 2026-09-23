/**
 * Composer blocks: the one way another plugin stops a session's input.
 *
 * The composer cannot read the plugins that would know — the dependency runs
 * ui-model-selection → ui-conversation, never back — so each blocker pushes
 * under its own owner key and the bar reads its own session's store. A block
 * carries the localized reason it exists, because the plugin that raised it
 * owns that copy; the composer only knows how to render an inert textarea with
 * a placeholder, exactly as it already does for a session with no workspace.
 * Owners accumulate rather than overwrite: an archived session whose model
 * route is also unconfigured keeps both blocks, and the store publishes the
 * first present owner in {@link COMPOSER_BLOCK_OWNERS} order.
 *
 * This is an affordance, not enforcement: the Host refuses a prompt it cannot
 * route regardless of what any client disables.
 */

import { createSnapshotStore, type SnapshotStore } from '@taiji/dsh-client-store'
import type { SessionId } from '@taiji/dsh-session/types'
import { COMPOSER_BLOCK_OWNERS, type ComposerBlock, type ComposerBlockOwner, type ComposerBlocks } from '../contract/composer-blocks.ts'

/** The per-session composer-block registry (one instance per plugin fiber). */
export class ComposerBlockRegistry implements ComposerBlocks {
  private readonly stores = new Map<SessionId, SnapshotStore<ComposerBlock | undefined>>()
  private readonly raised = new Map<SessionId, Map<ComposerBlockOwner, ComposerBlock>>()

  /** @inheritdoc */
  set(sessionId: SessionId, owner: ComposerBlockOwner, block: ComposerBlock | undefined): void {
    const owners = this.raised.get(sessionId)
    if (block === undefined) {
      if (owners === undefined || !owners.delete(owner)) return
      if (owners.size === 0) this.raised.delete(sessionId)
    } else {
      if (owners?.get(owner)?.reason === block.reason) return
      if (owners === undefined) this.raised.set(sessionId, new Map([[owner, block]]))
      else owners.set(owner, block)
    }
    const next = this.resolved(sessionId)
    const store = this.storeFor(sessionId)
    if (store.getSnapshot()?.reason !== next?.reason) store.set(next)
  }

  /** @inheritdoc */
  storeFor(sessionId: SessionId): SnapshotStore<ComposerBlock | undefined> {
    const existing = this.stores.get(sessionId)
    if (existing !== undefined) return existing
    const created = createSnapshotStore<ComposerBlock | undefined>(undefined)
    this.stores.set(sessionId, created)
    return created
  }

  /** @inheritdoc */
  forget(sessionId: SessionId): void {
    this.stores.delete(sessionId)
    this.raised.delete(sessionId)
  }

  /** The first present owner's block, or undefined while no owner holds one. */
  private resolved(sessionId: SessionId): ComposerBlock | undefined {
    const owners = this.raised.get(sessionId)
    if (owners === undefined) return undefined
    for (const owner of COMPOSER_BLOCK_OWNERS) {
      const block = owners.get(owner)
      if (block !== undefined) return block
    }
    return undefined
  }
}
