import type { SnapshotStore } from '@taiji/dsh-client-store'
import type { SessionId } from '@taiji/dsh-session/types'

/** Why one session's composer is inert. */
export interface ComposerBlock {
  /** Localized placeholder owned by the plugin that raised the block. */
  readonly reason: string
}

/**
 * The plugins that may keep a composer inert, in precedence order. Several can
 * hold a block on one Session at once — an archived Session whose model route
 * is also unconfigured — and only one placeholder can render, so the first
 * present owner supplies it: `workspace-archive` states the Session is
 * read-only, which outranks the `model-selection` request to choose a model
 * that no prompt could be routed through yet.
 */
export const COMPOSER_BLOCK_OWNERS = ['workspace-archive', 'model-selection'] as const

/** One plugin that raises a composer block. */
export type ComposerBlockOwner = typeof COMPOSER_BLOCK_OWNERS[number]

/** The registry face other plugins reach through `ctx.conversation.blocks`. */
export interface ComposerBlocks {
  /**
   * Raise or clear one owner's block for this session. Other owners' blocks
   * stay raised; the store publishes the first present owner in
   * {@link COMPOSER_BLOCK_OWNERS} order.
   * @param sessionId - Session whose composer is affected.
   * @param owner - the plugin raising or clearing its own block.
   * @param block - Block to raise, or undefined to clear this owner's.
   */
  set(sessionId: SessionId, owner: ComposerBlockOwner, block: ComposerBlock | undefined): void
  /**
   * Resolve the observable block state for one Session.
   * @param sessionId - Session to observe.
   * @returns Identity-stable block store.
   */
  storeFor(sessionId: SessionId): SnapshotStore<ComposerBlock | undefined>
  /**
   * Drop one Session's store.
   * @param sessionId - Session being released.
   */
  forget(sessionId: SessionId): void
}