/** Model catalog, connection facts, and adapter dependencies for the Taiji route. */
import type { ResolvedRetryPolicy } from '@taiji/dsh-llm'

/**
 * One advisory model entry for the Taiji runtime. The runtime serves exactly
 * one language organ and its chat endpoint carries no model id, so an entry is
 * a selector label the harness routes by — never a wire value.
 */
export interface TaijiCatalogModel {
  /** Selector id accepted by `GenerateOptions.model`; never sent to the runtime. */
  id: string
  /** Selector label; defaults to {@link id}. */
  name?: string
  /** Optional selector detail for a deployment running several runtimes. */
  description?: string
  /** Known combined request/response capacity; omitted when the deployment knows none. */
  contextWindow?: number
}

/**
 * Validated connection facts for one operation. The plugin's
 * `resolveAdapterOptions` is the one explicit resolve step producing this
 * shape; the adapter re-reads it per operation, so a configuration change
 * reaches the next request without re-registration.
 */
export interface TaijiConnectionOptions {
  /** Taiji runtime HTTP root; the chat and health paths are appended to it. */
  baseURL: string
  /** Advisory models exposed to discovery consumers; requests remain unrestricted. */
  models: readonly TaijiCatalogModel[]
  /** Provider-owned model-request retry policy, already resolved. */
  retryPolicy: ResolvedRetryPolicy
}

/** Constructor options for {@link TaijiAdapter}: the operation-local resolution hook the plugin owns. */
export interface TaijiAdapterOptions {
  /** Current validated connection facts; called once per operation. */
  options: () => TaijiConnectionOptions
}
