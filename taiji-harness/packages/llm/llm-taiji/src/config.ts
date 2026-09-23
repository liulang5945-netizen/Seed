/** Plugin configuration and the one explicit resolution step for the Taiji runtime route. */
import type { Volatile } from '@taiji/cordis'

import z from '@taiji/schemastery'
import { isVolatile } from '@taiji/cosmokit'
import { resolveRetryPolicy, RetryPolicySchema } from '@taiji/dsh-llm'
import type { RetryPolicyConfig } from '@taiji/dsh-llm'
import { DEFAULT_BASE_URL, DEFAULT_MODELS } from './defaults.ts'
import type { TaijiCatalogModel, TaijiConnectionOptions } from './types.ts'

/**
 * Plugin config, validated by the same-named schemastery schema and doubling
 * as the `llm-taiji` settings-section shape. Every field is optional in yml:
 * omitting `baseURL` uses the runtime's local address, and omitting `models`
 * advertises the runtime's single language organ.
 */
export interface Config {
  /** Endpoint root of the Taiji runtime; defaults to the runtime's local address. */
  baseURL: Volatile<string | undefined>
  /** Advisory models shown by discovery consumers; defaults to the runtime's single entry. */
  models: Volatile<TaijiCatalogModel[]>
  /** Provider-owned model-request retry policy; omission uses normal mode with five retries. */
  retryPolicy: Volatile<RetryPolicyConfig | undefined>
}

/** Plain options accepted by the provider resolver. */
export type Options = { [K in keyof Config]?: Config[K] extends Volatile<infer T> ? Exclude<T, undefined> : never }

/**
 * Read the current value behind every reference of a validated Config.
 * @param config Parsed plugin Config.
 * @returns Plain options for the resolver.
 */
export function plainOptions(config: Config): Options {
  return Object.fromEntries(Object.entries(config).map(([key, value]) => [key, isVolatile(value) ? value.get() : value]))
}

const catalogModel: z<TaijiCatalogModel> = z.object({
  id: z.string().required(),
  name: z.string(),
  description: z.string(),
  contextWindow: z.number().step(1).min(1),
})

export const Config = z.object({
  baseURL: z.string().volatile(),
  models: z.array(catalogModel).default(DEFAULT_MODELS).volatile(),
  retryPolicy: RetryPolicySchema.volatile(),
})

/** One resolution's complete request facts for this route. */
export type ResolvedTaijiOptions = TaijiConnectionOptions

/** Resolve, validate, and detach the advisory model catalog. */
function resolveModels(models: readonly TaijiCatalogModel[] | undefined): TaijiCatalogModel[] {
  const seen = new Set<string>()
  return (models ?? DEFAULT_MODELS).map((model) => {
    if (model.id.length === 0) throw new Error('llm-taiji: catalog model ids must be non-empty')
    if (model.name !== undefined && model.name.length === 0) {
      throw new Error(`llm-taiji: catalog model "${model.id}" has an empty name`)
    }
    if (model.contextWindow !== undefined
      && (!Number.isInteger(model.contextWindow) || model.contextWindow <= 0)) {
      throw new Error(`llm-taiji: catalog model "${model.id}" contextWindow must be a positive integer`)
    }
    if (seen.has(model.id)) throw new Error(`llm-taiji: duplicate catalog model "${model.id}"`)
    seen.add(model.id)
    return {
      id: model.id,
      ...model.name === undefined ? {} : { name: model.name },
      ...model.description === undefined ? {} : { description: model.description },
      ...model.contextWindow === undefined ? {} : { contextWindow: model.contextWindow },
    }
  })
}

/**
 * The one explicit resolve step from raw config to validated connection facts.
 * Programmatic construction may bypass Schemastery normalization, so every
 * bound is re-judged here — for the composition entry at load (fail loud) and
 * for each settings snapshot at its first use.
 * @param config - raw plugin config or resolved settings snapshot.
 * @returns validated connection facts for one operation.
 */
export function resolveAdapterOptions(config: Options): ResolvedTaijiOptions {
  const baseURL = config.baseURL ?? DEFAULT_BASE_URL
  const parsed = new URL(baseURL)
  if (!['http:', 'https:'].includes(parsed.protocol) || parsed.username || parsed.password || parsed.search || parsed.hash) {
    throw new Error('llm-taiji: baseURL must be an HTTP(S) root without credentials, query, or fragment')
  }
  return {
    baseURL,
    models: resolveModels(config.models),
    retryPolicy: resolveRetryPolicy(config.retryPolicy, 'llm-taiji: retryPolicy'),
  }
}
