/**
 * Shared mechanics for a provider's model catalog: the duplicate-id guard and
 * the base field projection every provider catalog repeats.
 *
 * Providers own their extra fields (image budgets, token caps, modalities) and
 * their own validation copy; only the two steps that were byte-identical
 * across them live here, so a catalog rule cannot be fixed in one provider and
 * missed in the other.
 *
 * @module @taiji/dsh-llm/catalog-model
 */

/** The fields every provider catalog model shares. */
export interface CatalogModelBase {
  readonly id: string
  readonly name?: string
  readonly description?: string
  readonly contextWindow?: number
}

/**
 * Reject a model id already claimed earlier in the same catalog and record it.
 * @param pkg - the provider package name used in the diagnostic.
 * @param model - the model about to be admitted.
 * @param seen - ids already admitted by the caller's resolution pass.
 */
export function requireNewCatalogId(pkg: string, model: CatalogModelBase, seen: Set<string>): void {
  if (seen.has(model.id)) throw new Error(`${pkg}: duplicate catalog model "${model.id}"`)
  seen.add(model.id)
}

/**
 * Project the shared fields, omitting the optional ones the model leaves unset
 * so an absent value never becomes an explicit `undefined` in the resolved
 * options a consumer compares or persists.
 * @param model - the raw catalog entry.
 * @returns the base fields carried by the resolved model.
 */
export function catalogModelBaseFields(model: CatalogModelBase): {
  id: string
  name?: string
  description?: string
  contextWindow?: number
} {
  return {
    id: model.id,
    ...model.name === undefined ? {} : { name: model.name },
    ...model.description === undefined ? {} : { description: model.description },
    ...model.contextWindow === undefined ? {} : { contextWindow: model.contextWindow },
  }
}
