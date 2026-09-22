/** Desktop Office skills and bundled authoring dependencies. */

import { dirname, join } from 'node:path'
import type { Context } from '@taiji/cordis'
import * as officeSkills from '@taiji/dsh-skill-office'
import * as workspaceDependencies from '@taiji/dsh-tool-workspace-dependencies'

/** Loader identity for the application-owned Office composition. */
export const name = 'desktop-office'
/** Application-selected bundled payload and installation directories. */
export interface Config {
  /** Bundled payload directory. Missing sibling `office-skills` resources fail Host startup. */
  readonly source: string
  /** Harness-home directory where workspace dependencies are installed. */
  readonly root: string
}

/**
 * Enable offline Office authoring and structural checks in the Desktop profile.
 * @param ctx - Profile scope; child plugins declare their own service requirements.
 * @param config - Bundled payload source and Harness-home installation root.
 */
export async function apply(ctx: Context, config: Config): Promise<void> {
  await ctx.plugin(workspaceDependencies, config)
  await ctx.plugin(officeSkills, { assetRoot: join(dirname(config.source), 'office-skills') })
}
