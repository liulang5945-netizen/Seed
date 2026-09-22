import { clientBundle } from '../../client/tsdown.client.ts'

export default clientBundle(
  '@taiji/dsh-api-workspace-files',
  ['lib/types/index.js'],
  { hostPhase: true },
)
