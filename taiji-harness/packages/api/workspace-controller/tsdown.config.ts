import { clientBundle } from '../../client/tsdown.client.ts'

export default clientBundle(
  '@taiji/dsh-api-workspace-controller',
  ['lib/types/index.js'],
  { hostPhase: true },
)
