import { clientBundle } from '../../client/tsdown.client.ts'

export default clientBundle(
  '@taiji/dsh-api-remotes',
  ['lib/types/index.js'],
  { hostPhase: true },
)
