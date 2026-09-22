import { clientBundle } from '../../client/tsdown.client.ts'

export default clientBundle(
  '@taiji/dsh-api-terminal-controller',
  ['lib/types/index.js'],
  { hostPhase: true },
)
