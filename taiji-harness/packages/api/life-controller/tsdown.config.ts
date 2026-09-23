import { clientBundle } from '../../client/tsdown.client.ts'

export default clientBundle(
  '@taiji/dsh-api-life-controller',
  ['lib/types/index.js'],
  { hostPhase: true },
)