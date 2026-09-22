import { clientBundle } from '../../client/tsdown.client.ts'

export default clientBundle(
  '@taiji/dsh-api-job-controller',
  ['lib/types/index.js'],
  { hostPhase: true },
)
