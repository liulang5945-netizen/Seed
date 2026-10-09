/** D2 P1-③: the checkpoint name the shipped backend payload has to carry. */
import { existsSync, readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import { defaultCheckpointName } from '../scripts/backend-code-payload.ts'

const RUNTIME_SOURCE = fileURLToPath(new URL('../../../../api/seed_runtime.py', import.meta.url))
const CHECKPOINT_ROOT = fileURLToPath(new URL('../../../../checkpoints', import.meta.url))

describe('defaultCheckpointName', () => {
  it('reads the product default out of the source the backend ships', () => {
    const name = defaultCheckpointName(readFileSync(RUNTIME_SOURCE, 'utf8'))
    expect(name).toMatch(/\.pt$/u)
    expect(existsSync(`${CHECKPOINT_ROOT}/${name}`)).toBe(true)
  })

  it('reads a single-line assignment', () => {
    expect(defaultCheckpointName('DEFAULT_CHECKPOINT = Path("x") / "checkpoints" / "base.pt"\n')).toBe('base.pt')
  })

  it('refuses a source with no default', () => {
    expect(() => defaultCheckpointName('FACTORY_CHECKPOINT = Path("checkpoints") / "base.pt"\n'))
      .toThrow('no DEFAULT_CHECKPOINT assignment')
  })

  it('refuses a default that names more than one file', () => {
    expect(() => defaultCheckpointName('DEFAULT_CHECKPOINT = pick("a.pt", "b.pt")\n'))
      .toThrow('must name one .pt file, found 2')
  })
})
