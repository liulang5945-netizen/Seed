/**
 * Renderer-only tests for the memory recall block: no HTTP, no Cordis, just
 * the exact model-visible artifact and its budget behaviour.
 */
import { describe, expect, it } from 'vitest'
import { RECALL_NOTE, TRUNCATED_MARKER, renderRecall } from '../src/reading.ts'
import type { RecallEntry } from '../src/reading.ts'

const QUERY = '发布检查点'

const ENTRIES: readonly RecallEntry[] = [
  { kind: 'interaction', text: '问：如何发布检查点\n答：我无法发布。', score: 0.62 },
  { kind: 'note', text: '上次发布的产物在 checkpoints/ 下', score: 0.38 },
]

const HEADER_TWO = ['memory entries=2 query="发布检查点"', RECALL_NOTE]
const ENTRY_ONE = '- [0.62 interaction] 问：如何发布检查点 / 答：我无法发布。'
const ENTRY_TWO = '- [0.38 note] 上次发布的产物在 checkpoints/ 下'
const FULL = [...HEADER_TWO, ENTRY_ONE, ENTRY_TWO].join('\n')
const CLIPPED = [...['memory entries=1 query="发布检查点"', RECALL_NOTE], ENTRY_ONE, TRUNCATED_MARKER].join('\n')
const HEADER_ONLY = ['memory entries=0 query="发布检查点"', RECALL_NOTE].join('\n')

describe('renderRecall', () => {
  it('renders the header, the note, and one line per entry in the runtime order', () => {
    expect(renderRecall(ENTRIES, QUERY, 600)).toBe(FULL)
  })

  it('collapses an entry text onto its one line and formats every score with two decimals', () => {
    const entries: readonly RecallEntry[] = [
      { kind: 'note', text: 'first\nsecond\r\nthird', score: 0.887 },
      { kind: 'note', text: 'half', score: 0.5 },
      { kind: 'note', text: 'whole', score: 1 },
    ]
    const block = renderRecall(entries, 'q', 600)
    expect(block).toBe([
      'memory entries=3 query="q"',
      RECALL_NOTE,
      '- [0.89 note] first / second / third',
      '- [0.50 note] half',
      '- [1.00 note] whole',
    ].join('\n'))
  })

  it('injects nothing when recall found no entry', () => {
    expect(renderRecall([], QUERY, 600)).toBeUndefined()
  })

  it('drops whole entry lines and marks the block truncated', () => {
    const block = renderRecall(ENTRIES, QUERY, FULL.length - 1)
    expect(block).toBe(CLIPPED)
    expect(block?.endsWith(TRUNCATED_MARKER)).toBe(true)
    expect(block?.length).toBeLessThanOrEqual(FULL.length - 1)
  })

  it('squeezes the best entry in, clipped, when the budget is tight', () => {
    // Enough room for one entry line only if it is cut: the block keeps the
    // best-ranked entry rather than dropping everything it recalled.
    const block = renderRecall(ENTRIES, QUERY, CLIPPED.length - 1)

    expect(block?.length).toBeLessThanOrEqual(CLIPPED.length - 1)
    expect(block?.split('\n').filter(line => line.startsWith('- ['))).toHaveLength(1)
    expect(block).toContain('0.62')
    expect(block).toContain('…')
    expect(block?.endsWith(TRUNCATED_MARKER)).toBe(true)
  })

  it('keeps the header and the note when the budget carries no entry line', () => {
    const block = renderRecall(ENTRIES, QUERY, HEADER_ONLY.length - 1)
    expect(block).toBe(HEADER_ONLY)
    expect(block).not.toContain(TRUNCATED_MARKER)
  })

  it('clips the query echoed in the header to forty characters', () => {
    const long = 'a'.repeat(60)
    const block = renderRecall(ENTRIES, long, 600)
    expect(block?.startsWith(`memory entries=2 query="${'a'.repeat(40)}"`)).toBe(true)
    expect(block).not.toContain('a'.repeat(41))
  })

  it('never reorders or invents an entry', () => {
    const reversed: readonly RecallEntry[] = [...ENTRIES].reverse()
    const block = renderRecall(reversed, QUERY, 600)
    expect(block?.indexOf('0.38')).toBeLessThan(block?.indexOf('0.62') ?? Number.POSITIVE_INFINITY)
    expect(block?.split('\n').filter(line => line.startsWith('- ['))).toHaveLength(2)
  })

  it('clips a long entry instead of dropping it, and marks the block truncated', () => {
    // A journal interaction routinely carries a prompt and an answer, so one
    // recalled entry can be longer than the whole budget by itself.
    const long: readonly RecallEntry[] = [
      { kind: 'interaction', text: `问：如何发布检查点\n答：${'x'.repeat(3000)}`, score: 0.9 },
    ]

    const block = renderRecall(long, QUERY, 600)

    expect(block?.length).toBeLessThanOrEqual(600)
    expect(block).toContain('memory entries=1')
    expect(block?.split('\n').filter(line => line.startsWith('- ['))).toHaveLength(1)
    expect(block).toContain('- [0.90 interaction] 问：如何发布检查点 / 答：')
    expect(block).toContain('…')
    expect(block?.endsWith(TRUNCATED_MARKER)).toBe(true)
  })

  it('keeps the best entry when every entry is longer than its own room', () => {
    const long: readonly RecallEntry[] = [
      { kind: 'interaction', text: 'top '.repeat(400), score: 0.9 },
      { kind: 'note', text: 'second '.repeat(400), score: 0.4 },
    ]

    const block = renderRecall(long, QUERY, 600)

    expect(block?.length).toBeLessThanOrEqual(600)
    expect(block).toContain('entries=1')
    expect(block).toContain('0.90')
    expect(block).not.toContain('0.40')
    expect(block?.endsWith(TRUNCATED_MARKER)).toBe(true)
  })
})
