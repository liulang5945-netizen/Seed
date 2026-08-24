import { describe, it, expect } from 'vitest'

/**
 * parseMessageContent tests
 *
 * Tests the message parsing logic that separates thinking from final answer.
 */

// Simplified parseMessageContent for testing core logic
function parseMessageContent(text) {
  if (!text) return { reasoning: '', content: '' }

  let reasoning = ''
  let content = text

  // 1. think tags
  const thinkOpenMatch = text.match(/<think>/i)
  if (thinkOpenMatch) {
    const openIdx = thinkOpenMatch.index
    const afterOpen = text.slice(openIdx + 7) // '<think>'.length = 7
    const thinkCloseMatch = afterOpen.match(/<\/think>/i)
    if (thinkCloseMatch) {
      reasoning = afterOpen.slice(0, thinkCloseMatch.index).trim()
      const beforeThink = text.slice(0, openIdx)
      const afterThink = afterOpen.slice(thinkCloseMatch.index + 8) // '</think>'.length = 8
      content = (beforeThink + afterThink).trim()
      if (reasoning) return { reasoning, content }
    } else {
      reasoning = afterOpen.trim()
      content = text.slice(0, openIdx).trim()
      return { reasoning, content: content || '' }
    }
  }

  // 2. English patterns using RegExp constructor to avoid multiline regex literal issues
  const NL = '\\n'
  const engPatterns = [
    new RegExp('^\\s*[Tt]hought:\\s*([\\s\\S]*?)(?=' + NL + '\\s*(?:[Aa]nswer|[Ff]inal):|' + NL + '---|' + NL + NL + NL + '|$)', 'm'),
    new RegExp('^\\s*[Rr]easoning:\\s*([\\s\\S]*?)(?=' + NL + '\\s*(?:[Aa]nswer|[Ff]inal):|' + NL + '---|' + NL + NL + NL + '|$)', 'm'),
  ]
  for (const pattern of engPatterns) {
    const match = text.match(pattern)
    if (match && match[1] && match[1].trim().length > 5) {
      reasoning = match[1].trim()
      const after = text.slice(match.index + match[0].length)
      content = after.replace(/^\n?(?:Answer|Final):\s*/, '').replace(/^---\s*\n?/, '').trim()
      if (reasoning) return { reasoning, content }
    }
  }

  // 3. Reasoning: ... Answer: (simple single-line)
  const engMatch = text.match(new RegExp('^Reasoning:\\s*([\\s\\S]*?)\\nAnswer:\\s*([\\s\\S]*)$', 'im'))
  if (engMatch) {
    return { reasoning: engMatch[1].trim(), content: engMatch[2].trim() }
  }

  // 4. --- separator
  const sepMatch = text.match(new RegExp('^([\\s\\S]*?)\\n---\\n([\\s\\S]*)$'))
  if (sepMatch) {
    const first = sepMatch[1].trim()
    const second = sepMatch[2].trim()
    if (first.length < second.length * 1.5 && first.length > 20) {
      return { reasoning: first, content: second }
    }
  }

  return { reasoning: '', content: text }
}

describe('parseMessageContent', () => {
  describe('empty input', () => {
    it('returns empty for empty string', () => {
      expect(parseMessageContent('')).toEqual({ reasoning: '', content: '' })
    })

    it('returns empty for null/undefined', () => {
      expect(parseMessageContent(null)).toEqual({ reasoning: '', content: '' })
      expect(parseMessageContent(undefined)).toEqual({ reasoning: '', content: '' })
    })
  })

  describe('think tags', () => {
    it('parses closed think tag', () => {
      const text = '<think>analysis steps</think>\nfinal answer'
      const result = parseMessageContent(text)
      expect(result.reasoning).toBe('analysis steps')
      expect(result.content).toBe('final answer')
    })

    it('parses unclosed (streaming) think tag', () => {
      const text = '<think>still reasoning...'
      const result = parseMessageContent(text)
      expect(result.reasoning).toBe('still reasoning...')
      expect(result.content).toBe('')
    })

    it('preserves text before think tag', () => {
      const text = 'prefix text<think>reasoning</think>\nsuffix answer'
      const result = parseMessageContent(text)
      expect(result.reasoning).toBe('reasoning')
      expect(result.content).toBe('prefix text\nsuffix answer')
    })

    it('handles empty think content', () => {
      const text = '<think></think>\nanswer only'
      const result = parseMessageContent(text)
      expect(result.content).toContain('answer only')
    })

    it('is case insensitive for THINK', () => {
      const text = '<THINK>uppercase reasoning</THINK>\nanswer'
      const result = parseMessageContent(text)
      expect(result.reasoning).toBe('uppercase reasoning')
      expect(result.content).toBe('answer')
    })
  })

  describe('English Thought/Answer pattern', () => {
    it('parses Thought/Answer format', () => {
      const text = 'Thought: Let me think about this carefully\nAnswer: The result is 42'
      const result = parseMessageContent(text)
      expect(result.reasoning).toBe('Let me think about this carefully')
      expect(result.content).toBe('The result is 42')
    })

    it('parses Reasoning/Answer format', () => {
      const text = 'Reasoning: Step by step analysis here\nAnswer: Conclusion'
      const result = parseMessageContent(text)
      expect(result.reasoning).toBe('Step by step analysis here')
      expect(result.content).toBe('Conclusion')
    })
  })

  describe('no separator', () => {
    it('plain text is not split', () => {
      const text = 'This is a simple answer'
      const result = parseMessageContent(text)
      expect(result.reasoning).toBe('')
      expect(result.content).toBe('This is a simple answer')
    })

    it('short first part with --- does not split', () => {
      const text = 'short\n---\nlong content here'
      const result = parseMessageContent(text)
      expect(result.reasoning).toBe('')
      expect(result.content).toBe(text)
    })

    it('long first part with --- splits correctly', () => {
      const text = 'Some reasoning that is long enough\n---\nThis is the much longer answer section with more text'
      const result = parseMessageContent(text)
      expect(result.reasoning).toBe('Some reasoning that is long enough')
      expect(result.content).toBe('This is the much longer answer section with more text')
    })
  })
})
