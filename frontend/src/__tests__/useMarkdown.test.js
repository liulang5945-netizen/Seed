import { describe, it, expect } from 'vitest'
import { useMarkdown } from '@/composables/useMarkdown.js'

/**
 * useMarkdown 真实模块测试
 *
 * 历史教训：本文件曾复制一份「简化版」parseMessageContent 自测自答，
 * 于是 renderMarkdown 把所有代码块渲染成 [object Object] 期间测试全绿。
 * 现在只允许测真实导出。
 */

const { renderMarkdown, parseMessageContent, formatDuration } = useMarkdown()

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
      const result = parseMessageContent('<think>analysis steps</think>\nfinal answer')
      expect(result.reasoning).toBe('analysis steps')
      expect(result.content).toBe('final answer')
    })

    it('parses unclosed (streaming) think tag', () => {
      const result = parseMessageContent('<think>still reasoning...')
      expect(result.reasoning).toBe('still reasoning...')
      expect(result.content).toBe('')
    })

    it('preserves text before think tag', () => {
      const result = parseMessageContent('prefix text<think>reasoning</think>\nsuffix answer')
      expect(result.reasoning).toBe('reasoning')
      expect(result.content).toBe('prefix text\nsuffix answer')
    })

    it('handles empty think content', () => {
      expect(parseMessageContent('<think></think>\nanswer only').content).toContain('answer only')
    })

    it('is case insensitive for THINK', () => {
      const result = parseMessageContent('<THINK>uppercase reasoning</THINK>\nanswer')
      expect(result.reasoning).toBe('uppercase reasoning')
      expect(result.content).toBe('answer')
    })

    it('supports the Chinese 推理 tag', () => {
      const result = parseMessageContent('<推理>中文推理内容</推理>\n中文回答')
      expect(result.reasoning).toBe('中文推理内容')
      expect(result.content).toBe('中文回答')
    })
  })

  describe('label patterns', () => {
    it('parses Thought/Answer format without leaking the Answer prefix', () => {
      const result = parseMessageContent('Thought: Let me think about this carefully\nAnswer: The result is 42')
      expect(result.reasoning).toBe('Let me think about this carefully')
      expect(result.content).toBe('The result is 42')
    })

    it('parses Reasoning/Answer format', () => {
      const result = parseMessageContent('Reasoning: Step by step analysis here\nAnswer: Conclusion')
      expect(result.reasoning).toBe('Step by step analysis here')
      expect(result.content).toBe('Conclusion')
    })

    it('parses Chinese 思考过程/最终答案 without leaking the label', () => {
      const result = parseMessageContent('思考过程：先拆解题目再计算\n最终答案：结果是 42')
      expect(result.reasoning).toBe('先拆解题目再计算')
      expect(result.content).toBe('结果是 42')
    })

    it('parses markdown header separated reasoning', () => {
      const text = '## 思考\n逐步推导的中间过程\n## 回答\n最终结论'
      const result = parseMessageContent(text)
      expect(result.reasoning).toBe('逐步推导的中间过程')
      expect(result.content).toBe('最终结论')
    })
  })

  describe('no separator', () => {
    it('plain text is not split', () => {
      const result = parseMessageContent('This is a simple answer')
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

describe('renderMarkdown', () => {
  it('returns empty string for falsy input', () => {
    expect(renderMarkdown('')).toBe('')
    expect(renderMarkdown(null)).toBe('')
  })

  it('renders inline markdown', () => {
    const html = renderMarkdown('普通文本 `inline` **粗体**')
    expect(html).toContain('<code>inline</code>')
    expect(html).toContain('<strong>粗体</strong>')
  })

  describe('code fence (regression: marked v13+ passes a token object)', () => {
    it('emits the real code body, never [object Object]', () => {
      const html = renderMarkdown('```python\nx = 1\n```')
      expect(html).not.toContain('[object Object]')
      expect(html).toContain('x = 1')
    })

    it('labels the fence language instead of always "text"', () => {
      const html = renderMarkdown('```python\nx = 1\n```')
      expect(html).toContain('<span class="code-lang">python</span>')
    })

    it('falls back to "text" for a fence with no language', () => {
      const html = renderMarkdown('```\nplain block\n```')
      expect(html).toContain('<span class="code-lang">text</span>')
      expect(html).toContain('plain block')
    })

    it('keeps the copy button reachable by the delegation selector after sanitizing', () => {
      const html = renderMarkdown('```js\nconst a = 1\n```')
      const host = document.createElement('div')
      host.innerHTML = html
      const btn = host.querySelector('.code-copy-btn')
      expect(btn).not.toBeNull()
      // 委托处理器依赖 .code-block-wrapper > pre 这条链路取代码文本
      expect(btn.closest('.code-block-wrapper')?.querySelector('pre')).not.toBeNull()
    })

    it('does not rely on data-* hooks, which DOMPurify strips', () => {
      expect(renderMarkdown('```js\nconst a = 1\n```')).not.toContain('data-action')
    })
  })

  describe('escaping (regression: unloaded grammar must not emit live HTML)', () => {
    it('escapes angle brackets inside a code fence', () => {
      const html = renderMarkdown('```js\nif (a<b) return "<div>";\n```')
      expect(html).toContain('&lt;div&gt;')
      expect(html).not.toContain('return "<div>"')
    })

    it('escapes code for an unknown fence language', () => {
      const html = renderMarkdown('```definitely-not-a-language\n<img src=x>\n```')
      expect(html).not.toContain('<img src=x>')
      expect(html).toContain('&lt;img')
    })

    it('strips script tags from prose via DOMPurify', () => {
      const html = renderMarkdown('文本 <script>alert(1)</script> 结束')
      expect(html).not.toContain('<script')
    })
  })
})

describe('formatDuration', () => {
  it('returns - for non-positive input', () => {
    expect(formatDuration(0)).toBe('-')
    expect(formatDuration(null)).toBe('-')
  })

  it('formats seconds, minutes and hours', () => {
    expect(formatDuration(45)).toBe('45s')
    expect(formatDuration(125)).toBe('2m 5s')
    expect(formatDuration(3720)).toBe('1h 2m')
  })
})
