/**
 * Markdown 渲染 composable
 */
import { ref } from 'vue';
import { Marked } from 'marked';
import { markedHighlight } from 'marked-highlight';
import hljs from 'highlight.js/lib/core';
import 'highlight.js/styles/atom-one-dark.css';
import DOMPurify from 'dompurify';
import { HLJS_ALIASES } from './hljsAliases.js';

// 仅允许安全字符，防止模型输出 lang 注入到 class 属性（XSS）
function sanitizeLang(lang) {
  if (!lang) return 'text'
  const cleaned = String(lang).replace(/[^a-zA-Z0-9_-]/g, '')
  return cleaned || 'text'
}

// highlight.js 只装 core（74 KB），192 种语法按需动态加载。
// grammarVersion 让「语法到位」成为响应式信号：调用方 watch 它即可重渲染。
export const grammarVersion = ref(0);
const loading = new Set();
const failed = new Set();

// hljs 内部语言名只含 [a-z0-9_+#.-]，别名表已在构建期生成
function resolveGrammar(lang) {
  const key = String(lang || '').toLowerCase();
  if (!key) return '';
  if (!/^[a-z0-9_+#.-]+$/.test(key)) return '';
  return HLJS_ALIASES[key] || key;
}

// es/languages 下有 384 个文件：192 个真实语法 + 192 个 `<name>.js.js` 兼容 shim。
// 动态 import 的模板字符串会被 Rollup 当 `*.js` 展开，连 shim 一起产出 192 个死 chunk
// （实测 54 KB 永不加载的垃圾）。用 glob 的否定模式精确排除。
const GRAMMAR_MODULES = import.meta.glob([
  '../../node_modules/highlight.js/es/languages/*.js',
  '!../../node_modules/highlight.js/es/languages/*.js.js',
]);
const GRAMMAR_PREFIX = '../../node_modules/highlight.js/es/languages/';

function ensureGrammar(name) {
  if (!name || hljs.getLanguage(name) || loading.has(name) || failed.has(name)) return;
  const loader = GRAMMAR_MODULES[`${GRAMMAR_PREFIX}${name}.js`];
  if (!loader) {
    failed.add(name);
    return;
  }
  loading.add(name);
  loader()
    .then((mod) => {
      const def = mod.default || mod;
      if (typeof def !== 'function') throw new Error('bad grammar module');
      hljs.registerLanguage(name, def);
      grammarVersion.value += 1;
    })
    .catch(() => {
      // 未知语言（模型可能输出任意 fence 标记）：记下来，不再重试
      failed.add(name);
    })
    .finally(() => {
      loading.delete(name);
    });
}

const marked = new Marked(
  markedHighlight({
    langPrefix: 'hljs language-',
    highlight(code, lang) {
      const name = resolveGrammar(lang);
      if (!name) return escapeHtml(code);
      if (hljs.getLanguage(name)) {
        return hljs.highlight(code, { language: name, ignoreIllegals: true }).value;
      }
      // 语法未到位：先返回已转义的原文，加载完成后由 grammarVersion 驱动重渲染
      ensureGrammar(name);
      return escapeHtml(code);
    }
  })
);

// markedHighlight 会把 highlight() 的返回值标记为 escaped=true 并跳过自身转义，
// 所以未高亮分支必须自己转义，否则代码块里的 <div> 会变成真节点。
const ESCAPE_MAP = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (ch) => ESCAPE_MAP[ch]);
}

marked.use({
  renderer: {
    // marked v13+ 传入 token 对象；旧的位置参数写法会渲染成 [object Object]
    code(token) {
      const raw = typeof token === 'object' && token !== null ? token.lang : token;
      const language = sanitizeLang(raw);
      const text = typeof token === 'object' && token !== null ? token.text : '';
      const body = (typeof token === 'object' && token !== null && token.escaped)
        ? text
        : escapeHtml(text);
      return `<div class="code-block-wrapper">
  <div class="code-header">
    <span class="code-lang">${language}</span>
    <button class="code-copy-btn">📋 复制</button>
  </div>
  <pre><code class="hljs language-${language}">${body}</code></pre>
  </div>`;
    }
  }
});

// 安全复制
const safeCopyText = async (text) => {
  if (navigator.clipboard && window.isSecureContext) {
    try {
      await navigator.clipboard.writeText(text);
      return;
    } catch (e) { /* ignore and fallback */ }
  }
  const textArea = document.createElement("textarea");
  textArea.value = text;
  textArea.style.position = "fixed";
  textArea.style.top = "-999999px";
  document.body.appendChild(textArea);
  textArea.focus();
  textArea.select();
  try { document.execCommand('copy'); } catch (err) {}
  document.body.removeChild(textArea);
};

// 事件委托：处理代码块复制按钮点击
// 注意：不能用 data-* 做钩子——purifyConfig 里 ALLOW_DATA_ATTR: false 会把它清掉，
// 曾导致复制按钮在生产环境完全点不动。这里统一以 .code-copy-btn 类为锚点。
if (typeof document !== 'undefined' && !window.__taijiMarkdownCopyHandler) {
  window.__taijiMarkdownCopyHandler = true;
  document.addEventListener('click', async (e) => {
    const btn = e.target?.closest?.('.code-copy-btn');
    if (!btn) return;
    const pre = btn.closest('.code-block-wrapper')?.querySelector('pre');
    if (!pre) return;
    try {
      await safeCopyText(pre.innerText);
      const oldText = btn.innerText;
      btn.innerText = '✅ 成功';
      setTimeout(() => { btn.innerText = oldText; }, 2000);
    } catch (err) {
      console.error('复制失败', err);
    }
  });
}

/**
 * 解析消息内容，分离思考过程和最终回答
 * 支持格式：
 *   1. 或 <think></think> 标签（大小写不敏感）
 *   2. 思考过程\n\n---\n\n回答内容
 *   3. 思考过程\n\n回答内容（当文本中出现"最终答案："等标记时）
 *   4. 中文"思考过程："等模式（含前导空白）
 * @param {string} text - 原始消息文本
 * @returns {{ reasoning: string, content: string }}
 */
function parseMessageContent(text) {
  if (!text) return { reasoning: '', content: '' };

  let reasoning = '';
  let content = text;

  // 1. 匹配 <think> 或 <推理> 标签（大小写不敏感，支持已闭合和流式未闭合）
  const thinkOpenMatch = text.match(/<(?:think|THINK|推理)>/i);
  if (thinkOpenMatch) {
    const openTag = thinkOpenMatch[0];
    const openIdx = thinkOpenMatch.index;
    const afterOpen = text.slice(openIdx + openTag.length);

    // 查找对应的闭合标签
    const thinkCloseMatch = afterOpen.match(/<\/(?:think|THINK|推理)>/i);
    if (thinkCloseMatch) {
      // 已闭合：正常拆分
      reasoning = afterOpen.slice(0, thinkCloseMatch.index).trim();
      const beforeThink = text.slice(0, openIdx);
      const afterThink = afterOpen.slice(thinkCloseMatch.index + thinkCloseMatch[0].length);
      content = (beforeThink + afterThink).trim();
      if (reasoning) return { reasoning, content };
    } else {
      // 流式未闭合：`<think>` 已打开但 `</think>` 尚未到达
      // 将思考内容暂存，正文置空（等待后续 chunk 填充闭合后正文）
      reasoning = afterOpen.trim();
      content = text.slice(0, openIdx).trim();
      return { reasoning, content: content || '' };
    }
  }

  // 3. 匹配 思考(?:过程)?：... 然后换行分隔的回答
  //    例如 "思考过程：...\n最终答案：..."（放宽行首锚点，允许前缀空白）
  const thoughtPatterns = [
    /思考[：:]\s*([\s\S]*?)(?=\n(?:最终)?(?:回答|答案)[：:]|\n---|\n\n\n|$)/,
    /思考过程[：:]\s*([\s\S]*?)(?=\n(?:最终)?(?:回答|答案)[：:]|\n---|\n\n\n|$)/,
    /^\s*[Tt]hought[：:]\s*([\s\S]*?)(?=\n\s*(?:[Aa]nswer|[Ff]inal)[：:]|\n---|\n\n\n|$)/m,
    /^\s*[Rr]easoning[：:]\s*([\s\S]*?)(?=\n\s*(?:[Aa]nswer|[Ff]inal)[：:]|\n---|\n\n\n|$)/m,
  ];

  for (const pattern of thoughtPatterns) {
    const match = text.match(pattern);
    if (match && match[1] && match[1].trim().length > 5) {
      reasoning = match[1].trim();
      const after = text.slice(match.index + match[0].length);
      // after 以 `\n` 开头（lookahead 未消费分隔符），所以必须允许前导空白，
      // 否则「最终答案：」/「Answer:」这类前缀会残留在正文里。
      content = after
        .replace(/^\s*(?:(?:最终)?(?:回答|答案)|[Aa]nswer|[Ff]inal)[：:]\s*/, '')
        .replace(/^\s*---\s*\n?/, '')
        .trim();
      if (reasoning) return { reasoning, content };
    }
  }

  // 4. 匹配 "Reasoning: ... Answer: ..." 格式（英文模型常见输出）
  const engMatch = text.match(/^Reasoning:\s*([\s\S]*?)\nAnswer:\s*([\s\S]*)$/im);
  if (engMatch) {
    return { reasoning: engMatch[1].trim(), content: engMatch[2].trim() };
  }

  // 5. 匹配 "### 思考过程" 或 "## 思考" 等 markdown 标题分隔
  const headerMatch = text.match(/^#{1,3}\s*思考(?:过程)?\s*\n([\s\S]*?)(?=\n#{1,3}\s*(?:最终)?(?:回答|答案)|\n---|$)/m);
  if (headerMatch) {
    reasoning = headerMatch[1].trim();
    content = text.slice(headerMatch.index + headerMatch[0].length).replace(/^#{1,3}\s*(?:最终)?(?:回答|答案)[：:]?\s*/m, '').trim();
    if (reasoning) return { reasoning, content };
  }

  // 6. 没有找到分隔标记，将文本按第一个 \n---\n 或 \n\n\n 分割尝试
  const sepMatch = text.match(/^([\s\S]*?)\n---\n([\s\S]*)$/);
  if (sepMatch) {
    const first = sepMatch[1].trim();
    const second = sepMatch[2].trim();
    // 只有当第一部分看起来像思考过程时才拆分
    if (first.length < second.length * 1.5 && first.length > 20) {
      return { reasoning: first, content: second };
    }
  }

  // 默认：无思考过程
  return { reasoning: '', content: text };
}

export function useMarkdown() {
  // DOMPurify 配置：允许 img 标签及常见图片属性
  const purifyConfig = {
    ADD_TAGS: ['img'],
    ADD_ATTR: ['src', 'alt', 'width', 'height', 'loading', 'decoding', 'title'],
    ALLOW_DATA_ATTR: false,
  };

  const renderMarkdown = (text) => {
    if (!text) return '';
    // 读一次 grammarVersion：模板中同步调用时 Vue 会把它记为渲染依赖，
    // 语法模块异步到位后自增即触发重渲染，调用方无需改动。
    void grammarVersion.value;
    const raw = marked.parse(text);
    return typeof raw === 'string' ? DOMPurify.sanitize(raw, purifyConfig) : raw;
  };

  const formatDuration = (seconds) => {
    if (!seconds || seconds <= 0) return '-';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    if (h > 0) return h + 'h ' + m + 'm';
    if (m > 0) return m + 'm ' + s + 's';
    return s + 's';
  };

  return { renderMarkdown, formatDuration, parseMessageContent };
}
