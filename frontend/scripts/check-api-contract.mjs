import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptDir = path.dirname(fileURLToPath(import.meta.url))
const frontendRoot = path.resolve(scriptDir, '..')
const srcRoot = path.join(frontendRoot, 'src')
const openapiPath = path.resolve(frontendRoot, '..', 'tests', 'snapshots', 'openapi_baseline.json')
const openapi = JSON.parse(fs.readFileSync(openapiPath, 'utf8'))
const apiPaths = openapi.paths || {}
const allowedGenericTransport = new Set([
  path.join('composables', 'apiClient.js'),
  path.join('composables', 'useWorkbenchProjection.js'),
])
const ignoredFiles = new Set([
  path.join('components', 'FileUploadQueue.vue'),
])

function collectFiles(dir, result = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const absolute = path.join(dir, entry.name)
    if (entry.isDirectory()) collectFiles(absolute, result)
    else if (/\.(?:js|vue)$/.test(entry.name)) result.push(absolute)
  }
  return result
}

function endpointCandidates(expression) {
  const match = expression.match(/\/api\/[A-Za-z0-9._~!:@\/-]+/)
  if (!match) return { staticPath: '', candidates: [] }
  const staticPath = match[0].replace(/\/+$/, '') || '/api'
  if (apiPaths[staticPath]) return { staticPath, candidates: [staticPath] }
  const candidates = Object.keys(apiPaths).filter((candidate) => (
    candidate.startsWith(`${staticPath}/`)
    && candidate.split('/').length === staticPath.split('/').length + 1
  ))
  return { staticPath, candidates }
}

function matchingDelimiter(text, start, opening, closing) {
  let depth = 0
  let quote = ''
  for (let index = start; index < text.length; index += 1) {
    const char = text[index]
    if (quote) {
      if (char === '\\') index += 1
      else if (char === quote) quote = ''
      continue
    }
    if (char === "'" || char === '"' || char === '`') {
      quote = char
      continue
    }
    if (char === opening) depth += 1
    else if (char === closing) {
      depth -= 1
      if (depth === 0) return index
    }
  }
  return -1
}

function extractCalls(content) {
  const calls = []
  const callPattern = /\b(?:authFetch|fetch|readJson|authFetchJSON)\s*\(/g
  for (const match of content.matchAll(callPattern)) {
    const open = content.indexOf('(', match.index)
    const end = matchingDelimiter(content, open, '(', ')')
    if (end < 0) continue
    const text = content.slice(open, end + 1)
    let index = 1
    while (/\s/.test(text[index] || '')) index += 1
    const argStart = index
    let argEnd = index
    let quote = ''
    let depth = 0
    for (; argEnd < text.length; argEnd += 1) {
      const char = text[argEnd]
      if (quote) {
        if (char === '\\') argEnd += 1
        else if (char === quote) quote = ''
        continue
      }
      if (char === "'" || char === '"' || char === '`') {
        quote = char
        continue
      }
      if (char === '(' || char === '{' || char === '[') depth += 1
      else if (char === ')' || char === '}' || char === ']') depth -= 1
      else if (char === ',' && depth === 0) break
    }
    calls.push({ text, firstArg: text.slice(argStart, argEnd).trim() })
  }
  return calls
}

function jsonObjectKeys(callText) {
  const marker = 'JSON.stringify'
  const markerIndex = callText.indexOf(marker)
  if (markerIndex < 0) return []
  const brace = callText.indexOf('{', markerIndex)
  if (brace < 0) return []
  const end = matchingDelimiter(callText, brace, '{', '}')
  if (end < 0) return []
  const body = callText.slice(brace + 1, end)
  const keys = []
  let depth = 0
  let quote = ''
  for (let index = 0; index < body.length; index += 1) {
    const char = body[index]
    if (quote) {
      if (char === '\\') index += 1
      else if (char === quote) quote = ''
      continue
    }
    if (char === "'" || char === '"' || char === '`') {
      quote = char
      continue
    }
    if (char === '{' || char === '[' || char === '(') {
      depth += 1
      continue
    }
    if (char === '}' || char === ']' || char === ')') {
      depth -= 1
      continue
    }
    if (depth !== 0 || (index > 0 && body[index - 1] !== ',' && !/^\s*$/.test(body.slice(0, index)))) continue
    const match = body.slice(index).match(/^\s*([A-Za-z_$][\w$]*)\s*(?=:|,|$)/)
    if (!match) continue
    keys.push(match[1])
    index += match[0].length - 1
  }
  return keys
}

function resolveSchema(schema) {
  if (!schema || typeof schema !== 'object') return null
  if (schema.$ref) {
    const name = schema.$ref.split('/').pop()
    return openapi.components?.schemas?.[name] || null
  }
  return schema
}

function bodyProperties(operation) {
  const schema = operation?.requestBody?.content?.['application/json']?.schema
  const resolved = resolveSchema(schema)
  if (!resolved || resolved.additionalProperties === true) return null
  return new Set(Object.keys(resolved.properties || {}))
}

const failures = []
const seenLiterals = new Set()
for (const absolute of collectFiles(srcRoot)) {
  const relative = path.relative(srcRoot, absolute)
  if (relative.split(path.sep).includes('__tests__') || ignoredFiles.has(relative)) continue
  const content = fs.readFileSync(absolute, 'utf8')

  for (const call of extractCalls(content)) {
    const { candidates } = endpointCandidates(call.firstArg)
    if (!candidates.length || candidates.length !== 1) continue
    const literalKey = `${relative}:${call.firstArg}`
    if (!seenLiterals.has(literalKey)) seenLiterals.add(literalKey)
    const operationPath = candidates[0]
    const method = (call.text.match(/\bmethod\s*:\s*['"]([A-Za-z]+)['"]/i)?.[1] || 'GET').toLowerCase()
    const operation = apiPaths[operationPath]?.[method]
    if (!operation) {
      failures.push(`${relative}: ${method.toUpperCase()} ${operationPath} is absent from OpenAPI snapshot`)
      continue
    }

    const queryNames = [...call.firstArg.matchAll(/[?&]([A-Za-z_][\w-]*)=/g)].map((item) => item[1])
    const allowedQuery = new Set((operation.parameters || [])
      .filter((item) => item.in === 'query')
      .map((item) => item.name))
    for (const queryName of queryNames) {
      if (!allowedQuery.has(queryName)) {
        failures.push(`${relative}: ${method.toUpperCase()} ${operationPath} uses undeclared query parameter ${queryName}`)
      }
    }

    const properties = bodyProperties(operation)
    if (properties) {
      for (const bodyKey of jsonObjectKeys(call.text)) {
        if (!properties.has(bodyKey)) {
          failures.push(`${relative}: ${method.toUpperCase()} ${operationPath} uses undeclared JSON field ${bodyKey}`)
        }
      }
    }
  }

  // A few UI props/assignments carry a request template before passing it to
  // a helper.  Validate those real endpoint expressions, but do not scan
  // arbitrary status/source labels such as `/api/runtime/status.health`.
  const embeddedExpressions = [
    ...content.matchAll(/API_BASE[^`'"\n]{0,120}(\/api\/[A-Za-z0-9._~!:@\/-]+)/g),
    ...content.matchAll(/upload-endpoint\s*=\s*["'](\/api\/[A-Za-z0-9._~!:@\/-]+)/g),
  ]
  for (const match of embeddedExpressions) {
    const raw = match[1]
    const { staticPath, candidates } = endpointCandidates(raw)
    const key = `${relative}:${staticPath}`
    if (seenLiterals.has(key)) continue
    seenLiterals.add(key)
    if (!candidates.length) failures.push(`${relative}: endpoint ${staticPath} is absent from OpenAPI snapshot`)
  }

  if (allowedGenericTransport.has(relative)) continue
}

if (failures.length) {
  console.error('[api-contract] FAIL')
  for (const failure of failures) console.error(`- ${failure}`)
  process.exit(1)
}

console.log(`[api-contract] PASS: ${seenLiterals.size} API literals match OpenAPI paths; direct call methods/query/body fields validated`)
