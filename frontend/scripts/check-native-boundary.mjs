import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptDir = path.dirname(fileURLToPath(import.meta.url))
const srcRoot = path.resolve(scriptDir, '..', 'src')
const allowedLegacyArtifacts = new Set([
  path.join('components', 'FileUploadQueue.vue'),
])
const forbidden = [
  /\/api\/(?:rag|models?|agent|life|taiji)(?:\/|['"`])/i,
  /\/api\/system\/switch_model(?:\/|['"`])/i,
  /download_hf|gguf|huggingface|model_type|agent_max_iterations|agent_temperature/i,
]
const evidenceEntrypoints = {
  [path.join('views', 'LifeStatusView.vue')]: 'context="life"',
}
const evidenceExcludedEntrypoints = [
  path.join('components', 'ChatView.vue'),
  path.join('views', 'AgentConfigView.vue'),
  path.join('views', 'TrainingView.vue'),
  path.join('views', 'SettingsView.vue'),
  path.join('views', 'KBView.vue'),
]

function collectFiles(dir, result = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const absolute = path.join(dir, entry.name)
    if (entry.isDirectory()) collectFiles(absolute, result)
    else if (/\.(?:js|vue)$/.test(entry.name)) result.push(absolute)
  }
  return result
}

const failures = []
for (const absolute of collectFiles(srcRoot)) {
  const relative = path.relative(srcRoot, absolute)
  if (relative.split(path.sep).includes('__tests__')) continue
  if (allowedLegacyArtifacts.has(relative)) continue
  const content = fs.readFileSync(absolute, 'utf8')
  for (const pattern of forbidden) {
    if (pattern.test(content)) {
      failures.push(`${relative}: forbidden native-boundary residue ${pattern}`)
      break
    }
  }
}

for (const [relative, marker] of Object.entries(evidenceEntrypoints)) {
  const absolute = path.join(srcRoot, relative)
  const content = fs.readFileSync(absolute, 'utf8')
  if (!content.includes('RuntimeEvidenceStrip') || !content.includes(marker)) {
    failures.push(`${relative}: missing RuntimeEvidenceStrip ${marker}`)
  }
}

for (const relative of evidenceExcludedEntrypoints) {
  const absolute = path.join(srcRoot, relative)
  const content = fs.readFileSync(absolute, 'utf8')
  if (content.includes('RuntimeEvidenceStrip')) {
    failures.push(`${relative}: RuntimeEvidenceStrip belongs only to LifeStatusView`)
  }
}

if (failures.length) {
  console.error('[native-boundary] FAIL')
  for (const failure of failures) console.error(`- ${failure}`)
  process.exit(1)
}

console.log(`[native-boundary] PASS: ${Object.keys(evidenceEntrypoints).length} authoritative evidence entrypoint and Legacy boundary clean`)
