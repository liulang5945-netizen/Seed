/** The local update channel: the folder an unsigned install reads its feed from. */
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { LOCAL_UPDATE_CONFIG_FILE, startLocalUpdateSource, type LocalUpdateSource } from '../src/local-update-source.ts'

const roots: string[] = []
const started: LocalUpdateSource[] = []

afterEach(async () => {
  await Promise.all(started.splice(0).map(async (source) => { await source.close() }))
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true })
})

/** Start one channel over a private temp root. */
async function channel(): Promise<{ readonly source: LocalUpdateSource; readonly root: string }> {
  const root = mkdtempSync(join(tmpdir(), 'desktop-local-update-'))
  roots.push(root)
  const source = await startLocalUpdateSource({
    directory: join(root, 'updates'),
    configPath: join(root, LOCAL_UPDATE_CONFIG_FILE),
    channelFile: 'nightly.yml',
    version: '1.0.0',
  })
  started.push(source)
  return { source, root }
}

describe('local update source', () => {
  it('writes the provider configuration and reports the running version while the folder is empty', async () => {
    const { source, root } = await channel()

    // The folder release tooling writes into exists, and the feed URL is the listener's.
    expect(existsSync(join(root, 'updates'))).toBe(true)
    const config = readFileSync(join(root, LOCAL_UPDATE_CONFIG_FILE), 'utf8')
    expect(config).toContain('provider: generic')
    expect(config).toContain(`url: ${source.url}`)
    expect(config).toContain('updaterCacheDirName: seed-desktop-updater')

    // No release yet: the channel answers as a feed whose latest version is the running one.
    const feed = await fetch(new URL('nightly.yml', source.url))
    expect(feed.status).toBe(200)
    expect(feed.headers.get('content-type')).toBe('text/yaml')
    expect(await feed.text()).toBe('version: 1.0.0\nfiles: []\n')
  })

  it('serves the release files a publisher drops in and nothing outside the folder', async () => {
    const { source, root } = await channel()
    writeFileSync(join(root, 'updates', 'nightly.yml'), 'version: 1.0.1\nfiles: []\n')
    writeFileSync(join(root, 'updates', 'seed-1.0.1-win-x64.exe'), Buffer.from([1, 2, 3, 4]))
    writeFileSync(join(root, 'secrets.txt'), 'outside the channel')

    const feed = await fetch(new URL('nightly.yml', source.url))
    expect(await feed.text()).toBe('version: 1.0.1\nfiles: []\n')
    const installer = await fetch(new URL('seed-1.0.1-win-x64.exe', source.url))
    expect(installer.headers.get('content-type')).toBe('application/octet-stream')
    expect(installer.headers.get('content-length')).toBe('4')
    expect(new Uint8Array(await installer.arrayBuffer())).toEqual(new Uint8Array([1, 2, 3, 4]))

    // A HEAD answer carries the length without a body.
    const head = await fetch(new URL('seed-1.0.1-win-x64.exe', source.url), { method: 'HEAD' })
    expect(head.headers.get('content-length')).toBe('4')
    expect(await head.text()).toBe('')

    // Anything else — a missing name, a path that would leave the folder, another verb — is refused.
    expect((await fetch(new URL('missing.exe', source.url))).status).toBe(404)
    expect((await fetch(`${source.url}..%2Fsecrets.txt`)).status).toBe(404)
    expect((await fetch(new URL('nightly.yml', source.url), { method: 'POST' })).status).toBe(405)
  })

  it('releases the listener when closed', async () => {
    const { source } = await channel()
    await source.close()
    await expect(fetch(new URL('nightly.yml', source.url))).rejects.toThrow()
  })
})
