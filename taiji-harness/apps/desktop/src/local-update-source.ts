/**
 * The local update channel of an unsigned install: a folder under the Harness
 * home that release tooling fills with one channel file and its installer,
 * served back to this process over a loopback listener so electron-updater's
 * generic provider reads it exactly like it reads a remote feed.
 *
 * Upstream ships a packaged `app-update.yml` and refuses to build unsigned
 * installs at all; this fork serves the same feed shape from a folder instead,
 * so "check for updates" works on a machine whose feed is a directory. The
 * folder without a release still answers — with the running version — so a
 * check reports "up to date" instead of a transport failure.
 */

import { createReadStream, existsSync, mkdirSync, statSync, writeFileSync } from 'node:fs'
import { createServer, type Server, type ServerResponse } from 'node:http'
import { join } from 'node:path'

/** Where the provider configuration is written, relative to the Electron user data directory. */
export const LOCAL_UPDATE_CONFIG_FILE = 'local-update.yml'

/** Name of the updater's own download cache directory under the user's cache root. */
const LOCAL_UPDATE_CACHE_DIR = 'seed-desktop-updater'

/** Everything one local channel needs to start. */
export interface LocalUpdateSourceOptions {
  /** Folder the channel serves; release tooling writes the channel file and installer here. */
  readonly directory: string
  /** Absolute path of the provider configuration this source writes for electron-updater. */
  readonly configPath: string
  /** Channel file electron-updater requests for this build (`nightly.yml` on Windows). */
  readonly channelFile: string
  /** Running application version; the synthesized channel reports it as the latest release. */
  readonly version: string
}

/** One running local update channel. */
export interface LocalUpdateSource {
  /** Feed URL the provider configuration points at. */
  readonly url: string
  /** Folder the channel serves. */
  readonly directory: string
  /** Stop the listener; the provider configuration file is left in place. */
  close(): Promise<void>
}

/**
 * Start the local channel on an ephemeral loopback port and write the provider
 * configuration electron-updater reads in place of a packaged `app-update.yml`.
 * @param options - folder, configuration destination, channel file, and version.
 * @returns the running source, with its feed URL and a close operation.
 */
export async function startLocalUpdateSource(options: LocalUpdateSourceOptions): Promise<LocalUpdateSource> {
  mkdirSync(options.directory, { recursive: true })
  const server = createServer((request, response) => { serve(request.url, request.method, response, options) })
  await listen(server)
  const address = server.address()
  if (address === null || typeof address === 'string') {
    server.close()
    throw new Error('local update source: the loopback listener has no port')
  }
  const url = `http://127.0.0.1:${String(address.port)}/`
  // electron-updater reads the provider, the feed URL, and its cache directory
  // name from this file; without it the updater has no packaged update source.
  writeFileSync(options.configPath, [
    'provider: generic',
    `url: ${url}`,
    `updaterCacheDirName: ${LOCAL_UPDATE_CACHE_DIR}`,
    '',
  ].join('\n'))
  let closed = false
  return {
    url,
    directory: options.directory,
    async close() {
      if (closed) return
      closed = true
      // The updater's Electron session keeps connections alive; close() alone
      // would wait for them to idle. Closing must never block a quit, so a
      // listener that already stopped is not an error.
      server.closeAllConnections()
      await new Promise<void>((resolve) => {
        server.close((error) => {
          if (error !== undefined && (error as NodeJS.ErrnoException).code !== 'ERR_SERVER_NOT_RUNNING') {
            console.warn('local update source: close failed', error)
          }
          resolve()
        })
      })
    },
  }
}

/** Serve one request: the channel file, one release file, or a 404. */
function serve(rawUrl: string | undefined, method: string | undefined, response: ServerResponse, options: LocalUpdateSourceOptions): void {
  if (method !== 'GET' && method !== 'HEAD') {
    response.writeHead(405).end()
    return
  }
  const pathname = new URL(rawUrl ?? '/', 'http://127.0.0.1').pathname
  let name: string
  try {
    name = decodeURIComponent(pathname).replace(/^\/+/u, '')
  } catch {
    response.writeHead(400).end()
    return
  }
  // One flat directory: a name carrying a separator is never a release file.
  if (name === '' || name.includes('/') || name.includes('\\')) {
    response.writeHead(404).end()
    return
  }
  if (name === options.channelFile) {
    const channel = join(options.directory, options.channelFile)
    if (!existsSync(channel)) {
      // No release has been dropped in yet: answer as a feed whose latest
      // version is the running one, which the updater reports as up to date.
      response.writeHead(200, { 'content-type': 'text/yaml' })
      response.end(`version: ${options.version}\nfiles: []\n`)
      return
    }
    sendFile(channel, response, method === 'HEAD')
    return
  }
  sendFile(join(options.directory, name), response, method === 'HEAD')
}

/** Send one flat file with its length; a missing path or a directory answers 404. */
function sendFile(path: string, response: ServerResponse, headOnly: boolean): void {
  const stats = statSync(path, { throwIfNoEntry: false })
  if (stats === undefined || !stats.isFile()) {
    response.writeHead(404).end()
    return
  }
  response.writeHead(200, {
    'content-type': path.endsWith('.yml') ? 'text/yaml' : 'application/octet-stream',
    'content-length': String(stats.size),
  })
  if (headOnly) {
    response.end()
    return
  }
  const stream = createReadStream(path)
  stream.on('error', () => { response.destroy() })
  stream.pipe(response)
}

/** Bind the listener to an ephemeral loopback port. */
async function listen(server: Server): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    const failed = (error: Error): void => { reject(error) }
    server.once('error', failed)
    server.listen(0, '127.0.0.1', () => {
      server.off('error', failed)
      resolve()
    })
  })
  // A later listener failure must not take the whole application down.
  server.on('error', (error: Error) => { console.warn('local update source: listener failed', error) })
}
