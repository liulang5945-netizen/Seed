/**
 * Publish one built Desktop release into the local update channel: copy the
 * installer and its blockmap into the Harness home's `updates` folder, write
 * the channel file electron-updater reads, and drop the release it supersedes.
 *
 * An unsigned install serves exactly that folder for "check for updates"
 * (`../src/local-update-source.ts`), so running this script is what makes the
 * next check offer the new version. Only the newest release stays in the
 * folder: an installer carries the whole application, and the older one is
 * never served again.
 *
 * The build itself is the ordinary release build with a build version
 * (`DSH_DESKTOP_BUILD_VERSION`, e.g. `0.1.7-alpha.1.20261002.1`), which ranks
 * above the product version it extends.
 *
 * Usage: `pnpm --filter @taiji/dsh-desktop run publish:local:win:x64`
 */

import { createHash } from 'node:crypto'
import { createReadStream, copyFileSync, existsSync, mkdirSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { resolveDshHome } from '@taiji/dsh-home-paths'
import { desktopChannelFile } from '../src/update-channel.ts'
import { desktopUpdateMetadataFilename } from './desktop-auto-update-environment.mjs'
import { desktopTargetBuildPaths, resolveDesktopBuildTarget } from './desktop-build-paths.mjs'

/** Names one release contributes to the channel folder; everything else there is swept. */
const RELEASE_FILE = /^seed-.*\.exe(?:\.blockmap)?$/u

/**
 * Publish the target's single installer into the local channel.
 * @param argv - optional target name (`win-x64`); defaults to this host's target.
 */
async function main(argv: readonly string[]): Promise<void> {
  const target = argv[2] ?? resolveDesktopBuildTarget(process.env)
  if (target !== 'win-x64') {
    // The local channel exists for unsigned Windows installs; other targets publish signed feeds.
    throw new Error(`publish-local-update: ${target} has no local channel`)
  }
  const paths = desktopTargetBuildPaths(target)
  const source = existsSync(paths.unsignedArtifacts) ? paths.unsignedArtifacts : paths.artifacts
  // Older builds stay beside the newest one in the output directory; the
  // release this script publishes is the one the run just wrote.
  const installers = readdirSync(source)
    .filter(name => /^seed-.*-win-x64\.exe$/u.test(name))
    .sort((left, right) => statSync(join(source, right)).mtimeMs - statSync(join(source, left)).mtimeMs)
  if (installers.length === 0) {
    throw new Error(`publish-local-update: no installer in ${source}`)
  }
  const installer = installers[0]!
  const version = installer.replace(/^seed-/u, '').replace(/-win-x64\.exe$/u, '')
  const channelFile = desktopChannelFile('win32')
  // The app serves what this script writes; the two namings must agree.
  if (channelFile !== desktopUpdateMetadataFilename(version, 'win32')) {
    throw new Error(`publish-local-update: channel file ${channelFile} does not match the release metadata naming`)
  }
  const blockmap = `${installer}.blockmap`
  const file = join(source, installer)
  const size = statSync(file).size
  const sha512 = await hashFile(file)

  const directory = join(resolveDshHome(), 'updates')
  mkdirSync(directory, { recursive: true })
  // Sweep first: the new copy must not be removed by its own sweep.
  for (const name of readdirSync(directory)) {
    if (name === installer || name === blockmap) continue
    if (RELEASE_FILE.test(name) || name.endsWith('.yml')) rmSync(join(directory, name), { force: true })
  }
  copyFileSync(file, join(directory, installer))
  if (existsSync(join(source, blockmap))) copyFileSync(join(source, blockmap), join(directory, blockmap))

  writeFileSync(join(directory, channelFile), [
    `version: ${version}`,
    'files:',
    `  - url: ${installer}`,
    `    sha512: ${sha512}`,
    `    size: ${size}`,
    `path: ${installer}`,
    `sha512: ${sha512}`,
    `releaseDate: '${new Date().toISOString()}'`,
    '',
  ].join('\n'))

  console.log(`publish-local-update: ${version} published to ${directory}`)
  console.log(`  installer: ${installer} (${(size / (1024 * 1024)).toFixed(1)} MiB, sha512 ${sha512.slice(0, 16)}…)`)
  console.log(`  channel:   ${channelFile}`)
}

/** Stream one file through SHA-512, the digest electron-updater verifies. */
async function hashFile(path: string): Promise<string> {
  const hash = createHash('sha512')
  await new Promise<void>((resolve, reject) => {
    const stream = createReadStream(path)
    stream.on('data', (chunk) => { hash.update(chunk as Buffer) })
    stream.on('error', reject)
    stream.on('end', resolve)
  })
  return hash.digest('base64')
}

await main(process.argv)
