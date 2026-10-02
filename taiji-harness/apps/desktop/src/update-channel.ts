/**
 * The one update channel this fork publishes on. The Desktop feed, the local
 * channel under the Harness home, and the release tooling all name the same
 * channel file, so the name lives here once — free of Electron imports, so
 * packaging scripts can read it too.
 */

/** Channel name electron-updater selects for this product. */
export const DESKTOP_UPDATE_CHANNEL = 'nightly'

/**
 * Channel file electron-updater requests for one platform.
 * @param platform - Node platform name of the running build.
 * @returns the channel file name inside the feed.
 */
export function desktopChannelFile(platform: string): string {
  return platform === 'darwin' ? `${DESKTOP_UPDATE_CHANNEL}-mac.yml` : `${DESKTOP_UPDATE_CHANNEL}.yml`
}
