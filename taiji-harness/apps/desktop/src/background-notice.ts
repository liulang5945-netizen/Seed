/** One-time Windows confirmation before hiding the application in the system tray. */

import { existsSync, mkdirSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'

/** Copy the confirmation resolves on every prompt, so a language change reaches it. */
export interface DesktopBackgroundNoticeMessages {
  /** Product name used as the dialog title. */
  readonly productName: string
  /** One-sentence explanation of what hiding the window does to running work. */
  readonly backgroundNoticeBody: string
  /** The single acknowledgement button. */
  readonly backgroundNoticeConfirm: string
}

/** Native dialog request this notice issues; a subset of Electron's message box. */
export interface DesktopBackgroundNoticeDialogOptions {
  readonly type: 'info'
  readonly title: string
  readonly message: string
  readonly buttons: readonly string[]
  readonly defaultId: number
  readonly cancelId: number
}

/** Construction seams for the one-time confirmation. */
export interface DesktopBackgroundNoticeOptions {
  /** Marker file whose presence records the acknowledgement for later runs. */
  readonly markerPath: string
  /** Read the current shell locale. */
  locale(): { readonly messages: DesktopBackgroundNoticeMessages }
  /** Show the native information box and resolve with the clicked button index. */
  show(options: DesktopBackgroundNoticeDialogOptions): Promise<{ readonly response: number }>
  /** Focus the window while a prompt is already pending. */
  focus(): void
}

/**
 * Only an explicit acknowledgement permits the first hide; cancelled prompts
 * remain eligible, and the marker file keeps later runs prompt-free.
 */
export class DesktopBackgroundNotice {
  private acknowledged = false
  private pending = false
  private disposed = false

  /** @param options - Marker path, localized copy, and the dialog and focus actions. */
  constructor(private readonly options: DesktopBackgroundNoticeOptions) {}

  /**
   * Request a window hide, prompting until acknowledged and coalescing repeated requests.
   * @param hide - Hide the still-owned window after acknowledgement, or immediately when already recorded.
   */
  close(hide: () => void): void {
    if (this.disposed) return
    if (this.pending) {
      this.options.focus()
      return
    }
    if (this.acknowledged || existsSync(this.options.markerPath)) {
      hide()
      return
    }
    this.pending = true
    void this.confirm(hide)
  }

  /** Ignore late dialog responses after application shutdown begins. */
  dispose(): void {
    this.disposed = true
  }

  private async confirm(hide: () => void): Promise<void> {
    try {
      const { messages } = this.options.locale()
      const result = await this.options.show({
        type: 'info',
        title: messages.productName,
        message: messages.backgroundNoticeBody,
        buttons: [messages.backgroundNoticeConfirm],
        defaultId: 0,
        cancelId: -1,
      })
      if (this.disposed || result.response !== 0) return
      this.acknowledged = true
      try {
        mkdirSync(dirname(this.options.markerPath), { recursive: true })
        writeFileSync(this.options.markerPath, '')
      } catch (error) {
        console.warn('desktop tray: could not record background confirmation', error)
      }
      hide()
    } catch (error) {
      console.warn('desktop tray: background confirmation unavailable', error)
    } finally {
      this.pending = false
    }
  }
}
