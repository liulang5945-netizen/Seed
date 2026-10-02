/** Windows system tray: the always-present way back to a hidden window and the explicit quit entry. */

/** Minimal native tray operations the shell needs; Electron supplies the real ones. */
export interface DesktopTraySurface {
  /** Register the left-click action that reopens the window. */
  onClick(listener: () => void): void
  /** Replace the hover tooltip. */
  setToolTip(text: string): void
  /** Replace the context menu with these plain items, in order. */
  setMenu(items: readonly TrayMenuItem[]): void
  /** Remove the icon and release the native handle. */
  destroy(): void
}

/** One context-menu entry: a labelled action, or a separator between groups. */
export type TrayMenuItem = { readonly label: string; readonly click: () => void } | { readonly type: 'separator' }

/** Copy the tray resolves on every relabel, so a language change reaches the live icon. */
export interface DesktopTrayMessages {
  /** Product name used as the hover tooltip. */
  readonly productName: string
  /** Menu entry that shows and focuses the primary window. */
  readonly openApplication: string
  /** Menu entry that stops the application through the ordinary quit path. */
  readonly quitApplication: string
}

/** Construction seams for one tray icon. */
export interface DesktopTrayOptions {
  /** Create the native surface; throws when the platform cannot show a tray. */
  create(): DesktopTraySurface
  /** Read the current shell locale. */
  locale(): { readonly messages: DesktopTrayMessages }
  /** Focus or recreate the primary window. */
  open(): void
  /** Request the ordinary quit path. */
  quit(): void
}

/**
 * Tray icon present for the whole run, not only while the window is hidden:
 * left-click reopens the window and the context menu carries the explicit quit.
 */
export class DesktopTray {
  private surface: DesktopTraySurface | undefined

  /** @param options - Surface factory, locale reader, and the open and quit actions. */
  constructor(private readonly options: DesktopTrayOptions) {
    const surface = options.create()
    this.surface = surface
    surface.onClick(() => { options.open() })
    this.relabel()
  }

  /** Rebuild the tooltip and context menu in the current locale. */
  relabel(): void {
    const surface = this.surface
    if (surface === undefined) return
    const { messages } = this.options.locale()
    surface.setToolTip(messages.productName)
    surface.setMenu([
      { label: messages.openApplication, click: () => { this.options.open() } },
      { type: 'separator' },
      { label: messages.quitApplication, click: () => { this.options.quit() } },
    ])
  }

  /** Remove the icon; called once the quit is confirmed so no dead icon outlives the process. */
  dispose(): void {
    const surface = this.surface
    this.surface = undefined
    surface?.destroy()
  }
}
